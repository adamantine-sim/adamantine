/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_PHYSICS_TEMPLATES_HH
#define THERMAL_PHYSICS_TEMPLATES_HH

#include <ThermalOperator.hh>
#if defined(ADAMANTINE_HAVE_CUDA) && defined(__CUDACC__)
#include <ThermalOperatorDevice.hh>
#endif
#include <CubeHeatSource.hh>
#include <ElectronBeamHeatSource.hh>
#include <GoldakHeatSource.hh>
#include <ThermalPhysics.hh>

#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <algorithm>
#include <execution>

namespace adamantine
{
namespace
{
template <int dim, int fe_degree, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value,
              int> = 0>
dealii::LA::distributed::Vector<double, MemorySpaceType> vmult_and_scale(
    std::shared_ptr<ThermalOperatorBase<dim, MemorySpaceType>> thermal_operator,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &y,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &value,
    std::vector<Timer> &timers)
{
  // Apply the Thermal Operator.
  thermal_operator->vmult_add(value, y);

  // Multiply by the inverse of the mass matrix.
  value.scale(*thermal_operator->get_inverse_mass_matrix());

  timers[evol_time_eval_th_ph].stop();

  return value;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value,
              int> = 0>
void init_dof_vector(
    double const value,
    dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType> &vector)
{
  unsigned int const local_size = vector.local_size();
  for (unsigned int i = 0; i < local_size; ++i)
    vector.local_element(i) = value;
}

#ifdef ADAMANTINE_HAVE_CUDA
template <int dim, int fe_degree, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::CUDA>::value,
              int> = 0>
dealii::LA::distributed::Vector<double, MemorySpaceType> vmult_and_scale(
    std::shared_ptr<ThermalOperatorBase<dim, MemorySpaceType>> thermal_operator,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &y,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &value,
    std::vector<Timer> &timers)
{
  dealii::LA::distributed::Vector<double, MemorySpaceType> value_dev(
      value.get_partitioner());
  value_dev.import(value, dealii::VectorOperation::insert);

  // Apply the Thermal Operator.
  thermal_operator->vmult_add(value_dev, y);

  // Multiply by the inverse of the mass matrix.
  value_dev.scale(*thermal_operator->get_inverse_mass_matrix());

  timers[evol_time_eval_th_ph].stop();

  return value_dev;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::CUDA>::value,
              int> = 0>
void init_dof_vector(
    double const value,
    dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType> &vector)
{
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      vector_host(vector.get_partitioner());
  unsigned int const local_size = vector_host.local_size();
  for (unsigned int i = 0; i < local_size; ++i)
    vector_host.local_element(i) = value;

  vector.import(vector_host, dealii::VectorOperation::insert);
}
#endif
} // namespace

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::ThermalPhysics(
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    Geometry<dim> &geometry)
    : _geometry(geometry), _dof_handler(_geometry.get_triangulation())
{
  // Create the FECollection
  _fe_collection.push_back(dealii::FE_Q<dim>(fe_degree));
  _fe_collection.push_back(dealii::FE_Nothing<dim>());

  // Create the QCollection
  _q_collection.push_back(QuadratureType(fe_degree + 1));
  _q_collection.push_back(QuadratureType(1));

  // Create the material properties
  boost::property_tree::ptree const &material_database =
      database.get_child("materials");
  _material_database = material_database;
  _material_properties.reset(new MaterialProperty<dim>(
      communicator, _geometry.get_triangulation(), material_database));

  // Create the heat sources
  boost::property_tree::ptree const &source_database =
      database.get_child("sources");
  // PropertyTreeInput sources.n_beams
  unsigned int const n_beams = source_database.get<unsigned int>("n_beams");
  _heat_sources.resize(n_beams);
  for (unsigned int i = 0; i < n_beams; ++i)
  {
    // PropertyTreeInput sources.beam_X.type
    boost::property_tree::ptree const &beam_database =
        source_database.get_child("beam_" + std::to_string(i));
    std::string type = beam_database.get<std::string>("type");
    if (type == "goldak")
    {
      //_heat_sources[i] =
      // std::make_unique<GoldakHeatSource<dim>>(beam_database);
      _heat_sources[i] = std::make_shared<GoldakHeatSource<dim>>(beam_database);
    }
    else if (type == "electron_beam")
    {
      //_heat_sources[i] =
      //  std::make_unique<ElectronBeamHeatSource<dim>>(beam_database);
      _heat_sources[i] =
          std::make_shared<ElectronBeamHeatSource<dim>>(beam_database);
    }
    else if (type == "cube")
    {
      //_heat_sources[i] = std::make_unique<CubeHeatSource<dim>>(beam_database);
      _heat_sources[i] = std::make_shared<CubeHeatSource<dim>>(beam_database);
    }
    else
    {
      ASSERT_THROW(false, "Error: Beam type '" +
                              beam_database.get<std::string>("type") +
                              "' not recognized.");
    }
  }

  // Create the thermal operator
  if (std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value)
    _thermal_operator =
        std::make_shared<ThermalOperator<dim, fe_degree, MemorySpaceType>>(
            communicator, _material_properties, _heat_sources);
#if defined(ADAMANTINE_HAVE_CUDA) && defined(__CUDACC__)
  else
    _thermal_operator = std::make_shared<
        ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>>(
        communicator, _material_properties, _heat_sources);
#endif

  // Create the time stepping scheme
  boost::property_tree::ptree const &time_stepping_database =
      database.get_child("time_stepping");
  // PropertyTreeInput time_stepping.method
  std::string method = time_stepping_database.get<std::string>("method");
  std::transform(method.begin(), method.end(), method.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (method.compare("forward_euler") == 0)
    _time_stepping =
        std::make_unique<dealii::TimeStepping::ExplicitRungeKutta<LA_Vector>>(
            dealii::TimeStepping::FORWARD_EULER);
  else if (method.compare("rk_third_order") == 0)
    _time_stepping =
        std::make_unique<dealii::TimeStepping::ExplicitRungeKutta<LA_Vector>>(
            dealii::TimeStepping::RK_THIRD_ORDER);
  else if (method.compare("rk_fourth_order") == 0)
    _time_stepping =
        std::make_unique<dealii::TimeStepping::ExplicitRungeKutta<LA_Vector>>(
            dealii::TimeStepping::RK_CLASSIC_FOURTH_ORDER);
  else if (method.compare("heun_euler") == 0)
  {
    _time_stepping = std::make_unique<
        dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector>>(
        dealii::TimeStepping::HEUN_EULER);
    _embedded_method = true;
  }
  else if (method.compare("bogacki_shampine") == 0)
  {
    _time_stepping = std::make_unique<
        dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector>>(
        dealii::TimeStepping::BOGACKI_SHAMPINE);
    _embedded_method = true;
  }
  else if (method.compare("dopri") == 0)
  {
    _time_stepping = std::make_unique<
        dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector>>(
        dealii::TimeStepping::DOPRI);
    _embedded_method = true;
  }
  else if (method.compare("fehlberg") == 0)
  {
    _time_stepping = std::make_unique<
        dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector>>(
        dealii::TimeStepping::FEHLBERG);
    _embedded_method = true;
  }
  else if (method.compare("cash_karp") == 0)
  {
    _time_stepping = std::make_unique<
        dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector>>(
        dealii::TimeStepping::CASH_KARP);
    _embedded_method = true;
  }
  else if (method.compare("backward_euler") == 0)
  {
    _time_stepping =
        std::make_unique<dealii::TimeStepping::ImplicitRungeKutta<LA_Vector>>(
            dealii::TimeStepping::BACKWARD_EULER);
    _implicit_method = true;
  }
  else if (method.compare("implicit_midpoint") == 0)
  {
    _time_stepping =
        std::make_unique<dealii::TimeStepping::ImplicitRungeKutta<LA_Vector>>(
            dealii::TimeStepping::IMPLICIT_MIDPOINT);
    _implicit_method = true;
  }
  else if (method.compare("crank_nicolson") == 0)
  {
    _time_stepping =
        std::make_unique<dealii::TimeStepping::ImplicitRungeKutta<LA_Vector>>(
            dealii::TimeStepping::CRANK_NICOLSON);
    _implicit_method = true;
  }
  else if (method.compare("sdirk2") == 0)
  {
    _time_stepping =
        std::make_unique<dealii::TimeStepping::ImplicitRungeKutta<LA_Vector>>(
            dealii::TimeStepping::SDIRK_TWO_STAGES);
    _implicit_method = true;
  }

  if (_embedded_method == true)
  {
    // PropertyTreeInput time_steppping.coarsening_parameter
    double coarsen_param =
        time_stepping_database.get("coarsening_parameter", 1.2);
    // PropertyTreeInput time_steppping.refining_parameter
    double refine_param = time_stepping_database.get("refining_parameter", 0.8);
    // PropertyTreeInput time_stepping.min_time_step
    double min_delta = time_stepping_database.get("min_time_step", 1e-14);
    // PropertyTreeInput time_stepping.max_time_step
    double max_delta = time_stepping_database.get("max_time_step", 1e100);
    // PropertyTreeInput time_stepping.refining_tolerance
    double refine_tol = time_stepping_database.get("refining_tolerance", 1e-8);
    // PropertyTreeInput time_stepping.coarsening_tolerance
    double coarsen_tol =
        time_stepping_database.get("coarsening_tolerance", 1e-12);
    dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector> *embedded_rk =
        static_cast<
            dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector> *>(
            _time_stepping.get());
    embedded_rk->set_time_adaptation_parameters(coarsen_param, refine_param,
                                                min_delta, max_delta,
                                                refine_tol, coarsen_tol);
  }

  // If the time stepping scheme is implicit, set the parameters for the solver
  // and create the implicit operator.
  if (_implicit_method == true)
  {
    // PropertyTreeInput time_stepping.max_iteration
    _max_iter = time_stepping_database.get("max_iteration", 1000);
    // PropertyTreeInput time_stepping.tolerance
    _tolerance = time_stepping_database.get("tolerance", 1e-12);
    // PropertyTreeInput time_stepping.right_preconditioning
    _right_preconditioning =
        time_stepping_database.get("right_preconditioning", false);
    // PropertyTreeInput time_stepping.n_tmp_vectors
    _max_n_tmp_vectors = time_stepping_database.get("n_tmp_vectors", 30);
    // PropertyTreeInput time_stepping.newton_max_iteration
    unsigned int newton_max_iter =
        time_stepping_database.get("newton_max_iteration", 100);
    // PropertyTreeInput time_stepping.newton_tolerance
    double newton_tolerance =
        time_stepping_database.get("newton_tolerance", 1e-6);
    dealii::TimeStepping::ImplicitRungeKutta<LA_Vector> *implicit_rk =
        static_cast<dealii::TimeStepping::ImplicitRungeKutta<LA_Vector> *>(
            _time_stepping.get());
    implicit_rk->set_newton_solver_parameters(newton_max_iter,
                                              newton_tolerance);

    // PropertyTreeInput time_stepping.jfnk
    bool jfnk = time_stepping_database.get("jfnk", false);
    _implicit_operator = std::make_unique<ImplicitOperator<MemorySpaceType>>(
        _thermal_operator, jfnk);
  }

  // Set material on part of the domain
  // PropertyTreeInput geometry.material_height
  double const material_height = database.get("geometry.material_height", 1e9);
  for (auto const &cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    // If the center of the cell is below material_height, it contains material
    // otherwise it does not.
    if (cell->center()[axis<dim>::z] < material_height)
      cell->set_active_fe_index(0);
    else
      cell->set_active_fe_index(1);
  }
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
void ThermalPhysics<dim, fe_degree, MemorySpaceType,
                    QuadratureType>::setup_dofs()
{
  _dof_handler.distribute_dofs(_fe_collection);
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler,
                                                  locally_relevant_dofs);
  _affine_constraints.clear();
  _affine_constraints.reinit(locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler,
                                                  _affine_constraints);
  _affine_constraints.close();

  _thermal_operator->reinit(_dof_handler, _affine_constraints, _q_collection);

  // Update the current height of the object
  // Loop over the locally owned cells with an active FE index of zero
  for (auto const &cell : dealii::filter_iterators(
           _dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    // Loop over the faces
    for (auto face_index : dealii::GeometryInfo<dim>::face_indices())
    {
      _current_height = std::max(
          _current_height, cell->face(face_index)->center()[axis<dim>::z]);
    }
  }
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
void ThermalPhysics<dim, fe_degree, MemorySpaceType,
                    QuadratureType>::compute_inverse_mass_matrix()
{
  _thermal_operator->compute_inverse_mass_matrix(
      _dof_handler, _affine_constraints, _fe_collection);
  if (_implicit_method == true)
    _implicit_operator->set_inverse_mass_matrix(
        _thermal_operator->get_inverse_mass_matrix());
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
void ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    add_material(
        std::vector<
            typename dealii::DoFHandler<dim>::active_cell_iterator> const
            &elements_to_activate,
        double initial_temperature,
        dealii::LA::distributed::Vector<double, MemorySpaceType> &solution)
{
  std::vector<dealii::Vector<double>> data_to_transfer;
  unsigned int const dofs_per_cell = _dof_handler.get_fe().n_dofs_per_cell();
  dealii::Vector<double> cell_solution(dofs_per_cell);
  dealii::Vector<double> dummy_cell_solution(dofs_per_cell);
  for (auto &val : dummy_cell_solution)
  {
    val = std::numeric_limits<double>::infinity();
  }
  solution.update_ghost_values();
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if ((cell->is_locally_owned()) && (cell->active_fe_index() == 0))
    {
      cell->get_dof_values(solution, cell_solution);
      data_to_transfer.push_back(cell_solution);
    }
    else
    {
      data_to_transfer.push_back(dummy_cell_solution);
    }
  }

  // Activate elements by updating the fe_index
  for (auto const &cell : elements_to_activate)
  {
    cell->set_active_fe_index(0);
  }

  dealii::parallel::distributed::Triangulation<dim> &triangulation =
      dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
          const_cast<dealii::Triangulation<dim> &>(
              _dof_handler.get_triangulation()));
  triangulation.prepare_coarsening_and_refinement();
  dealii::parallel::distributed::CellDataTransfer<
      dim, dim, std::vector<dealii::Vector<double>>>
      cell_data_trans(triangulation);
  cell_data_trans.prepare_for_coarsening_and_refinement(data_to_transfer);
  triangulation.execute_coarsening_and_refinement();

  setup_dofs();

  // Update MaterialProperty DoFHandler and resize the state vectors
  _material_properties->reinit_dofs();

  // Recompute the inverse of the mass matrix
  compute_inverse_mass_matrix();

  initialize_dof_vector(initial_temperature, solution);
  std::vector<dealii::Vector<double>> transferred_data(
      triangulation.n_active_cells(), dealii::Vector<double>(dofs_per_cell));
  cell_data_trans.unpack(transferred_data);

  unsigned int cell_i = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if ((cell->is_locally_owned()) && (transferred_data[cell_i][0] !=
                                       std::numeric_limits<double>::infinity()))
    {
      cell->set_dof_values(transferred_data[cell_i], solution);
    }
    ++cell_i;
  }

  // Communicate the results.
  solution.compress(dealii::VectorOperation::min);

  // Set the value to the newly create DoFs. Here we need to be careful with the
  // hanging nodes. When there is a hanging node, the dofs at the vertices are
  // "doubled": there is a dof associated to the coarse cell and a dof
  // associated to the fine cell. The final value is decided by
  // AffineConstraints. Thus, we need to make sure that the newly activated
  // cells are at the same level than their neighbors.
  std::for_each(std::execution::par_unseq, solution.begin(), solution.end(),
                [&](double &val) {
                  if (val == std::numeric_limits<double>::infinity())
                  {
                    val = initial_temperature;
                  }
                });
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
double ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    evolve_one_time_step(
        double t, double delta_t,
        dealii::LA::distributed::Vector<double, MemorySpaceType> &solution,
        std::vector<Timer> &timers)
{
  auto eval = [&](double const t, LA_Vector const &y) {
    return evaluate_thermal_physics(t, y, timers);
  };
  auto id_m_Jinv = [&](double const t, double const tau, LA_Vector const &y) {
    return id_minus_tau_J_inverse(t, tau, y, timers);
  };

  double time = _time_stepping->evolve_one_time_step(eval, id_m_Jinv, t,
                                                     delta_t, solution);

  // If the method is embedded, get the next time step. Otherwise, just use the
  // current time step.
  if (_embedded_method == false)
    _delta_t_guess = delta_t;
  else
  {
    dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector> *embedded_rk =
        static_cast<
            dealii::TimeStepping::EmbeddedExplicitRungeKutta<LA_Vector> *>(
            _time_stepping.get());
    _delta_t_guess = embedded_rk->get_status().delta_t_guess;
  }

  // Return the time at the end of the time step. This may be different than
  // t+delta_t for embedded methods.
  return time;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
void ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    initialize_dof_vector(
        dealii::LA::distributed::Vector<double, MemorySpaceType> &vector) const
{
  _thermal_operator->initialize_dof_vector(vector);
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
void ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    initialize_dof_vector(
        double const value,
        dealii::LA::distributed::Vector<double, MemorySpaceType> &vector) const
{
  // Resize the vector
  _thermal_operator->initialize_dof_vector(vector);

  init_dof_vector<dim, fe_degree, MemorySpaceType>(value, vector);
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
dealii::LA::distributed::Vector<double, MemorySpaceType>
ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    evaluate_thermal_physics(
        double const t,
        dealii::LA::distributed::Vector<double, MemorySpaceType> const &y,
        std::vector<Timer> &timers) const
{

  timers[evol_time_eval_th_ph].start();

  // Do we still need this?
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> source(
      y.get_partitioner());
  source = 0.;

  _thermal_operator->update_time_and_height(t, _current_height);
  /*  
  // Compute the source term.
  dealii::hp::QCollection<dim> source_q_collection;
  source_q_collection.push_back(dealii::QGauss<dim>(fe_degree + 1));
  source_q_collection.push_back(dealii::QGauss<dim>(1));
  dealii::hp::FEValues<dim> hp_fe_values(_fe_collection, source_q_collection,
                                         dealii::update_quadrature_points |
                                             dealii::update_values |
                                             dealii::update_JxW_values);
  unsigned int const dofs_per_cell = _fe_collection.max_dofs_per_cell();
  unsigned int const n_q_points = source_q_collection.max_n_quadrature_points();
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
  dealii::Vector<double> cell_source(dofs_per_cell);

  // Loop over the locally owned cells with an active FE index of zero
  for (auto const &cell : dealii::filter_iterators(
           _dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    cell_source = 0.;
    hp_fe_values.reinit(cell);
    dealii::FEValues<dim> const &fe_values =
        hp_fe_values.get_present_fe_values();
    double const inv_rho_cp = _thermal_operator->get_inv_rho_cp(cell);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double quad_pt_source = 0.;
        dealii::Point<dim> const &q_point = fe_values.quadrature_point(q);
        for (auto &beam : _heat_sources)
          quad_pt_source += beam->value(q_point, t, _current_height);

        cell_source[i] += inv_rho_cp * quad_pt_source *
                          fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }
    cell->get_dof_indices(local_dof_indices);
    _affine_constraints.distribute_local_to_global(cell_source,
                                                   local_dof_indices, source);
  }
  source.compress(dealii::VectorOperation::add);
  */
  return vmult_and_scale<dim, fe_degree, MemorySpaceType>(_thermal_operator, y,
                                                          source, timers);
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
dealii::LA::distributed::Vector<double, MemorySpaceType>
ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    id_minus_tau_J_inverse(
        double const /*t*/, double const tau,
        dealii::LA::distributed::Vector<double, MemorySpaceType> const &y,
        std::vector<Timer> &timers) const
{
  timers[evol_time_J_inv].start();
  _implicit_operator->set_tau(tau);
  dealii::LA::distributed::Vector<double, MemorySpaceType> solution(
      y.get_partitioner());

  // TODO Add a geometric multigrid preconditioner.
  dealii::PreconditionIdentity preconditioner;

  dealii::SolverControl solver_control(_max_iter, _tolerance * y.l2_norm());
  // We need to inverse (I - tau M^{-1} J). While M^{-1} and J are SPD,
  // (I - tau M^{-1} J) is symmetric indefinite in the general case.
  typename dealii::SolverGMRES<
      dealii::LA::distributed::Vector<double, MemorySpaceType>>::AdditionalData
      additional_data(_max_n_tmp_vectors, _right_preconditioning);
  dealii::SolverGMRES<dealii::LA::distributed::Vector<double, MemorySpaceType>>
      solver(solver_control, additional_data);
  solver.solve(*_implicit_operator, solution, y, preconditioner);

  timers[evol_time_J_inv].stop();

  return solution;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
void ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    extract_stateful_material_properties(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &vector)
{
  _thermal_operator->extract_stateful_material_properties(vector);
}

} // namespace adamantine

#endif
