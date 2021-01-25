/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_PHYSICS_TEMPLATES_HH
#define THERMAL_PHYSICS_TEMPLATES_HH

#include <ThermalOperator.hh>
#ifdef ADAMANTINE_HAVE_CUDA
#include <ThermalOperatorDevice.hh>
#endif
#include <CubeHeatSource.hh>
#include <ElectronBeamHeatSource.hh>
#include <GoldakHeatSource.hh>
#include <ThermalPhysics.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <algorithm>

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
  unsigned int const n_beams = source_database.get<unsigned int>("n_beams");
  _heat_sources.resize(n_beams);
  for (unsigned int i = 0; i < n_beams; ++i)
  {
    boost::property_tree::ptree const &beam_database =
        source_database.get_child("beam_" + std::to_string(i));
    std::string type = beam_database.get<std::string>("type");
    if (type == "goldak")
    {
      _heat_sources[i] = std::make_unique<GoldakHeatSource<dim>>(beam_database);
    }
    else if (type == "electron_beam")
    {
      _heat_sources[i] =
          std::make_unique<ElectronBeamHeatSource<dim>>(beam_database);
    }
    else if (type == "cube")
    {
      _heat_sources[i] = std::make_unique<CubeHeatSource<dim>>(beam_database);
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
            communicator, _material_properties, material_database);
#ifdef ADAMANTINE_HAVE_CUDA
  else
    _thermal_operator = std::make_shared<
        ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>>(
        communicator, _material_properties);
#endif

  // Create the time stepping scheme
  boost::property_tree::ptree const &time_stepping_database =
      database.get_child("time_stepping");
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
    double coarsen_param =
        time_stepping_database.get("coarsening_parameter", 1.2);
    double refine_param = time_stepping_database.get("refining_parameter", 0.8);
    double min_delta = time_stepping_database.get("min_time_step", 1e-14);
    double max_delta = time_stepping_database.get("max_time_step", 1e100);
    double refine_tol = time_stepping_database.get("refining_tolerance", 1e-8);
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
    _max_iter = time_stepping_database.get("max_iteration", 1000);
    _tolerance = time_stepping_database.get("tolerance", 1e-12);
    _right_preconditioning =
        time_stepping_database.get("right_preconditioning", false);
    _max_n_tmp_vectors = time_stepping_database.get("n_tmp_vectors", 30);
    unsigned int newton_max_iter =
        time_stepping_database.get("newton_max_iteration", 100);
    double newton_tolerance =
        time_stepping_database.get("newton_tolerance", 1e-6);
    dealii::TimeStepping::ImplicitRungeKutta<LA_Vector> *implicit_rk =
        static_cast<dealii::TimeStepping::ImplicitRungeKutta<LA_Vector> *>(
            _time_stepping.get());
    implicit_rk->set_newton_solver_parameters(newton_max_iter,
                                              newton_tolerance);

    bool jfnk = time_stepping_database.get("jfnk", false);
    _implicit_operator = std::make_unique<ImplicitOperator<MemorySpaceType>>(
        _thermal_operator, jfnk);
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
  // Loop over the locally owned cells with an acttive FE index of zero
  for (auto const &cell : dealii::filter_iterators(
           _dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    // Loop over the faces
    for (auto face_index : dealii::GeometryInfo<dim>::face_indices())
    {
      _current_height =
          std::max(_current_height, cell->face(face_index)->center()[1]);
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
  /*
  timers[evol_time_eval_mat_prop].start();
  if (std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value)
    _thermal_operator->evaluate_material_properties(y);
  else
  {
    // TODO do this on the GPU
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> y_host(
        y.get_partitioner());
    y_host.import(y, dealii::VectorOperation::insert);
    _thermal_operator->evaluate_material_properties(y_host);
  }
  timers[evol_time_eval_mat_prop].stop();
  */
  timers[evol_time_eval_th_ph].start();
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> source(
      y.get_partitioner());
  source = 0.;

  // Compute the source term.

  // Currently assumes that the material is always "material_0" and that the
  // properties are scalar-valued
  double specific_heat_solid =
      _material_database.get<double>("material_0.solid.specific_heat");
  double specific_heat_liquid =
      _material_database.get<double>("material_0.liquid.specific_heat");
  double density_solid =
      _material_database.get<double>("material_0.solid.density");
  double density_liquid =
      _material_database.get<double>("material_0.liquid.density");

  double solidus = _material_database.get<double>("material_0.solidus");
  double liquidus = _material_database.get<double>("material_0.liquidus");
  double latent_heat = _material_database.get<double>("material_0.latent_heat");

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

    // double const inv_rho_cp = _thermal_operator->get_inv_rho_cp(cell);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double quad_pt_source = 0.;
        dealii::Point<dim> const &q_point = fe_values.quadrature_point(q);
        for (auto &beam : _heat_sources)
          quad_pt_source += beam->value(q_point, t, _current_height);

        // Calculate the inv_rho_cp
        double temperature = fe_values.shape_value(i, q);

        double liquid_ratio = -1.;
        if (temperature < solidus)
          liquid_ratio = 0.;
        else if (temperature > liquidus)
          liquid_ratio = 1.;
        else
        {
          liquid_ratio = (temperature - solidus) / (liquidus - solidus);

          specific_heat_liquid += latent_heat / (liquidus - solidus);
          specific_heat_solid += latent_heat / (liquidus - solidus);
        }

        // Calculate the local material properties
        double specific_heat = liquid_ratio * specific_heat_liquid +
                               (1.0 - liquid_ratio) * specific_heat_solid;

        double density = liquid_ratio * density_liquid +
                         (1.0 - liquid_ratio) * density_solid;

        double inv_rho_cp = 1.0 / (density * specific_heat);

        cell_source[i] += inv_rho_cp * quad_pt_source *
                          fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }
    cell->get_dof_indices(local_dof_indices);
    _affine_constraints.distribute_local_to_global(cell_source,
                                                   local_dof_indices, source);
  }
  source.compress(dealii::VectorOperation::add);

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
        dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
{
  _thermal_operator->extract_stateful_material_properties(vector);
}

} // namespace adamantine

#endif
