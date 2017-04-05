/* Copyright (c) 2016 - 2017, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_PHYSICS_TEMPLATES_HH
#define THERMAL_PHYSICS_TEMPLATES_HH

#include "ThermalPhysics.hh"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <algorithm>

namespace adamantine
{
template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::ThermalPhysics(
    boost::mpi::communicator const &communicator,
    boost::property_tree::ptree const &database, Geometry<dim> &geometry)
    : _embedded_method(false), _implicit_method(false), _geometry(geometry),
      _fe(fe_degree), _dof_handler(_geometry.get_triangulation()),
      _quadrature(fe_degree + 1)
{
  // Create the material properties
  boost::property_tree::ptree const &material_database =
      database.get_child("materials");
  _material_properties.reset(new MaterialProperty<dim>(
      communicator, _geometry.get_triangulation(), material_database));

  // Create the electron beams
  boost::property_tree::ptree const &source_database =
      database.get_child("sources");
  unsigned int const n_beams = source_database.get<unsigned int>("n_beams");
  _electron_beams.resize(n_beams);
  for (unsigned int i = 0; i < n_beams; ++i)
  {
    boost::property_tree::ptree const &beam_database =
        source_database.get_child("beam_" + std::to_string(i));
    _electron_beams[i] = std::make_unique<ElectronBeam<dim>>(beam_database);
    _electron_beams[i]->set_max_height(_geometry.get_max_height());
  }

  // Create the thermal operator
  _thermal_operator =
      std::make_shared<ThermalOperator<dim, fe_degree, NumberType>>(
          communicator, _material_properties);

  // Create the time stepping scheme
  boost::property_tree::ptree const &time_stepping_database =
      database.get_child("time_stepping");
  std::string method = time_stepping_database.get<std::string>("method");
  std::transform(method.begin(), method.end(), method.begin(),
                 [](unsigned char c)
                 {
                   return std::tolower(c);
                 });
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
    _implicit_operator =
        std::make_unique<ImplicitOperator<NumberType>>(_thermal_operator, jfnk);
  }
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
void ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::setup_dofs()
{
  _dof_handler.distribute_dofs(_fe);
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler,
                                                  locally_relevant_dofs);
  _constraint_matrix.clear();
  _constraint_matrix.reinit(locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler,
                                                  _constraint_matrix);
  _constraint_matrix.close();

  _thermal_operator->setup_dofs(_dof_handler, _constraint_matrix, _quadrature);
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
void ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::reinit()
{
  _thermal_operator->reinit(_dof_handler, _constraint_matrix);
  if (_implicit_method == true)
    _implicit_operator->set_inverse_mass_matrix(
        _thermal_operator->get_inverse_mass_matrix());
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
double ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    evolve_one_time_step(double t, double delta_t,
                         dealii::LA::distributed::Vector<NumberType> &solution,
                         std::vector<Timer> &timers)
{
  // TODO: this assume that the material properties do no change during the time
  // steps. This is wrong and needs to be changed.
  timers[evol_time_eval_mat_prop].start();
  _thermal_operator->evaluate_material_properties(solution);
  timers[evol_time_eval_mat_prop].stop();

  double time = _time_stepping->evolve_one_time_step(
      std::bind(&ThermalPhysics<dim, fe_degree, NumberType,
                                QuadratureType>::evaluate_thermal_physics,
                this, std::placeholders::_1, std::placeholders::_2,
                std::ref(timers)),
      std::bind(&ThermalPhysics<dim, fe_degree, NumberType,
                                QuadratureType>::id_minus_tau_J_inverse,
                this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::ref(timers)),
      t, delta_t, solution);

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

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
void ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    initialize_dof_vector(
        NumberType const value,
        dealii::LA::distributed::Vector<NumberType> &vector) const
{
  // Resize the vector
  initialize_dof_vector(vector);

  // TODO this should be done in a matrix free fashion.
  // TODO this assumes that the material properties are constant
  dealii::QGauss<dim> quadrature(1);
  dealii::FEValues<dim> fe_values(_fe, quadrature, dealii::update_values);
  unsigned int const dofs_per_cell = _fe.dofs_per_cell;
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
  dealii::IndexSet local_elements = vector.locally_owned_elements();
  for (auto cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    // Compute the enthalpy
    dealii::LA::distributed::Vector<NumberType> dummy;
    // Cast to Triangulation<dim>::cell_iterator to access the material_id
    typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(cell);
    double const density =
        _material_properties->get(cell_tria, Property::density, dummy);
    double const specific_heat =
        _material_properties->get(cell_tria, Property::specific_heat, dummy);
    NumberType const enthalpy_value = value * density * specific_heat;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      dealii::types::global_dof_index const dof_index = local_dof_indices[i];
      if (local_elements.is_element(dof_index) == true)
        vector[dof_index] = enthalpy_value;
    }
  }
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
dealii::LA::distributed::Vector<NumberType>
ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    evaluate_thermal_physics(
        double const t, dealii::LA::distributed::Vector<NumberType> const &y,
        std::vector<Timer> &timers) const
{
  timers[evol_time_eval_th_ph].start();
  LA_Vector value(y.get_partitioner());
  value = 0.;

  // Compute the source term.
  for (auto &beam : _electron_beams)
    beam->set_time(t);
  dealii::QGauss<dim> source_quadrature(fe_degree + 1);
  dealii::FEValues<dim> fe_values(_fe, source_quadrature,
                                  dealii::update_quadrature_points |
                                      dealii::update_values |
                                      dealii::update_JxW_values);
  unsigned int const dofs_per_cell = _fe.dofs_per_cell;
  unsigned int const n_q_points = source_quadrature.size();
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
  dealii::Vector<NumberType> cell_source(dofs_per_cell);

  for (auto cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    cell_source = 0.;
    fe_values.reinit(cell);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double source = 0.;
        dealii::Point<dim> const &q_point = fe_values.quadrature_point(q);
        for (auto &beam : _electron_beams)
          source += beam->value(q_point);

        cell_source[i] +=
            source * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }
    cell->get_dof_indices(local_dof_indices);
    _constraint_matrix.distribute_local_to_global(cell_source,
                                                  local_dof_indices, value);
  }

  // Apply the Thermal Operator.
  _thermal_operator->vmult_add(value, y);

  // Multiply by the inverse of the mass matrix.
  value.scale(*_thermal_operator->get_inverse_mass_matrix());

  timers[evol_time_eval_th_ph].stop();

  return value;
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
dealii::LA::distributed::Vector<NumberType>
ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    id_minus_tau_J_inverse(double const /*t*/, double const tau,
                           dealii::LA::distributed::Vector<NumberType> const &y,
                           std::vector<Timer> &timers) const
{
  timers[evol_time_J_inv].start();
  _implicit_operator->set_tau(tau);
  dealii::LA::distributed::Vector<NumberType> solution(y.get_partitioner());

  // TODO Add a geometric multigrid preconditioner.
  dealii::PreconditionIdentity preconditioner;

  dealii::SolverControl solver_control(_max_iter, _tolerance * y.l2_norm());
  // We need to inverse (I - tau M^{-1} J). While M^{-1} and J are SPD,
  // (I - tau M^{-1} J) is symmetric indefinite in the general case.
  typename dealii::SolverGMRES<
      dealii::LA::distributed::Vector<NumberType>>::AdditionalData
      additional_data(_max_n_tmp_vectors, _right_preconditioning);
  dealii::SolverGMRES<dealii::LA::distributed::Vector<NumberType>> solver(
      solver_control, additional_data);
  solver.solve(*_implicit_operator, solution, y, preconditioner);

  timers[evol_time_J_inv].stop();

  return solution;
}
}

#endif
