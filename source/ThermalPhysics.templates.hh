/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _THERMAL_PHYSICS_TEMPLATES_HH_
#define _THERMAL_PHYSICS_TEMPLATES_HH_

#include "ThermalPhysics.hh"
#include "MaterialProperty.hh"
#include <algorithm>

namespace adamantine
{
template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::ThermalPhysics(
    boost::mpi::communicator &communicator,
    boost::property_tree::ptree const &database, Geometry<dim> &geometry)
    : _embedded_method(false), _geometry(geometry), _fe(fe_degree),
      _dof_handler(_geometry.get_triangulation()), _quadrature(fe_degree + 1)
{
  // Create the material properties
  boost::property_tree::ptree const &material_database =
      database.get_child("materials");
  std::shared_ptr<MaterialProperty> material_properties(
      new MaterialProperty(material_database));

  // Create the thermal operator
  _thermal_operator =
      std::make_unique<ThermalOperator<dim, fe_degree, NumberType>>(
          communicator, material_properties);

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

  if (_embedded_method == true)
  {
    double coarsen_param =
        time_stepping_database.get("coarsening_parameter", 1.2);
    double refine_param = time_stepping_database.get("refining_parmeter", 0.8);
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
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
void ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::reinit()
{
  _dof_handler.distribute_dofs(_fe);
  // TODO: For now only homogeneous Neumann boundary conditions and uniform mesh
  _constraint_matrix.clear();
  _constraint_matrix.close();
  _thermal_operator->reinit(_dof_handler, _constraint_matrix, _quadrature);
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
double ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    evolve_one_time_step(double t, double delta_t,
                         dealii::LA::distributed::Vector<NumberType> &solution)
{
  double time = _time_stepping->evolve_one_time_step(
      std::bind(&ThermalPhysics<dim, fe_degree, NumberType,
                                QuadratureType>::evaluate_thermal_physics,
                this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&ThermalPhysics<dim, fe_degree, NumberType,
                                QuadratureType>::id_minus_tau_J_inverse,
                this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3),
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
dealii::LA::distributed::Vector<NumberType>
ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    evaluate_thermal_physics(
        double const t,
        dealii::LA::distributed::Vector<NumberType> const &y) const
{
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
dealii::LA::distributed::Vector<NumberType>
ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    id_minus_tau_J_inverse(
        double const /*t*/, double const /*tau*/,
        dealii::LA::distributed::Vector<NumberType> const &y) const
{
  // This function is not used since we don't allow implicit method. So just
  // return y.
  return y;
}
}

#endif
