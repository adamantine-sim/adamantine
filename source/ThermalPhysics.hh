/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _THERMAL_PHYSICS_HH_
#define _THERMAL_PHYSICS_HH_

#include "ElectronBeam.hh"
#include "Geometry.hh"
#include "Physics.hh"
#include "ThermalOperator.hh"
#include <deal.II/base/time_stepping.h>
#include <boost/property_tree/ptree.hpp>

namespace adamantine
{

/**
 * This class takes care of building the linear operator and the
 * right-hand-side. Also used to evolve the system in time.
 */
template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
class ThermalPhysics : public Physics<NumberType>
{
public:
  ThermalPhysics(boost::mpi::communicator &communicator,
                 boost::property_tree::ptree const &database,
                 Geometry<dim> &geometry);

  /**
   * Reinit needs to be called everytime the mesh is modified.
   */
  void reinit();

  double
  evolve_one_time_step(double t, double delta_t,
                       dealii::LA::distributed::Vector<NumberType> &solution);

  double get_delta_t_guess() const;

  void
  initialize_dof_vector(dealii::LA::distributed::Vector<NumberType> &vector);

private:
  typedef typename dealii::LA::distributed::Vector<NumberType> LA_Vector;

  LA_Vector evaluate_thermal_physics(double const t, LA_Vector const &y) const;

  // For now, this is a dummy function which does nothing. It is only necessary
  // for implicit methods.
  LA_Vector id_minus_tau_J_inverse(double const t, double const tau,
                                   LA_Vector const &y) const;

  bool _embedded_method;
  double _delta_t_guess;
  Geometry<dim> &_geometry;
  dealii::FE_Q<dim> _fe;
  dealii::DoFHandler<dim> _dof_handler;
  dealii::ConstraintMatrix _constraint_matrix;
  QuadratureType _quadrature;
  std::shared_ptr<MaterialProperty> _material_properties;
  // Use unique_ptr due to a strange bug involving TBB, std::vector, and
  // dealii::FunctionParser.
  std::vector<std::unique_ptr<ElectronBeam<dim>>> _electron_beams;
  std::unique_ptr<ThermalOperator<dim, fe_degree, NumberType>>
      _thermal_operator;
  std::unique_ptr<dealii::TimeStepping::RungeKutta<LA_Vector>> _time_stepping;
};

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
inline double ThermalPhysics<dim, fe_degree, NumberType,
                             QuadratureType>::get_delta_t_guess() const
{
  return _delta_t_guess;
}

template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
inline void ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::
    initialize_dof_vector(dealii::LA::distributed::Vector<NumberType> &vector)
{
  _thermal_operator->get_matrix_free().initialize_dof_vector(vector);
}
}

#endif
