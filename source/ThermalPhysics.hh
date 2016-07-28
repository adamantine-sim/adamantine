/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _THERMAL_PHYSICS_HH_
#define _THERMAL_PHYSICS_HH_

#include "Geometry.hh"
#include "ThermalOperator.hh"
#include <boost/property_tree/ptree.hpp>

namespace adamantine
{

/**
 * This class takes care of building the linear operator and the
 * right-hand-side. Also used to evolve the system in time.
 */
template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
class ThermalPhysics
{
public:
  ThermalPhysics(boost::mpi::communicator &communicator,
                 boost::property_tree::ptree const &database,
                 Geometry<dim> &geometry);

  /**
   * Reinit needs to be called everytime the mesh is modified.
   */
  void reinit();

  double evolve_one_time_step(double t, double delta_t,
                              dealii::LA::distributed::Vector<NumberType> &y);

private:
  Geometry<dim> &_geometry;
  dealii::FE_Q<dim> _fe;
  dealii::DoFHandler<dim> _dof_handler;
  dealii::ConstraintMatrix _constraint_matrix;
  QuadratureType _quadrature;
  std::unique_ptr<ThermalOperator<dim, fe_degree, NumberType>>
      _thermal_operator;
};
}

#endif
