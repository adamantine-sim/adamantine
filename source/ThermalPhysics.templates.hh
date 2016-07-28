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

namespace adamantine
{
template <int dim, int fe_degree, typename NumberType, typename QuadratureType>
ThermalPhysics<dim, fe_degree, NumberType, QuadratureType>::ThermalPhysics(
    boost::mpi::communicator &communicator,
    boost::property_tree::ptree const &database, Geometry<dim> &geometry)
    : _geometry(geometry), _fe(fe_degree),
      _dof_handler(_geometry.get_triangulation()), _quadrature(fe_degree + 1)
{
  boost::property_tree::ptree const &material_database =
      database.get_child("materials");
  std::shared_ptr<MaterialProperty> material_properties(
      new MaterialProperty(material_database));
  _thermal_operator =
      std::make_unique<ThermalOperator<dim, fe_degree, NumberType>>(
          communicator, material_properties);
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
}

#endif
