/* Copyright (c) 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <BodyForce.hh>
#include <MaterialProperty.hh>
#include <instantiation.hh>

namespace adamantine
{
template <int dim, typename MemorySpaceType>
GravityForce<dim, MemorySpaceType>::GravityForce(
    MaterialProperty<dim, MemorySpaceType> &material_properties)
    : _material_properties(material_properties)
{
}

template <int dim, typename MemorySpaceType>
dealii::Tensor<1, dim, double> GravityForce<dim, MemorySpaceType>::eval(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
{
  // Note that the density is independent of the temperature
  double density = _material_properties.get_mechanical_property(
      cell, StateProperty::density_s);
  dealii::Tensor<1, dim, double> body_force;
  body_force[axis<dim>::z] = -density * g;

  return body_force;
}
} // namespace adamantine

INSTANTIATE_DIM_HOST(GravityForce)
INSTANTIATE_DIM_DEVICE(GravityForce)
