/* SPDX-FileCopyrightText: Copyright (c) 2023 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <BodyForce.hh>
#include <MaterialProperty.hh>
#include <instantiation.hh>

namespace adamantine
{
template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
GravityForce<dim, p_order, MaterialStates, MemorySpaceType>::GravityForce(
    MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>
        &material_properties)
    : _material_properties(material_properties)
{
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::Tensor<1, dim, double>
GravityForce<dim, p_order, MaterialStates, MemorySpaceType>::eval(
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

INSTANTIATE_DIM_PORDER_MATERIALSTATES_HOST(GravityForce)
INSTANTIATE_DIM_PORDER_MATERIALSTATES_DEVICE(GravityForce)
