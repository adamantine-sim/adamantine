/* SPDX-FileCopyrightText: Copyright (c) 2023 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef BODY_FORCE_HH
#define BODY_FORCE_HH

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

namespace adamantine
{
/**
 * Base class that describes the interface that body forces need to implement.
 */
template <int dim>
struct BodyForce
{
  /**
   * Evaluate the body force given a cell.
   */
  virtual dealii::Tensor<1, dim, double>
  eval(typename dealii::Triangulation<dim>::active_cell_iterator const
           &cell) = 0;
};

// Forward declaration
template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
class MaterialProperty;

/**
 * Gravity's body force.
 */
template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
class GravityForce final : public BodyForce<dim>
{
public:
  GravityForce(MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                MemorySpaceType> &material_properties);

  dealii::Tensor<1, dim, double>
  eval(typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
      final;

private:
  /**
   * Gravity in \f$m/s^2\f$
   */
  static double constexpr g = 9.80665;
  MaterialProperty<dim, n_materials, p_order, MaterialStates, MemorySpaceType>
      &_material_properties;
};
} // namespace adamantine

#endif
