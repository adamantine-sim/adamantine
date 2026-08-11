/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MECHANICAL_OPERATOR_DEVICE_BASE_HH
#define MECHANICAL_OPERATOR_DEVICE_BASE_HH

#include <MaterialProperty.hh>

#include <deal.II/lac/la_parallel_vector.h>

namespace adamantine
{
template <int dim, int n_materials, int p_order, typename MaterialStates>
class MechanicalOperatorDeviceBase
{
public:
  MechanicalOperatorDeviceBase() = default;

  virtual ~MechanicalOperatorDeviceBase() = default;

  virtual void
  reinit(dealii::DoFHandler<dim> const &dof_handler,
         dealii::AffineConstraints<double> const &affine_constraints) = 0;

  virtual void
  vmult(dealii::LinearAlgebra::distributed::Vector<
            double, dealii::MemorySpace::Default> &dst,
        dealii::LinearAlgebra::distributed::Vector<
            double, dealii::MemorySpace::Default> const &src) const = 0;

  virtual void
  vmult_add(dealii::LinearAlgebra::distributed::Vector<
                double, dealii::MemorySpace::Default> &dst,
            dealii::LinearAlgebra::distributed::Vector<
                double, dealii::MemorySpace::Default> const &src) const = 0;

  virtual void initialize_dof_vector(
      dealii::LinearAlgebra::distributed::Vector<
          double, dealii::MemorySpace::Default> &vector) const = 0;
};

} // namespace adamantine

#endif
