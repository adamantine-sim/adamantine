/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef THERMAL_OPERATOR_BASE_HH
#define THERMAL_OPERATOR_BASE_HH

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/affine_constraints.h>

namespace adamantine
{
template <int dim, typename MemorySpaceType>
class ThermalOperatorBase
{
public:
  ThermalOperatorBase() = default;

  virtual ~ThermalOperatorBase() = default;

  /**
   * Return the dimension of the codomain (or range) space. To remember: the
   * matrix is of dimension m×n.
   */
  virtual dealii::types::global_dof_index m() const = 0;

  /**
   * Return the dimension of the domain space. To remember: the matrix is of
   * dimension m×n.
   */
  virtual dealii::types::global_dof_index n() const = 0;

  /**
   * Matrix-vector multiplication. This function applies the operator to the
   * vector src.
   * \param[in] src
   * \param[out] dst
   */
  virtual void
  vmult(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
        dealii::LA::distributed::Vector<double, MemorySpaceType> const &src)
      const = 0;

  /**
   * Matrix-vector multiplication and addition of the result to dst. This
   * function applies the operator to the vector src and add the result to the
   * vector dst.
   * \param[in] src
   * \param[inout] dst
   */
  virtual void
  vmult_add(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
            dealii::LA::distributed::Vector<double, MemorySpaceType> const &src)
      const = 0;

  virtual void
  reinit(dealii::DoFHandler<dim> const &dof_handler,
         dealii::AffineConstraints<double> const &affine_constraints,
         dealii::hp::QCollection<1> const &q_collection) = 0;

  virtual void compute_inverse_mass_matrix(
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<double> const &affine_constraints) = 0;

  virtual std::shared_ptr<
      dealii::LA::distributed::Vector<double, MemorySpaceType>>
  get_inverse_mass_matrix() const = 0;

  virtual void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const = 0;

  virtual void clear() = 0;

  virtual void get_state_from_material_properties() = 0;

  virtual void set_state_to_material_properties() = 0;

  virtual void set_material_deposition_orientation(
      std::vector<double> const &deposition_cos,
      std::vector<double> const &deposition_sin) = 0;

  virtual void set_time_and_source_height(double, double) = 0;
};
} // namespace adamantine
#endif
