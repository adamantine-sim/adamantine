/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef THERMAL_OPERATOR_BASE_HH
#define THERMAL_OPERATOR_BASE_HH

#include <Operator.hh>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/affine_constraints.h>

namespace adamantine
{
template <int dim, typename MemorySpaceType>
class ThermalOperatorBase : public Operator<MemorySpaceType>
{
public:
  ThermalOperatorBase() = default;

  virtual ~ThermalOperatorBase() = default;

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
