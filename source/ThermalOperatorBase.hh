/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_OPERATOR_BASE_HH
#define THERMAL_OPERATOR_BASE_HH

#include <Operator.hh>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>

namespace adamantine
{
template <int dim, typename MemorySpaceType>
class ThermalOperatorBase : public Operator<MemorySpaceType>
{
public:
  ThermalOperatorBase() = default;

  virtual ~ThermalOperatorBase() = default;

  // The function cannot be virtual and templated
  virtual void
  reinit(dealii::DoFHandler<dim> const &dof_handler,
         dealii::AffineConstraints<double> const &affine_constraints,
         dealii::QGaussLobatto<1> const &quad) = 0;

  virtual void
  reinit(dealii::DoFHandler<dim> const &dof_handler,
         dealii::AffineConstraints<double> const &affine_constraints,
         dealii::QGauss<1> const &quad) = 0;

  virtual void compute_inverse_mass_matrix(
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<double> const &affine_constraints) = 0;

  virtual std::shared_ptr<
      dealii::LA::distributed::Vector<double, MemorySpaceType>>
  get_inverse_mass_matrix() const = 0;

  virtual void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const = 0;

  virtual void evaluate_material_properties(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &state) = 0;

  void evaluate_material_properties(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> const
          &)
  {
    ASSERT(false, "Internal error");
  }
};
} // namespace adamantine
#endif
