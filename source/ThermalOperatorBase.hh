/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
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
      dealii::AffineConstraints<double> const &affine_constraints,
      dealii::hp::FECollection<dim> const &fe_collection) = 0;

  virtual std::shared_ptr<
      dealii::LA::distributed::Vector<double, MemorySpaceType>>
  get_inverse_mass_matrix() const = 0;

  virtual void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const = 0;

  virtual void evaluate_material_properties(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &state) = 0;

  virtual void sync_stateful_material_properties() = 0;

  virtual void extract_stateful_material_properties(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector) = 0;

  void evaluate_material_properties(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> const
          &)
  {
    ASSERT(false, "Internal error");
  }

  virtual double get_inv_rho_cp(
      typename dealii::DoFHandler<dim>::cell_iterator const &) const = 0;

  virtual void update_time_and_height(double time, double height) = 0;
};
} // namespace adamantine
#endif
