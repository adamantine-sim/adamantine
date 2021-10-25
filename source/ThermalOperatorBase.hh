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
      dealii::LA::distributed::Vector<double, MemorySpaceType> const
          &state) = 0;

  virtual void clear() = 0;

  virtual void get_state_from_material_properties() = 0;

  virtual void set_state_to_material_properties() = 0;

  virtual void set_time_and_source_height(double, double) = 0;

  virtual double
  get_inv_rho_cp(typename dealii::DoFHandler<dim>::cell_iterator const &,
                 unsigned int) const = 0;
};
} // namespace adamantine
#endif
