/* Copyright (c) 2022 - 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MECHANICAL_PHYSICS_HH
#define MECHANICAL_PHYSICS_HH

#include <Geometry.hh>
#include <MechanicalOperator.hh>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/hp/fe_collection.h>

namespace adamantine
{
template <int dim, typename MemorySpaceType>
class MechanicalPhysics
{
public:
  /**
   * Constructor.
   */
  MechanicalPhysics(MPI_Comm const &communicator, unsigned int fe_degree,
                    Geometry<dim> &geometry,
                    MaterialProperty<dim, MemorySpaceType> &material_properties,
                    std::vector<double> initial_temperatures);

  /**
   * Setup the DoFHandler, the AffineConstraints, and the
   * MechanicalOperator.
   */
  void
  setup_dofs(std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces =
                 std::vector<std::shared_ptr<BodyForce<dim>>>());

  /**
   * Same as above when solving a thermo-mechanical problem.
   */
  void setup_dofs(
      dealii::DoFHandler<dim> const &thermal_dof_handler,
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &temperature,
      std::vector<bool> const &has_melted,
      std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces =
          std::vector<std::shared_ptr<BodyForce<dim>>>());

  /**
   * Solve the mechanical problem and return the solution.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solve();

  /**
   * Return the DoFHandler.
   */
  dealii::DoFHandler<dim> &get_dof_handler();

  /**
   * Return the AffineConstraints<double>.
   */
  dealii::AffineConstraints<double> &get_affine_constraints();

private:
  /**
   * Associated Geometry.
   */
  Geometry<dim> &_geometry;
  /**
   * Associated MaterialProperty.
   */
  MaterialProperty<dim, MemorySpaceType> &_material_properties;
  /**
   * Associated FECollection.
   */
  dealii::hp::FECollection<dim> _fe_collection;
  /**
   * Associated DoFHandler.
   */
  dealii::DoFHandler<dim> _dof_handler;
  /**
   * Associated AffineConstraints.
   */
  dealii::AffineConstraints<double> _affine_constraints;
  /**
   * Associated QCollection.
   */
  dealii::hp::QCollection<dim> _q_collection;
  /**
   * Pointer to the MechanicalOperator
   */
  std::unique_ptr<MechanicalOperator<dim, MemorySpaceType>>
      _mechanical_operator;
  /**
   * Whether to include a gravitional body force in the calculation.
   */
  bool _include_gravity;
};

template <int dim, typename MemorySpaceType>
inline dealii::DoFHandler<dim> &
MechanicalPhysics<dim, MemorySpaceType>::get_dof_handler()
{
  return _dof_handler;
}

template <int dim, typename MemorySpaceType>
inline dealii::AffineConstraints<double> &
MechanicalPhysics<dim, MemorySpaceType>::get_affine_constraints()
{
  return _affine_constraints;
}

} // namespace adamantine

#endif
