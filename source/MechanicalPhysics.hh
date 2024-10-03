/* Copyright (c) 2022 - 2024, the adamantine authors.
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
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/hp/fe_collection.h>

namespace adamantine
{
template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
class MechanicalPhysics
{
public:
  /**
   * Constructor.
   */
  MechanicalPhysics(MPI_Comm const &communicator, unsigned int const fe_degree,
                    Geometry<dim> &geometry,
                    MaterialProperty<dim, p_order, MaterialStates,
                                     MemorySpaceType> &material_properties,
                    std::vector<double> const &initial_temperatures);

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
   * Prepare displacement and stresses to be communicated when activating cells
   * or refining the mesh.
   */
  void prepare_transfer_mpi();

  /**
   * Complete transfer of displacment and stress data after activating cells or
   * refining the mesh.
   */
  void complete_transfer_mpi();

  /**
   * Solve the mechanical problem and return the displacement.
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

  /**
   * Return the stress tensor associated to each quadrature point.
   */
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> &
  get_stress_tensor();

private:
  // Compute the stress using linear combination of isotropic and kinematic
  // hardening in the book Plasticity Modeling & Computation from Ronaldo I.
  // Borja.
  void compute_stress(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &displacement);

  /**
   * Associated Geometry.
   */
  Geometry<dim> &_geometry;
  /**
   * Associated MaterialProperty.
   */
  MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>
      &_material_properties;
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
  std::unique_ptr<
      MechanicalOperator<dim, p_order, MaterialStates, MemorySpaceType>>
      _mechanical_operator;
  /**
   * Whether to include a gravitional body force in the calculation.
   */
  bool _include_gravity;

  /**
   * Save displacement from the previous time step.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      _old_displacement;

  std::vector<std::vector<double>> _saved_old_displacement;

  /**
   * Plastic internal variable related to the strain
   */
  std::vector<std::vector<double>> _plastic_internal_variable;

  /**
   * Stress tensor at each (cell, quadrature point).
   */
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> _stress;

  /**
   * Back stress tensor at each (cell, quadrature point).
   */
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> _back_stress;

  dealii::parallel::distributed::SolutionTransfer<
      dim, dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>>
      _solution_transfer;

  dealii::parallel::distributed::CellDataTransfer<
      dim, dim, std::vector<std::vector<double>>>
      _cell_data_transfer;

  std::vector<std::vector<double>> _data_to_transfer;

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      _relevant_displacement;
};

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline dealii::DoFHandler<dim> &
MechanicalPhysics<dim, p_order, MaterialStates,
                  MemorySpaceType>::get_dof_handler()
{
  return _dof_handler;
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline dealii::AffineConstraints<double> &
MechanicalPhysics<dim, p_order, MaterialStates,
                  MemorySpaceType>::get_affine_constraints()
{
  return _affine_constraints;
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> &
MechanicalPhysics<dim, p_order, MaterialStates,
                  MemorySpaceType>::get_stress_tensor()
{
  return _stress;
}

} // namespace adamantine

#endif
