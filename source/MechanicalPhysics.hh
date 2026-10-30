/* SPDX-FileCopyrightText: Copyright (c) 2022 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MECHANICAL_PHYSICS_HH
#define MECHANICAL_PHYSICS_HH

#include <Boundary.hh>
#include <ClosestSourcePointAdaptation.hh>
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
template <int dim>
struct MechanicalConstitutiveState
{
  dealii::SymmetricTensor<2, dim> stress;
  dealii::SymmetricTensor<2, dim> back_stress;
  double plastic_internal_variable = 0.;
  double thermal_stress = 0.;
};

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
class MechanicalPhysics
{
public:
  /**
   * Constructor.
   */
  MechanicalPhysics(MPI_Comm const &communicator, unsigned int const fe_degree,
                    Geometry<dim> &geometry, Boundary const &boundary,
                    MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                     MemorySpaceType> &material_properties,
                    std::vector<double> const &initial_temperatures,
                    bool verbose_output = false);

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
      std::vector<bool> const &has_melted, bool rebuild_matrix,
      std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces =
          std::vector<std::shared_ptr<BodyForce<dim>>>());

  /**
   * Recompute the right-hand-side.
   */
  void
  update_rhs(std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces =
                 std::vector<std::shared_ptr<BodyForce<dim>>>());

  /**
   * Same as above when solving a thermo-mechanical problem
   */
  void update_rhs(
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
   *
   * The default solver selection uses the elastic limit from the material
   * properties. An infinite elastic limit selects the linear solver; any
   * finite elastic limit selects the nonlinear solver. The nonlinear solver
   * uses the residual \f$ R(u) = f_{internal}(u) - f_{external} \f$ with the
   * elastic system matrix as a fixed approximate Jacobian. If verbose output
   * was enabled in the constructor, the nonlinear iteration count is printed
   * on rank zero.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solve();

  /**
   * Evaluate the mechanical residual for a displacement. The existing
   * constitutive update is used to associate the current quadrature-point
   * stress with the displacement before assembling
   * \f[
   *   R(u) = f_internal(u) - f_external.
   * \f]
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
  evaluate_residual(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &displacement);

  /**
   * Return the DoFHandler.
   */
  dealii::DoFHandler<dim> &get_dof_handler();

  /**
   * Return the AffineConstraints<double>.
   */
  dealii::AffineConstraints<double> &get_affine_constraints();

  /**
   * Return the constitutive state associated to each quadrature point.
   */
  std::vector<std::vector<MechanicalConstitutiveState<dim>>> const &
  get_constitutive_state() const;

  /**
   * Replace the constitutive state and discard any trial state.
   *
   * This is used to initialize or restore material history. Normal material
   * evolution must proceed through the trial-state and update interfaces.
   */
  void set_constitutive_state(
      std::vector<std::vector<MechanicalConstitutiveState<dim>>> const &state);

  /**
   * Return the stress tensor associated to each quadrature point.
   */
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>>
  get_stress_tensor() const;

private:
  /**
   * Solve a linear system with the elastic matrix and return a constrained
   * displacement vector.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
  solve_linear_system(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &rhs,
      double const tolerance) const;

  /**
   * Solve the nonlinear mechanical residual with KINSOL using the elastic
   * matrix as a fixed approximate Jacobian.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
  solve_nonlinear_system();

  /**
   * Evaluate the local constitutive response without modifying the state.
   */
  MechanicalConstitutiveState<dim> evaluate_constitutive_response(
      dealii::SymmetricTensor<2, dim> const &strain_increment,
      double const temperature, double const reference_temperature,
      double const lambda, double const mu, double const plastic_modulus,
      double const iso_hardening_coef, double const alpha,
      bool const include_thermal_loading,
      MechanicalConstitutiveState<dim> const &state) const;

  /**
   * Compute the displacement increment relative to the previous displacement.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
  compute_incremental_displacement(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &displacement) const;

  /**
   * Evaluate the constitutive response at all quadrature points for the
   * displacement increment and store the result in the trial state.
   */
  void evaluate_trial_state(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &incremental_displacement);

  /**
   * Associated Geometry.
   */
  Geometry<dim> &_geometry;
  /**
   * Associated Boundary.
   */
  Boundary _boundary;
  /**
   * Associated MaterialProperty.
   */
  MaterialProperty<dim, n_materials, p_order, MaterialStates, MemorySpaceType>
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
  std::unique_ptr<MechanicalOperator<dim, n_materials, p_order, MaterialStates,
                                     MemorySpaceType>>
      _mechanical_operator;
  /**
   * Reference temperature for each material followed by the initial
   * temperature of the substrate. An empty vector denotes a mechanical-only
   * simulation.
   */
  std::vector<double> _reference_temperatures;
  /**
   * Non-owning pointer to the thermal DoFHandler.
   */
  dealii::DoFHandler<dim> const *_thermal_dof_handler = nullptr;
  /**
   * Current temperature field.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      _temperature;
  /**
   * Indicator for cells that have melted at least once.
   */
  std::vector<bool> _has_melted;
  /**
   * Whether to include a gravitional body force in the calculation.
   */
  bool _include_gravity;
  /**
   * Whether all mechanical material has an infinite elastic limit.
   */
  bool _is_purely_elastic;
  /**
   * Whether to print nonlinear solver information on rank zero.
   */
  bool _verbose_output;
  /**
   * Save displacement from the previous time step.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      _old_displacement;

  /**
   * Constitutive state at each (cell, quadrature point).
   */
  std::vector<std::vector<MechanicalConstitutiveState<dim>>> _state;

  /**
   * Trial constitutive state at each (cell, quadrature point).
   */
  std::vector<std::vector<MechanicalConstitutiveState<dim>>> _trial_state;

  /**
   * Solution transfer object used for updating _old_displacement when the
   * triangulation is updated when adding material
   */
#if DEAL_II_VERSION_GTE(9, 7, 0)
  dealii::SolutionTransfer<
#else
  dealii::parallel::distributed::SolutionTransfer<
#endif
      dim, dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>>
      _solution_transfer;

  /**
   * Object used for interpolating to and from the closest quadrature point in
   * the cell data transfer object.
   */
  ClosestQuadPointAdaptation<dim, dim, std::vector<double>>
      _closest_quad_point_adaptation;

  /**
   * Cell data transfer object used for updating the constitutive state when
   * the triangulation is updated when adding material.
   */
  dealii::parallel::distributed::CellDataTransfer<
      dim, dim, std::vector<std::vector<std::vector<double>>>>
      _cell_data_transfer;

  /**
   * Temporary storaged used by _cell_data_transfer
   */
  std::vector<std::vector<std::vector<double>>> _data_to_transfer;
};

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline dealii::DoFHandler<dim> &
MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                  MemorySpaceType>::get_dof_handler()
{
  return _dof_handler;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline dealii::AffineConstraints<double> &
MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                  MemorySpaceType>::get_affine_constraints()
{
  return _affine_constraints;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline std::vector<std::vector<MechanicalConstitutiveState<dim>>> const &
MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                  MemorySpaceType>::get_constitutive_state() const
{
  return _state;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline void
MechanicalPhysics<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    set_constitutive_state(
        std::vector<std::vector<MechanicalConstitutiveState<dim>>> const &state)
{
  _state = state;
  _trial_state = _state;
}

} // namespace adamantine

#endif
