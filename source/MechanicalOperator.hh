/* SPDX-FileCopyrightText: Copyright (c) 2022 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MECHANICAL_OPERATOR_HH
#define MECHANICAL_OPERATOR_HH

#include <BodyForce.hh>
#include <MaterialProperty.hh>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#else
#include <deal.II/lac/trilinos_sparse_matrix.h>
#endif

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * This class is the operator associated with the solid mechanics equations.
 * The class is templated on the MemorySpace because it use MaterialProperty
 * which itself is templated on the MemorySpace but the operator is CPU only.
 */
template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
class MechanicalOperator
{
public:
  /**
   * Constructor. If the initial temperature is negative, the simulation is
   * mechanical only. Otherwise, we solve a thermo-mechanical problem.
   */
  MechanicalOperator(MPI_Comm const &communicator,
                     MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                      MemorySpaceType> &material_properties,
                     std::vector<double> const &reference_temperatures);

  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints,
              dealii::hp::QCollection<dim> const &quad,
              std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces =
                  std::vector<std::shared_ptr<BodyForce<dim>>>(),
              bool const discretization_has_changed = true);

  /**
   * Update the DoFHandler used by ThermalPhysics and update the temperature.
   */
  void update_temperature(
      dealii::DoFHandler<dim> const &thermal_dof_handler,
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &temperature,
      std::vector<bool> const &has_melted);

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const &
  rhs() const;

#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
  using TrilinosMatrixType =
      dealii::LinearAlgebra::TpetraWrappers::SparseMatrix<
          double, dealii::MemorySpace::Default>;
#else
  using TrilinosMatrixType = dealii::TrilinosWrappers::SparseMatrix;
#endif

  TrilinosMatrixType const &system_matrix() const;

private:
  /**
   * Assemble the matrix and the right-hand-side.
   * @note The 2D case does not represent any physical model but it is
   * convenient for testing.
   */
  void assemble_system(
      std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces);

  /**
   * MPI communicator.
   */
  MPI_Comm const &_communicator;
  /**
   * List of initial temperatures of the material. If the length of the vector
   * is nonzero, we solve a thermo-mechanical problem.
   */
  std::vector<double> _reference_temperatures;
  /**
   * Reference to the MaterialProperty from MechanicalPhysics.
   */
  MaterialProperty<dim, n_materials, p_order, MaterialStates, MemorySpaceType>
      &_material_properties;
  /**
   * Non-owning pointer to the DoFHandler from MechanicalPhysics
   */
  dealii::DoFHandler<dim> const *_dof_handler = nullptr;
  /**
   * Non-owning pointer to the DoFHandler from ThermalPhysics
   */
  dealii::DoFHandler<dim> const *_thermal_dof_handler = nullptr;
  /**
   * Non-owning pointer to the AffineConstraints from MechanicalPhysics
   */
  dealii::AffineConstraints<double> const *_affine_constraints = nullptr;
  /**
   * Non-owning pointer to the QCollection from MechanicalPhysics
   */
  dealii::hp::QCollection<dim> const *_q_collection = nullptr;
  /**
   * Right-hand-side of the mechanical problem.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      _system_rhs;
  /**
   * Matrix of the mechanical problem.
   */
  TrilinosMatrixType _system_matrix;
  /**
   * Temperature of the material.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      _temperature;
  /**
   * Indicator variable for whether a point has ever been above the solidus. The
   * value is false for material that has not yet melted and true for material
   * that has melted.
   */
  std::vector<bool> _has_melted;

  /**
   * Indicates whether the discretization has changed since the last time the
   * matrix was assembled.
   */
  bool _discretization_has_changed = true;
};

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline dealii::LA::distributed::Vector<double,
                                       dealii::MemorySpace::Host> const &
MechanicalOperator<dim, n_materials, p_order, MaterialStates,
                   MemorySpaceType>::rhs() const
{
  return _system_rhs;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline typename MechanicalOperator<dim, n_materials, p_order, MaterialStates,
                                   MemorySpaceType>::TrilinosMatrixType const &
MechanicalOperator<dim, n_materials, p_order, MaterialStates,
                   MemorySpaceType>::system_matrix() const
{
  return _system_matrix;
}
} // namespace adamantine
#endif
