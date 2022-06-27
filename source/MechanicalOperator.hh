/* Copyright (c) 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MECHANICAL_OPERATOR_HH
#define MECHANICAL_OPERATOR_HH

#include <Operator.hh>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
template <int dim>
class MechanicalOperator : public Operator<dealii::MemorySpace::Host>
{
public:
  MechanicalOperator(MPI_Comm const &communicator,
                     boost::property_tree::ptree const &database);

  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints,
              dealii::hp::QCollection<dim> const &quad);

  dealii::types::global_dof_index m() const override;

  dealii::types::global_dof_index n() const override;

  void
  vmult(dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &src) const override;

  void Tvmult(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &src) const override;

  void vmult_add(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &src) const override;

  void Tvmult_add(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &src) const override;

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const &
  rhs() const;

  dealii::TrilinosWrappers::SparseMatrix const &system_matrix() const;

private:
  /**
   * Assemble the matrix and the right-hand-side for an elastostatic system.
   * @Note The 2D case does not represent any physical model but it is
   * convenient for testing.
   */
  void assemble_elastostatic_system();

  /**
   * MPI communicator.
   */
  MPI_Comm const &_communicator;
  /**
   * Output the latex formula of the bilinear form
   */
  bool _bilinear_form_output = true;
  /**
   *
   */
  boost::property_tree::ptree const &_database;
  /**
   * Non-owning pointer to the DoFHandler from MechanicalPhysics
   */
  dealii::DoFHandler<dim> const *_dof_handler = nullptr;
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
  dealii::TrilinosWrappers::SparseMatrix _system_matrix;
};

template <int dim>
inline dealii::types::global_dof_index MechanicalOperator<dim>::m() const
{
  return _system_matrix.m();
}

template <int dim>
inline dealii::types::global_dof_index MechanicalOperator<dim>::n() const
{
  return _system_matrix.n();
}

template <int dim>
inline dealii::LA::distributed::Vector<double,
                                       dealii::MemorySpace::Host> const &
MechanicalOperator<dim>::rhs() const
{
  return _system_rhs;
}

template <int dim>
inline dealii::TrilinosWrappers::SparseMatrix const &
MechanicalOperator<dim>::system_matrix() const
{
  return _system_matrix;
}
} // namespace adamantine
#endif
