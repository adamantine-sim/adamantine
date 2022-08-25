/* Copyright (c) 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <MechanicalPhysics.hh>
#include <instantiation.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/vector_tools.h>

namespace adamantine
{
template <int dim, typename MemorySpaceType>
MechanicalPhysics<dim, MemorySpaceType>::MechanicalPhysics(
    MPI_Comm const &communicator, unsigned int fe_degree,
    Geometry<dim> &geometry,
    MaterialProperty<dim, MemorySpaceType> &material_properties,
    double initial_temperature)
    : _geometry(geometry), _dof_handler(_geometry.get_triangulation())
{
  // Create the FECollection
  _fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Q<dim>(fe_degree) ^ dim));
  _fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Nothing<dim>() ^ dim));

  // Create the QCollection
  _q_collection.push_back(dealii::QGauss<dim>(fe_degree + 1));
  _q_collection.push_back(dealii::QGauss<dim>(1));

  // Create the mechanical operator
  _mechanical_operator =
      std::make_unique<MechanicalOperator<dim, MemorySpaceType>>(
          communicator, material_properties, initial_temperature);
}

template <int dim, typename MemorySpaceType>
void MechanicalPhysics<dim, MemorySpaceType>::setup_dofs()
{
  _dof_handler.distribute_dofs(_fe_collection);
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler,
                                                  locally_relevant_dofs);
  _affine_constraints.clear();
  _affine_constraints.reinit(locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler,
                                                  _affine_constraints);
  // TODO For now only Dirichlet boundary condition
  dealii::VectorTools::interpolate_boundary_values(
      _dof_handler, 0, dealii::Functions::ZeroFunction<dim>(dim),
      _affine_constraints);
  _affine_constraints.close();

  _mechanical_operator->reinit(_dof_handler, _affine_constraints,
                               _q_collection);
}

template <int dim, typename MemorySpaceType>
void MechanicalPhysics<dim, MemorySpaceType>::setup_dofs(
    dealii::DoFHandler<dim> const &thermal_dof_handler,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &temperature)
{
  setup_dofs();
  _mechanical_operator->update_temperature(thermal_dof_handler, temperature);
}

template <int dim, typename MemorySpaceType>
dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
MechanicalPhysics<dim, MemorySpaceType>::solve()
{
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solution(
      _mechanical_operator->rhs().get_partitioner());

  unsigned int const max_iter = _dof_handler.n_dofs() / 10;
  double const tol = 1e-12 * _mechanical_operator->rhs().l2_norm();
  dealii::SolverControl solver_control(max_iter, tol);
  dealii::SolverCG<
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>>
      cg(solver_control);
  // TODO Use better preconditioner
  dealii::TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(_mechanical_operator->system_matrix());
  cg.solve(_mechanical_operator->system_matrix(), solution,
           _mechanical_operator->rhs(), preconditioner);
  _affine_constraints.distribute(solution);

  return solution;
}

} // namespace adamantine

INSTANTIATE_DIM_HOST(MechanicalPhysics)
