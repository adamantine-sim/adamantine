/* SPDX-FileCopyrightText: Copyright (c) 2022 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <MechanicalOperator.hh>
#include <instantiation.hh>
#include <utils.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/differentiation/ad.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/affine_constraints.h>
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
#include <deal.II/lac/affine_constraints.templates.h>
#endif
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali.h>
#endif

namespace adamantine
{
template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
MechanicalOperator<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    MechanicalOperator(
        MPI_Comm const &communicator,
        MaterialProperty<dim, n_materials, p_order, MaterialStates,
                         MemorySpaceType> &material_properties,
        std::vector<double> const &reference_temperatures)
    : _communicator(communicator),
      _reference_temperatures(reference_temperatures),
      _material_properties(material_properties)
{
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalOperator<dim, n_materials, p_order, MaterialStates,
                        MemorySpaceType>::
    reinit(dealii::DoFHandler<dim> const &dof_handler,
           dealii::AffineConstraints<double> const &affine_constraints,
           dealii::hp::QCollection<dim> const &q_collection,
           std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _dof_handler = &dof_handler;
  _affine_constraints = &affine_constraints;
  _q_collection = &q_collection;
  assemble_system(body_forces);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalOperator<dim, n_materials, p_order, MaterialStates,
                        MemorySpaceType>::
    update_temperature(
        dealii::DoFHandler<dim> const &thermal_dof_handler,
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &temperature,
        std::vector<bool> const &has_melted)
{
  _thermal_dof_handler = &thermal_dof_handler;
  _temperature = temperature;
  _has_melted = has_melted;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalOperator<dim, n_materials, p_order, MaterialStates,
                        MemorySpaceType>::
    assemble_system(
        std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("assemble mechanical system");
#endif

  // Create the sparsity pattern. Since we use a Trilinos matrix we don't need
  // the sparsity pattern to outlive the sparse matrix.
  auto locally_owned_dofs = _dof_handler->locally_owned_dofs();
  auto locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(*_dof_handler);
  dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
  dealii::DoFTools::make_sparsity_pattern(*_dof_handler, dsp,
                                          *_affine_constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned_dofs, _communicator, locally_relevant_dofs);

  _system_matrix.reinit(locally_owned_dofs, dsp, _communicator);

  dealii::hp::FEValues<dim> displacement_hp_fe_values(
      _dof_handler->get_fe_collection(), *_q_collection,
      dealii::update_values | dealii::update_gradients |
          dealii::update_JxW_values);

  unsigned int const dofs_per_cell =
      _dof_handler->get_fe_collection().max_dofs_per_cell();
  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over the locally owned cells that are not FE_Nothing and assemble the
  // sparse matrix and the right-hand-side
  for (auto const &cell : _dof_handler->active_cell_iterators() |
                              dealii::IteratorFilters::ActiveFEIndexEqualTo(
                                  0, /* locally owned */ true))
  {
    displacement_hp_fe_values.reinit(cell);
    auto const &fe_values = displacement_hp_fe_values.get_present_fe_values();
    auto const &fe = fe_values.get_fe();

    // Assemble the local martrix
    cell_matrix = 0;
    double const lambda = this->_material_properties.get_mechanical_property(
        cell, StateProperty::lame_first_parameter);
    double const mu = this->_material_properties.get_mechanical_property(
        cell, StateProperty::lame_second_parameter);
    for (auto const i : fe_values.dof_indices())
    {
      auto const component_i = fe.system_to_component_index(i).first;
      for (auto const j : fe_values.dof_indices())
      {
        auto const component_j = fe.system_to_component_index(j).first;
        for (auto const q_point : fe_values.quadrature_point_indices())
        {
          cell_matrix(i, j) +=
              // FIXME We should be able to use the following formulation but
              // the result is different. We need to understand why.
              // ((lambda + mu) * fe_values.shape_grad(i, q_point)[component_i]
              // * fe_values.shape_grad(j, q_point)[component_j] +
              ((fe_values.shape_grad(i, q_point)[component_i] *
                fe_values.shape_grad(j, q_point)[component_j] * lambda) +
               (fe_values.shape_grad(i, q_point)[component_j] *
                fe_values.shape_grad(j, q_point)[component_i] * mu) +
               ((component_i == component_j)
                    ? mu * fe_values.shape_grad(i, q_point) *
                          fe_values.shape_grad(j, q_point)
                    : 0.)) *
              fe_values.JxW(q_point);
        }
      }
    }
    cell->get_dof_indices(local_dof_indices);
    _affine_constraints->distribute_local_to_global(
        cell_matrix, local_dof_indices, _system_matrix);
  }

  // Assemble the rhs
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      assembled_rhs(locally_owned_dofs, locally_relevant_dofs, _communicator);
  dealii::Vector<double> cell_rhs(dofs_per_cell);
  // If the list of reference temperatures is non-empty, we solve the
  // thermo-elastic problem.
  if (_reference_temperatures.size() > 0)
  {
    // Create temperature hp::FEValues using the same finite elements as the
    // thermal simulation but evaluated at the quadrature points of the
    // mechanical simulations.
    dealii::hp::FEValues<dim> temperature_hp_fe_values(
        _thermal_dof_handler->get_fe_collection(), *_q_collection,
        dealii::update_values);

    // _has_melted is using its own indices. The indices are computed using the
    // locally owned cells of the thermal DoFHandler that have an FE_Index
    // equal to zero. We need to translate these indices to be used with the
    // mechanical DoFHandler. This is simplified by the fact that both the
    // thermal and the mechanical simulation use the same Triangulation and have
    // the same cells locally owned.
    auto &triangulation = _dof_handler->get_triangulation();
    unsigned int const n_local_active_cells = triangulation.n_active_cells();
    std::vector<unsigned int> cell_indices(n_local_active_cells);
    unsigned int cell_index = 0;
    for (auto const &tria_cell :
         triangulation.active_cell_iterators() |
             dealii::IteratorFilters::LocallyOwnedCell())
    {
      dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
          temperature_cell(&triangulation, tria_cell->level(),
                           tria_cell->index(), _thermal_dof_handler);
      if (temperature_cell->active_fe_index() == 0)
      {
        dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
            displacement_cell(&triangulation, tria_cell->level(),
                              tria_cell->index(), _dof_handler);
        if (displacement_cell->active_fe_index() == 0)
        {
          cell_indices[displacement_cell->active_cell_index()] = cell_index;
        }
        ++cell_index;
      }
    }

    _temperature.update_ghost_values();

    std::vector<dealii::types::global_dof_index> temperature_local_dof_indices(
        _thermal_dof_handler->get_fe_collection().max_dofs_per_cell());
    double const initial_temperature = _reference_temperatures.back();
    for (auto const &cell : _dof_handler->active_cell_iterators() |
                                dealii::IteratorFilters::ActiveFEIndexEqualTo(
                                    0, /* locally owned */ true))
    {
      cell_rhs = 0.;

      // Get the temperature cell associated to the mechanical cell
      dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
          temperature_cell(&triangulation, cell->level(), cell->index(),
                           _thermal_dof_handler);

      // Get the appropriate reference temperature for the cell. If the cell
      // is not in the unmelted substrate, the reference temperature depends
      // on the material.
      double reference_temperature =
          _has_melted[cell_indices[cell->active_cell_index()]]
              ? _reference_temperatures[temperature_cell->material_id()]
              : initial_temperature;

      displacement_hp_fe_values.reinit(cell);
      auto const &fe_values = displacement_hp_fe_values.get_present_fe_values();

      double const alpha = this->_material_properties.get_mechanical_property(
          cell, StateProperty::thermal_expansion_coef);
      double const lambda = this->_material_properties.get_mechanical_property(
          cell, StateProperty::lame_first_parameter);
      double const mu = this->_material_properties.get_mechanical_property(
          cell, StateProperty::lame_second_parameter);

      temperature_hp_fe_values.reinit(temperature_cell);
      auto &temperature_fe_values =
          temperature_hp_fe_values.get_present_fe_values();
      temperature_cell->get_dof_indices(temperature_local_dof_indices);

      dealii::FEValuesExtractors::Vector const displacements(0);

      for (auto const q_point : fe_values.quadrature_point_indices())
      {
        double delta_T = -reference_temperature;
        for (unsigned int j = 0; j < temperature_fe_values.dofs_per_cell; ++j)
        {
          delta_T += temperature_fe_values.shape_value(j, q_point) *
                     _temperature(temperature_local_dof_indices[j]);
        }
        auto B = dealii::Physics::Elasticity::StandardTensors<dim>::I;
        B *= (3. * lambda + 2 * mu) * alpha * delta_T;

        for (auto const i : fe_values.dof_indices())
        {
          cell_rhs(i) += dealii::scalar_product(
                             B, fe_values[displacements].gradient(i, q_point)) *
                         fe_values.JxW(q_point);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      _affine_constraints->distribute_local_to_global(
          cell_rhs, local_dof_indices, assembled_rhs);
    }
  }

  // Add gravitational body force
  if (body_forces.size())
  {
    for (auto const &cell : _dof_handler->active_cell_iterators() |
                                dealii::IteratorFilters::ActiveFEIndexEqualTo(
                                    0, /* locally owned */ true))
    {
      cell_rhs = 0.;

      displacement_hp_fe_values.reinit(cell);
      auto const &fe_values = displacement_hp_fe_values.get_present_fe_values();

      dealii::Tensor<1, dim, double> body_force;
      for (auto &force : body_forces)
      {
        body_force += force->eval(cell);
      }

      dealii::FEValuesExtractors::Vector const displacements(0);

      for (auto const i : fe_values.dof_indices())
      {
        for (auto const q_point : fe_values.quadrature_point_indices())
          cell_rhs(i) += body_force *
                         fe_values[displacements].value(i, q_point) *
                         fe_values.JxW(q_point);
      }

      cell->get_dof_indices(local_dof_indices);
      _affine_constraints->distribute_local_to_global(
          cell_rhs, local_dof_indices, assembled_rhs);
    }
  }

  _system_matrix.compress(dealii::VectorOperation::add);
  assembled_rhs.compress(dealii::VectorOperation::add);

  // When solving the system, we don't want ghost entries
  _system_rhs.reinit(_dof_handler->locally_owned_dofs(), _communicator);
  _system_rhs = assembled_rhs;

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_END("assemble mechanical system");
#endif
}
} // namespace adamantine

INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_HOST(MechanicalOperator)
INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_DEVICE(MechanicalOperator)
