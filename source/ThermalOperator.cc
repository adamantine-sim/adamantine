/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperator.hh>
#include <instantiation.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <algorithm>

namespace adamantine
{

template <int dim, int fe_degree, typename MemorySpaceType>
ThermalOperator<dim, fe_degree, MemorySpaceType>::ThermalOperator(
    MPI_Comm const &communicator,
    std::shared_ptr<MaterialProperty<dim>> material_properties,
    std::vector<std::shared_ptr<HeatSource<dim>>> heat_sources)
    : _communicator(communicator), _material_properties(material_properties),
      _heat_sources(heat_sources),
      _inverse_mass_matrix(
          new dealii::LA::distributed::Vector<double, MemorySpaceType>())
{
  _matrix_free_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, double>::AdditionalData::partition_color;
  _matrix_free_data.mapping_update_flags =
      dealii::update_values | dealii::update_gradients |
      dealii::update_JxW_values | dealii::update_quadrature_points;
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::hp::QCollection<1> const &q_collection)
{
  _matrix_free.reinit(dof_handler, affine_constraints, q_collection,
                      _matrix_free_data);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    compute_inverse_mass_matrix(
        dealii::DoFHandler<dim> const &dof_handler,
        dealii::AffineConstraints<double> const &affine_constraints,
        dealii::hp::FECollection<dim> const &fe_collection)
{
  // Compute the inverse of the mass matrix
  dealii::hp::QCollection<dim> mass_matrix_q_collection;
  mass_matrix_q_collection.push_back(dealii::QGaussLobatto<dim>(fe_degree + 1));
  mass_matrix_q_collection.push_back(dealii::QGaussLobatto<dim>(2));
  auto locally_owned_dofs = dof_handler.locally_owned_dofs();
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
  _inverse_mass_matrix->reinit(locally_owned_dofs, locally_relevant_dofs,
                               _communicator);
  dealii::hp::FEValues<dim> hp_fe_values(
      fe_collection, mass_matrix_q_collection,
      dealii::update_quadrature_points | dealii::update_values |
          dealii::update_JxW_values);
  unsigned int const dofs_per_cell = fe_collection.max_dofs_per_cell();
  unsigned int const n_q_points =
      mass_matrix_q_collection.max_n_quadrature_points();
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
  dealii::Vector<double> cell_mass(dofs_per_cell);
  for (auto cell :
       dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    if (cell->active_fe_index() == 0)
    {
      cell_mass = 0.;
      hp_fe_values.reinit(cell);
      dealii::FEValues<dim> const &fe_values =
          hp_fe_values.get_present_fe_values();
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {

            cell_mass[i] += fe_values.shape_value(j, q) *
                            fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
        }
      }
      cell->get_dof_indices(local_dof_indices);
      affine_constraints.distribute_local_to_global(
          cell_mass, local_dof_indices, *_inverse_mass_matrix);
    }
  }
  _inverse_mass_matrix->compress(dealii::VectorOperation::add);

  unsigned int const local_size = _inverse_mass_matrix->local_size();
  for (unsigned int k = 0; k < local_size; ++k)
  {
    if (_inverse_mass_matrix->local_element(k) > 1e-15)
      _inverse_mass_matrix->local_element(k) =
          1. / _inverse_mass_matrix->local_element(k);
    else
      _inverse_mass_matrix->local_element(k) = 0.;
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::clear()
{
  _matrix_free.clear();
  _inverse_mass_matrix->reinit(0);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::vmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::Tvmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::vmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  // Execute the matrix-free matrix-vector multiplication
  _matrix_free.cell_loop(&ThermalOperator::local_apply, this, dst, src);

  // Because cell_loop resolves the constraints, the constrained dofs are not
  // called they stay at zero. Thus, we need to force the value on the
  // constrained dofs by hand. The variable scaling is used so that we get the
  // right order of magnitude.
  // TODO: for now the value of scaling is set to 1
  double const scaling = 1.;
  std::vector<unsigned int> const &constrained_dofs =
      _matrix_free.get_constrained_dofs();
  for (auto &dof : constrained_dofs)
    dst.local_element(dof) += scaling * src.local_element(dof);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::Tvmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  // The system of equation is symmetric so we can use vmult_add
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::local_apply(
    dealii::MatrixFree<dim, double> const &data,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
    std::pair<unsigned int, unsigned int> const &cell_range) const
{
  // Get the subrange of cells associated with the fe index 0
  std::pair<unsigned int, unsigned int> cell_subrange =
      data.create_cell_subrange_hp_by_index(cell_range, 0);

  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(data);

  unsigned int const n_cells = _matrix_free.n_cell_batches();
  _material_properties->reinit_powder_ratio(n_cells, fe_eval.n_q_points);
  _material_properties->reinit_material_id(n_cells, fe_eval.n_q_points);

  dealii::Tensor<1, dim> unit_tensor;
  for (unsigned int i = 0; i < dim; ++i)
    unit_tensor[i] = 1.;

  std::array<dealii::VectorizedArray<double>,
             static_cast<unsigned int>(MaterialState::SIZE)>
      state_ratios = {dealii::make_vectorized_array(-1.0),
                      dealii::make_vectorized_array(-1.0),
                      dealii::make_vectorized_array(-1.0)};

  // Loop over the "cells". Note that we don't really work on a cell but on a
  // set of quadrature points.
  for (unsigned int cell = cell_subrange.first; cell < cell_subrange.second;
       ++cell)
  {
    // Reinit fe_eval on the current cell
    fe_eval.reinit(cell);
    // Store in a local vector the local values of src
    fe_eval.read_dof_values(src);
    // Evaluate only the function gradients on the reference cell
    // fe_eval.evaluate(false, true);
    fe_eval.evaluate(
        true,
        true); // Need the temperature to calculate the material properties

    // Apply the Jacobian of the transformation, multiply by the variable
    // coefficients and the quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {

      dealii::VectorizedArray<double> temperature = fe_eval.get_value(q);

      // Calculate the local material properties
      _material_properties->update_state_ratios(cell, q, temperature,
                                                state_ratios);

      dealii::VectorizedArray<double> inv_rho_cp =
          _material_properties->get_inv_rho_cp(cell, q, state_ratios,
                                               temperature);

      dealii::VectorizedArray<double> thermal_conductivity =
          _material_properties->get_thermal_conductivity(cell, q, state_ratios,
                                                         temperature);

      // Calculate the gradient contribution for the update

      std::cout << temperature[0] << " " << state_ratios[0][0] << " "
                << state_ratios[1][0] << " " << state_ratios[2][0] << " "
                << thermal_conductivity[0] << " " << inv_rho_cp[0] << std::endl;

      fe_eval.submit_gradient(
          -inv_rho_cp * thermal_conductivity * fe_eval.get_gradient(q), q);

      // Calculate the value contribution for the update
      dealii::Point<dim, dealii::VectorizedArray<double>> const &q_point =
          fe_eval.quadrature_point(q);

      dealii::VectorizedArray<double> quad_pt_source = 0.0;
      for (unsigned int i = 0;
           i < _matrix_free.n_active_entries_per_cell_batch(cell); ++i)
      {
        dealii::Point<dim> q_point_loc;
        for (unsigned int d = 0; d < dim; ++d)
          q_point_loc(d) = q_point(d)[i];

        for (auto &beam : _heat_sources)
          quad_pt_source[i] += beam->value(q_point_loc, _time, _current_height);
      }
      quad_pt_source *= inv_rho_cp;
      // std::cout << quad_pt_source << std::endl;

      fe_eval.submit_value(quad_pt_source, q);
    }

    // Sum over the quadrature points.
    fe_eval.integrate(true, true);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    evaluate_material_properties(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &temperature)
{
  // Update the state of the materials
  _material_properties->update(_matrix_free.get_dof_handler(), temperature);

  unsigned int const n_cells = _matrix_free.n_cell_batches();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      _matrix_free);
  _inv_rho_cp.reinit(n_cells, fe_eval.n_q_points);
  _thermal_conductivity.reinit(n_cells, fe_eval.n_q_points);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      for (unsigned int i = 0;
           i < _matrix_free.n_active_entries_per_cell_batch(cell); ++i)
      {
        typename dealii::DoFHandler<dim>::cell_iterator cell_it =
            _matrix_free.get_cell_iterator(cell, i);
        // Cast to Triangulation<dim>::cell_iterator to access the material_id
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell_it);

        _thermal_conductivity(cell, q)[i] = _material_properties->get(
            cell_tria, StateProperty::thermal_conductivity);

        _inv_rho_cp(cell, q)[i] =
            1. / (_material_properties->get(cell_tria, StateProperty::density) *
                  _material_properties->get(cell_tria,
                                            StateProperty::specific_heat));
        _cell_it_to_mf_cell_map[cell_it] = std::make_pair(cell, i);
      }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    extract_stateful_material_properties(
        dealii::LA::distributed::Vector<double, MemorySpaceType> &temperature)
{
  // Update the state of the materials (is this needed here?)
  _material_properties->update(_matrix_free.get_dof_handler(), temperature);

  unsigned int const n_cells = _matrix_free.n_cell_batches();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      _matrix_free);

  _material_properties->reinit_powder_ratio(n_cells, fe_eval.n_q_points);
  _material_properties->reinit_material_id(n_cells, fe_eval.n_q_points);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      for (unsigned int i = 0;
           i < _matrix_free.n_active_entries_per_cell_batch(cell); ++i)
      {
        typename dealii::DoFHandler<dim>::cell_iterator cell_it =
            _matrix_free.get_cell_iterator(cell, i);
        // Cast to Triangulation<dim>::cell_iterator to access the material_id
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell_it);

        _material_properties->set_powder_ratio(
            cell, q, i,
            _material_properties->get_state_ratio(cell_tria,
                                                  MaterialState::powder));
        _material_properties->set_material_id(
            cell, q, i, _material_properties->get_material_id(cell_tria));
      }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree,
                     MemorySpaceType>::sync_stateful_material_properties()
{
  // TODO
}

} // namespace adamantine

INSTANTIATE_DIM_FEDEGREE_HOST(TUPLE(ThermalOperator))
