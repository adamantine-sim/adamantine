/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperator.hh>
#include <instantiation.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace adamantine
{

template <int dim, int fe_degree, typename MemorySpaceType>
ThermalOperator<dim, fe_degree, MemorySpaceType>::ThermalOperator(
    MPI_Comm const &communicator,
    std::shared_ptr<MaterialProperty<dim>> material_properties,
    BoundaryType boundary_type)
    : _communicator(communicator), _material_properties(material_properties),
      _boundary_type(boundary_type),
      _inverse_mass_matrix(
          new dealii::LA::distributed::Vector<double, MemorySpaceType>())
{
  _matrix_free_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, double>::AdditionalData::partition_color;
  _matrix_free_data.mapping_update_flags = dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::hp::QCollection<1> const &q_collection)
{
  _matrix_free.reinit(dealii::StaticMappingQ1<dim>::mapping, dof_handler,
                      affine_constraints, q_collection, _matrix_free_data);
  _affine_constraints = &affine_constraints;
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
  for (auto const &cell : dealii::filter_iterators(
           dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
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
    affine_constraints.distribute_local_to_global(cell_mass, local_dof_indices,
                                                  *_inverse_mass_matrix);
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
  _matrix_free.cell_loop(&ThermalOperator::cell_local_apply, this, dst, src);
  // We compute the boundary conditions by hand. We would like to use
  // dealii::MatrixFree::loop() but there are two problems: 1. the function does
  // not work with FE_Nothing 2. even if it did we could only use it for the
  // domain boundary, we would still need to deal with the interface with
  // FE_Nothing ourselves.
  if (!(_boundary_type & BoundaryType::adiabatic))
  {
    unsigned int const dofs_per_cell = _matrix_free.get_dofs_per_cell();
    dealii::Vector<double> cell_src(dofs_per_cell);
    dealii::Vector<double> cell_dst(dofs_per_cell);
    auto &dof_handler = _matrix_free.get_dof_handler();
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);
    dealii::QGauss<dim - 1> face_quadrature(fe_degree + 1);
    dealii::FEFaceValues<dim> fe_face_values(
        dof_handler.get_fe(), face_quadrature,
        dealii::update_values | dealii::update_quadrature_points |
            dealii::update_JxW_values);
    unsigned int const n_face_q_points = face_quadrature.size();
    // Loop over the locally owned cells with an active FE index of zero
    for (auto const &cell : dealii::filter_iterators(
             dof_handler.active_cell_iterators(),
             dealii::IteratorFilters::LocallyOwnedCell(),
             dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
    {
      cell_dst = 0.;
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_src[i] = src[local_dof_indices[i]];
      double const inv_rho_cp = get_inv_rho_cp(cell);

      bool is_on_domain_boundary = false;
      for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell;
           ++f)
      {
        // We need to add the boundary conditions on the faces on the boundary
        // but also on the faces at the interface with FE_Nothing
        auto const &face = cell->face(f);
        if ((face->at_boundary()) ||
            ((!face->at_boundary()) &&
             (cell->neighbor(f)->active_fe_index() != 0)))
        {
          double conv_temperature_infty = 0.;
          double conv_heat_transfer_coef = 0.;
          double rad_temperature_infty = 0.;
          double rad_heat_transfer_coef = 0.;
          if (_boundary_type & BoundaryType::convective)
          {
            conv_temperature_infty = _material_properties->get(
                cell, Property::convection_temperature_infty);
            conv_heat_transfer_coef = _material_properties->get(
                cell, StateProperty::convection_heat_transfer_coef);
          }
          if (_boundary_type & BoundaryType::radiative)
          {
            rad_temperature_infty = _material_properties->get(
                cell, Property::radiation_temperature_infty);
            rad_heat_transfer_coef = _material_properties->get(
                cell, StateProperty::radiation_heat_transfer_coef);
          }

          fe_face_values.reinit(cell, face);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                cell_dst[i] -=
                    inv_rho_cp *
                    (conv_heat_transfer_coef + rad_heat_transfer_coef) *
                    fe_face_values.shape_value(i, q) *
                    fe_face_values.shape_value(j, q) * cell_src[j] *
                    fe_face_values.JxW(q);
              }
            }
          }
          is_on_domain_boundary = true;
        }
      }

      if (is_on_domain_boundary)
      {
        _affine_constraints->distribute_local_to_global(cell_dst,
                                                        local_dof_indices, dst);
      }
    }
  }

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
void ThermalOperator<dim, fe_degree, MemorySpaceType>::cell_local_apply(
    dealii::MatrixFree<dim, double> const &data,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
    std::pair<unsigned int, unsigned int> const &cell_range) const
{
  // Get the subrange of cells associated with the fe index 0
  std::pair<unsigned int, unsigned int> cell_subrange =
      data.create_cell_subrange_hp_by_index(cell_range, 0);

  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(data);
  dealii::Tensor<1, dim> unit_tensor;
  for (unsigned int i = 0; i < dim; ++i)
    unit_tensor[i] = 1.;

  // Loop over the "cells". Note that we don't really work on a cell but on a
  // set of quadrature point.
  for (unsigned int cell = cell_subrange.first; cell < cell_subrange.second;
       ++cell)
  {
    // Reinit fe_eval on the current cell
    fe_eval.reinit(cell);
    // Store in a local vector the local values of src
    fe_eval.read_dof_values(src);
    // Evaluate only the function gradients on the reference cell
    fe_eval.evaluate(false, true);
    // Apply the Jacobian of the transformation, multiply by the variable
    // coefficients and the quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      fe_eval.submit_gradient(-_inv_rho_cp(cell, q) *
                                  _thermal_conductivity(cell, q) *
                                  fe_eval.get_gradient(q),
                              q);
    // Sum over the quadrature points.
    fe_eval.integrate(false, true);
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

  // Store the volumetric material properties
  unsigned int const n_cells = _matrix_free.n_cell_batches();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      _matrix_free);
  unsigned int const fe_eval_n_q_points = fe_eval.n_q_points;
  _inv_rho_cp.reinit(n_cells, fe_eval_n_q_points);
  _thermal_conductivity.reinit(n_cells, fe_eval_n_q_points);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < fe_eval_n_q_points; ++q)
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
} // namespace adamantine

INSTANTIATE_DIM_FEDEGREE_HOST(TUPLE(ThermalOperator))
