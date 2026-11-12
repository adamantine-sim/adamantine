/* SPDX-FileCopyrightText: Copyright (c) 2022 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <MechanicalPhysics.hh>
#include <connected_components.hh>
#include <instantiation.hh>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/sundials/kinsol.h>

#include <algorithm>
#include <iostream>
#include <limits>

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali.h>
#endif

namespace adamantine
{
template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    MechanicalPhysics(
        MPI_Comm const &communicator, unsigned int const fe_degree,
        Geometry<dim> &geometry, Boundary const &boundary,
        MaterialProperty<dim, n_materials, p_order, MaterialStates,
                         MemorySpaceType> &material_properties,
        std::vector<double> const &reference_temperatures,
        bool const verbose_output)
    : _geometry(geometry), _boundary(boundary),
      _material_properties(material_properties),
      _dof_handler(_geometry.get_triangulation()),
      _reference_temperatures(reference_temperatures),
      _verbose_output(verbose_output), _solution_transfer(_dof_handler),
      _closest_quad_point_adaptation(dealii::QGauss<dim>(fe_degree + 1)),
      _cell_data_transfer(
          dynamic_cast<const dealii::parallel::distributed::Triangulation<dim>
                           &>(_dof_handler.get_triangulation()),
          /* transfer_variable_size_data */ false,
          [&](const typename dealii::Triangulation<dim>::cell_iterator &parent,
              const std::vector<std::vector<double>> &parent_values)
          {
            return _closest_quad_point_adaptation.coarse_to_fine(parent,
                                                                 parent_values);
          },
          [&](const typename dealii::Triangulation<dim>::cell_iterator &parent,
              const std::vector<std::vector<std::vector<double>>> &child_values)
          {
            return _closest_quad_point_adaptation.fine_to_coarse(parent,
                                                                 child_values);
          })
{
  _is_purely_elastic = _material_properties.has_infinite_elastic_limit();

  // Create the FECollection
  _fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Q<dim>(fe_degree) ^ dim));
  _fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Nothing<dim>() ^ dim));

  // Create the QCollection
  _q_collection.push_back(dealii::QGauss<dim>(fe_degree + 1));
  _q_collection.push_back(dealii::QGauss<dim>(1));

  // Solve the mechanical problem only on the part of the domain that has solid
  // material.
  unsigned int n_active_cells =
      _dof_handler.get_triangulation().n_active_cells();
  for (auto const &cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    if (_material_properties.get_state_ratio(
            cell, MaterialStates::State::solid) > 0.99)
    {
      cell->set_active_fe_index(0);
    }
    else
    {
      cell->set_active_fe_index(1);
    }
  }

  // Create the mechanical operator
  _mechanical_operator =
      std::make_unique<MechanicalOperator<dim, n_materials, p_order,
                                          MaterialStates, MemorySpaceType>>(
          communicator, _material_properties);

  // Create the constitutive state used to compute the stress tensor
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();
  _state.reserve(n_active_cells);

  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    std::vector<MechanicalConstitutiveState<dim>> cell_state(n_quad_pts);
    if (cell->is_locally_owned())
    {
      auto elastic_limit = _material_properties.get_mechanical_property(
          cell, StateProperty::elastic_limit);
      for (auto &state : cell_state)
        state.plastic_internal_variable = elastic_limit;
    }
    else
    {
      for (auto &state : cell_state)
        state.plastic_internal_variable =
            std::numeric_limits<double>::signaling_NaN();
    }
    _state.emplace_back(std::move(cell_state));
  }
  _trial_state.resize(
      n_active_cells,
      std::vector<MechanicalConstitutiveState<dim>>(n_quad_pts));
  _trial_stress.resize(
      n_active_cells, std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
  _trial_state = _state;
  _constitutive_tangent.resize(
      n_active_cells, std::vector<dealii::SymmetricTensor<4, dim>>(n_quad_pts));
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
std::vector<std::vector<dealii::SymmetricTensor<2, dim>>>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                  MemorySpaceType>::get_stress_tensor() const
{
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> stress(
      _state.size());

  dealii::hp::FEValues<dim> displacement_hp_fe_values(
      _fe_collection, _q_collection, dealii::update_gradients);
  unsigned int const n_q_points = _q_collection.max_n_quadrature_points();
  std::vector<dealii::SymmetricTensor<2, dim>> strain_tensor(n_q_points);
  const dealii::FEValuesExtractors::Vector displacement_extr(0);

  unsigned int cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    stress[cell_id].resize(_state[cell_id].size());

    if (cell->is_locally_owned() && cell->active_fe_index() == 0)
    {
      displacement_hp_fe_values.reinit(cell);
      auto const &fe_values = displacement_hp_fe_values.get_present_fe_values();
      if (_old_displacement.size() > 0)
        fe_values[displacement_extr].get_function_symmetric_gradients(
            _old_displacement, strain_tensor);

      double const lambda = _material_properties.get_mechanical_property(
          cell, StateProperty::lame_first_parameter);
      double const mu = _material_properties.get_mechanical_property(
          cell, StateProperty::lame_second_parameter);
      dealii::SymmetricTensor<4, dim> const stiffness_tensor =
          lambda * dealii::outer_product(dealii::unit_symmetric_tensor<dim>(),
                                         dealii::unit_symmetric_tensor<dim>()) +
          2 * mu * dealii::identity_tensor<dim>();

      for (auto const q : fe_values.quadrature_point_indices())
      {
        auto elastic_strain = strain_tensor[q] -
                              _state[cell_id][q].activation_strain -
                              _state[cell_id][q].plastic_strain;
        elastic_strain -= _state[cell_id][q].thermal_strain *
                          dealii::unit_symmetric_tensor<dim>();
        stress[cell_id][q] = stiffness_tensor * elastic_strain;
      }
    }
    ++cell_id;
  }
  return stress;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
std::tuple<MechanicalConstitutiveState<dim>, dealii::SymmetricTensor<2, dim>,
           dealii::SymmetricTensor<4, dim>>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    evaluate_constitutive_response(
        dealii::SymmetricTensor<2, dim> const &strain, double const temperature,
        double const reference_temperature, double const lambda,
        double const mu, double const plastic_modulus,
        double const iso_hardening_coef, double const alpha,
        bool const include_thermal_loading,
        MechanicalConstitutiveState<dim> const &state) const
{
  MechanicalConstitutiveState<dim> result = state;

  dealii::SymmetricTensor<4, dim> stiffness_tensor =
      lambda * dealii::outer_product(dealii::unit_symmetric_tensor<dim>(),
                                     dealii::unit_symmetric_tensor<dim>()) +
      2 * mu * dealii::identity_tensor<dim>();
  dealii::SymmetricTensor<4, dim> constitutive_tangent;

  // Compute the trial elastic stress from the strain decomposition
  // epsilon = epsilon_e + epsilon_p + epsilon_th.
  double current_thermal_strain = state.thermal_strain;
  if (include_thermal_loading)
  {
    current_thermal_strain = alpha * (temperature - reference_temperature);
    result.thermal_strain = current_thermal_strain;
  }
  auto elastic_strain = strain - state.activation_strain - state.plastic_strain;
  elastic_strain -=
      current_thermal_strain * dealii::unit_symmetric_tensor<dim>();
  auto const elastic_stress = stiffness_tensor * elastic_strain;
  dealii::SymmetricTensor<2, dim> trial_stress;

  // Formulation based on the combined isotropic-kinematic hardening model for
  // J2 plasticity in Chapter 3 of R. Borja, Plasticity: Modeling and
  // Computation, Springer-Verlag, 2013. DOI: 10.1007/978-3-642-38547-6
  auto const stress_deviator = dealii::deviator(elastic_stress);
  auto const effective_stress = stress_deviator - state.back_stress;
  double const effective_stress_norm = effective_stress.norm();
  if (effective_stress_norm <= state.plastic_internal_variable)
  {
    // The deformation is elastic.
    trial_stress = elastic_stress;
    constitutive_tangent = stiffness_tensor;
  }
  else
  {
    // The deformation is plastic. Apply the analytical radial return and
    // update only the trial internal variables.
    double const plastic_strain_increment =
        (effective_stress_norm - state.plastic_internal_variable) /
        (2. * mu + plastic_modulus);
    auto const plastic_flow_direction =
        effective_stress / effective_stress_norm;
    trial_stress = elastic_stress -
                   2. * mu * plastic_strain_increment * plastic_flow_direction;
    result.plastic_strain += plastic_strain_increment * plastic_flow_direction;
    result.plastic_internal_variable +=
        iso_hardening_coef * plastic_modulus * plastic_strain_increment;
    result.back_stress += (1. - iso_hardening_coef) * plastic_modulus *
                          plastic_strain_increment * plastic_flow_direction;

    auto const deviatoric_projection = dealii::deviator_tensor<dim>();
    auto const flow_direction_projection =
        dealii::outer_product(plastic_flow_direction, plastic_flow_direction);
    double const two_mu = 2. * mu;
    constitutive_tangent =
        stiffness_tensor -
        (two_mu * two_mu / (two_mu + plastic_modulus)) *
            flow_direction_projection -
        (two_mu * two_mu * plastic_strain_increment / effective_stress_norm) *
            (deviatoric_projection - flow_direction_projection);
  }

  return {result, trial_stress, constitutive_tangent};
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    setup_dofs(std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _dof_handler.distribute_dofs(_fe_collection);
  dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
  dealii::IndexSet locally_owned_dofs = _dof_handler.locally_owned_dofs();
  _affine_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler,
                                                  _affine_constraints);

  std::map<dealii::types::boundary_id, const dealii::Function<dim> *>
      boundary_function_map;
  dealii::Functions::ZeroFunction<dim> zero_function(dim);
  auto boundary_ids = _boundary.get_boundary_ids(BoundaryType::clamped);
  for (auto id : boundary_ids)
  {
    boundary_function_map[id] = &zero_function;
  }
  dealii::VectorTools::interpolate_boundary_values(
      _dof_handler, boundary_function_map, _affine_constraints);
  _affine_constraints.close();

  _mechanical_operator->reinit(_dof_handler, _affine_constraints, _q_collection,
                               body_forces);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    update_rhs(std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _mechanical_operator->assemble_rhs(body_forces);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    update_rhs(
        dealii::DoFHandler<dim> const &thermal_dof_handler,
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &temperature,
        std::vector<bool> const &has_melted,
        std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _thermal_dof_handler = &thermal_dof_handler;
  _temperature = temperature;
  _has_melted = has_melted;
  _mechanical_operator->assemble_rhs(body_forces);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::prepare_transfer_mpi()
{
  _old_displacement.update_ghost_values();
  _solution_transfer.prepare_for_coarsening_and_refinement(_old_displacement);

  _data_to_transfer.clear();
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();
  unsigned int const n_doubles_per_quad_scalar = 2;
  unsigned int const n_doubles_per_quad_tensor =
      dealii::SymmetricTensor<2, dim>::n_independent_components;

  unsigned int const n_doubles_per_quad =
      n_doubles_per_quad_scalar + n_doubles_per_quad_tensor * 3;
  std::vector<std::vector<double>> dummy_cell_data(
      n_quad_pts, std::vector<double>(n_doubles_per_quad,
                                      std::numeric_limits<double>::infinity()));
  std::vector<std::vector<double>> cell_data = dummy_cell_data;
  unsigned int cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      unsigned int const activation_strain_offset = n_doubles_per_quad_scalar;
      unsigned int const plastic_strain_offset =
          n_doubles_per_quad_scalar + n_doubles_per_quad_tensor;
      unsigned int const back_stress_offset =
          n_doubles_per_quad_scalar + 2 * n_doubles_per_quad_tensor;

      for (unsigned int quad = 0; quad < n_quad_pts; ++quad)
      {
        std::vector<double> &cell_data_quad = cell_data[quad];
        cell_data_quad[0] = _state[cell_id][quad].plastic_internal_variable;
        cell_data_quad[1] = _state[cell_id][quad].thermal_strain;
        for (unsigned int i = 0; i < n_doubles_per_quad_tensor; ++i)
        {
          cell_data_quad[activation_strain_offset + i] =
              _state[cell_id][quad].activation_strain.access_raw_entry(i);
          cell_data_quad[plastic_strain_offset + i] =
              _state[cell_id][quad].plastic_strain.access_raw_entry(i);
          cell_data_quad[back_stress_offset + i] =
              _state[cell_id][quad].back_stress.access_raw_entry(i);
        }
      }
      _data_to_transfer.push_back(cell_data);
    }
    else
    {
      _data_to_transfer.push_back(dummy_cell_data);
    }
    ++cell_id;
  }
  _cell_data_transfer.prepare_for_coarsening_and_refinement(_data_to_transfer);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::complete_transfer_mpi()
{
  _dof_handler.distribute_dofs(_fe_collection);

  const dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
  _old_displacement.reinit(_dof_handler.locally_owned_dofs(),
                           locally_relevant_dofs,
#if DEAL_II_VERSION_GTE(9, 7, 0)
                           _dof_handler.get_mpi_communicator()
#else
                           _dof_handler.get_communicator()
#endif
  );
  _solution_transfer.interpolate(_old_displacement);
  if (_old_displacement.size() > 0)
    _old_displacement.update_ghost_values();

  auto n_active_cells = _dof_handler.get_triangulation().n_active_cells();
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();

  _state.resize(n_active_cells,
                std::vector<MechanicalConstitutiveState<dim>>(n_quad_pts));
  _trial_state.resize(
      n_active_cells,
      std::vector<MechanicalConstitutiveState<dim>>(n_quad_pts));
  _trial_stress.resize(
      n_active_cells, std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
  _constitutive_tangent.resize(
      n_active_cells, std::vector<dealii::SymmetricTensor<4, dim>>(n_quad_pts));

  unsigned int const n_doubles_per_quad_scalar = 2;
  unsigned int const n_doubles_per_quad_tensor =
      dealii::SymmetricTensor<2, dim>::n_independent_components;

  unsigned int const n_doubles_per_quad =
      n_doubles_per_quad_scalar + n_doubles_per_quad_tensor * 3;
  std::vector<std::vector<std::vector<double>>> data_to_unpack(
      n_active_cells, std::vector<std::vector<double>>(
                          n_quad_pts, std::vector<double>(n_doubles_per_quad)));
  _cell_data_transfer.unpack(data_to_unpack);

  unsigned int cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      unsigned int const activation_strain_offset = n_doubles_per_quad_scalar;
      unsigned int const plastic_strain_offset =
          n_doubles_per_quad_scalar + n_doubles_per_quad_tensor;
      unsigned int const back_stress_offset =
          n_doubles_per_quad_scalar + 2 * n_doubles_per_quad_tensor;

      for (unsigned int quad = 0; quad < n_quad_pts; ++quad)
      {
        _state[cell_id][quad].plastic_internal_variable =
            data_to_unpack[cell_id][quad][0];
        _state[cell_id][quad].thermal_strain = data_to_unpack[cell_id][quad][1];
        for (unsigned int i = 0; i < n_doubles_per_quad_tensor; ++i)
        {
          _state[cell_id][quad].activation_strain.access_raw_entry(i) =
              data_to_unpack[cell_id][quad][activation_strain_offset + i];
          _state[cell_id][quad].plastic_strain.access_raw_entry(i) =
              data_to_unpack[cell_id][quad][plastic_strain_offset + i];
          _state[cell_id][quad].back_stress.access_raw_entry(i) =
              data_to_unpack[cell_id][quad][back_stress_offset + i];
        }
      }
    }
    ++cell_id;
  }
  _trial_state = _state;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    setup_dofs(
        dealii::DoFHandler<dim> const &thermal_dof_handler,
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &temperature,
        std::vector<bool> const &has_melted, bool rebuild_matrix,
        std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _thermal_dof_handler = &thermal_dof_handler;
  _temperature = temperature;
  _has_melted = has_melted;
  // Update the active fe indices, the plastic variables, and the displacement.
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();
  unsigned int cell_id = 0;
  std::vector<std::vector<double>> saved_old_displacement;
  std::vector<std::vector<MechanicalConstitutiveState<dim>>> tmp_state;
  std::vector<bool> newly_activated_cells(_state.size(), false);
  // The number of cells to activate/deactive should be small, so we can
  // already reserve the memory.
  unsigned int const n_dofs_per_cell = _fe_collection.max_dofs_per_cell();
  unsigned int const n_old_active_cells = _state.size();
  double const no_previous_displacement =
      std::numeric_limits<double>::infinity();
  std::vector<dealii::types::global_dof_index> global_dof_indices(
      n_dofs_per_cell);
  tmp_state.reserve(n_old_active_cells);
  // First we save _old_displacement if it exists
  if (_old_displacement.size())
  {
    _old_displacement.update_ghost_values();

    std::vector<double> cell_values(n_dofs_per_cell);
    saved_old_displacement.reserve(n_old_active_cells);
    for (auto const &cell : _dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        auto fe_index = cell->active_fe_index();
        if (fe_index == 0)
        {
          // The cell contains solid material, we need to save the displacement
          cell->get_dof_indices(global_dof_indices);
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          {
            cell_values[i] = _old_displacement[global_dof_indices[i]];
          }
        }
        else
        {
          // The cell does not contain material or it is liquid. The
          // displacement is ignored.
          cell_values.assign(n_dofs_per_cell, no_previous_displacement);
        }
        saved_old_displacement.push_back(cell_values);
      }
      else
      {
        saved_old_displacement.push_back(
            std::vector<double>(n_dofs_per_cell, no_previous_displacement));
      }
    }
  }

  // Now we can update the fe indices and the plastic variables.
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      auto current_fe_index = cell->active_fe_index();
      if (_material_properties.get_state_ratio(
              cell, MaterialStates::State::solid) > 0.99)
      {
        // Only enable the cell if it is also enabled for the thermal simulation
        // Get the thermal DoFHandler cell iterator
        dealii::DoFCellAccessor<dim, dim, false> thermal_cell(
            &(_dof_handler.get_triangulation()), cell->level(), cell->index(),
            &thermal_dof_handler);
        auto updated_fe_index = thermal_cell.active_fe_index();
        if (current_fe_index == updated_fe_index)
        {
          // The cells is unchanged, we just copy the constitutive state as-is.
          tmp_state.push_back(_state[cell_id]);
        }
        else
        {
          // The cell has solidified or material has been added. The new cells
          // are initialized with default values.
          auto elastic_limit = _material_properties.get_mechanical_property(
              cell, StateProperty::elastic_limit);
          std::vector<MechanicalConstitutiveState<dim>> state(n_quad_pts);
          for (auto &value : state)
            value.plastic_internal_variable = elastic_limit;
          tmp_state.push_back(std::move(state));

          newly_activated_cells[cell_id] =
              current_fe_index != 0 && updated_fe_index == 0;

          cell->set_active_fe_index(updated_fe_index);
          rebuild_matrix = true;
        }
      }
      else
      {
        if (current_fe_index == 0)
        {
          rebuild_matrix = true;
        }

        // The cell is liquid. We don't need to save the plastic variables.
        cell->set_active_fe_index(1);
        std::vector<MechanicalConstitutiveState<dim>> state(n_quad_pts);
        for (auto &value : state)
          value.plastic_internal_variable =
              std::numeric_limits<double>::signaling_NaN();
        tmp_state.push_back(std::move(state));
      }
    }
    else
    {
      std::vector<MechanicalConstitutiveState<dim>> state(n_quad_pts);
      for (auto &value : state)
        value.plastic_internal_variable =
            std::numeric_limits<double>::signaling_NaN();
      tmp_state.push_back(std::move(state));
    }
    ++cell_id;
  }

  // Check if we need to rebuild the matrix
  rebuild_matrix =
      dealii::Utilities::MPI::logical_or(rebuild_matrix,
#if DEAL_II_VERSION_GTE(9, 7, 0)
                                         _dof_handler.get_mpi_communicator()
#else
                                         _dof_handler.get_communicator()
#endif
      );
  // If we do not need to rebuild the matrix. Update the rhs and exit.
  if (!rebuild_matrix)
  {
    update_rhs(body_forces);
    _trial_state = _state;
    return;
  }

#ifdef ADAMANTINE_DEBUG
  auto connected_components = ConnectedComponents::find_components(
      _dof_handler, _boundary.get_boundary_ids(BoundaryType::clamped), 0);

  for (const auto &component : connected_components)
  {
    ASSERT(component._touches_target_boundary,
           "At least one component doesn't touch the boundary!");
  }
#endif

  _state.swap(tmp_state);
  _trial_state = _state;
  _trial_stress.resize(
      _state.size(), std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
  _constitutive_tangent.resize(
      _state.size(), std::vector<dealii::SymmetricTensor<4, dim>>(n_quad_pts));

  setup_dofs(body_forces);

  // Update _old_displacement if necessary
  const dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
  const dealii::IndexSet locally_owned_dofs = _dof_handler.locally_owned_dofs();
  _old_displacement.reinit(locally_owned_dofs, locally_relevant_dofs,
#if DEAL_II_VERSION_GTE(9, 7, 0)
                           _dof_handler.get_mpi_communicator()
#else
                           _dof_handler.get_communicator()
#endif
  );

  if (saved_old_displacement.size())
  {
    cell_id = 0;
    for (auto const &cell : _dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        auto fe_index = cell->active_fe_index();
        if (fe_index == 0 &&
            saved_old_displacement[cell_id][0] != no_previous_displacement)
        {
          cell->get_dof_indices(global_dof_indices);
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          {
            if (locally_owned_dofs.is_element(global_dof_indices[i]))
              _old_displacement[global_dof_indices[i]] =
                  saved_old_displacement[cell_id][i];
          }
        }
      }
      ++cell_id;
    }
    _old_displacement.compress(dealii::VectorOperation::insert);
  }

  if (_old_displacement.size() > 0)
    _old_displacement.update_ghost_values();

  // Record the strain already present when a cell becomes mechanically active.
  // The cell inherits shared displacement DoFs from neighboring solid cells,
  // but this deformation must not be interpreted as new elastic loading.
  dealii::hp::FEValues<dim> displacement_hp_fe_values(
      _fe_collection, _q_collection, dealii::update_gradients);
  std::vector<dealii::SymmetricTensor<2, dim>> activation_strain(n_quad_pts);
  const dealii::FEValuesExtractors::Vector displacement_extr(0);
  cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (newly_activated_cells[cell_id] && cell->is_locally_owned() &&
        _old_displacement.size() > 0)
    {
      displacement_hp_fe_values.reinit(cell);
      auto const &fe_values = displacement_hp_fe_values.get_present_fe_values();
      fe_values[displacement_extr].get_function_symmetric_gradients(
          _old_displacement, activation_strain);
      for (auto const q : fe_values.quadrature_point_indices())
        _state[cell_id][q].activation_strain = activation_strain[q];
    }
    ++cell_id;
  }
  _trial_state = _state;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    solve_linear_system(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &rhs,
        double const tolerance) const
{
  dealii::IndexSet locally_owned_dofs = _dof_handler.locally_owned_dofs();
  dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
  using TrilinosVectorType = dealii::LinearAlgebra::TpetraWrappers::Vector<
      double, dealii::MemorySpace::Default>;
#else
  using TrilinosVectorType = dealii::TrilinosWrappers::MPI::Vector;
#endif
  TrilinosVectorType displacement(
      locally_owned_dofs, _mechanical_operator->rhs().get_mpi_communicator());
  TrilinosVectorType rhs_device(
      locally_owned_dofs, _mechanical_operator->rhs().get_mpi_communicator());
  dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

  rw_vector.import_elements(rhs, dealii::VectorOperation::insert);
  rhs_device.import_elements(rw_vector, dealii::VectorOperation::insert);

  unsigned int const max_iter = _dof_handler.n_dofs() / 10;
  dealii::SolverControl solver_control(max_iter, tolerance);
  dealii::SolverCG<TrilinosVectorType> cg(solver_control);
  cg.solve(_mechanical_operator->system_matrix(), displacement, rhs_device,
           _mechanical_operator->preconditioner());

  rw_vector.import_elements(displacement, dealii::VectorOperation::insert);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      displacement_host(locally_owned_dofs, locally_relevant_dofs,
                        _mechanical_operator->rhs().get_mpi_communicator());
  displacement_host.import_elements(rw_vector, dealii::VectorOperation::insert);
  _affine_constraints.distribute(displacement_host);

  return displacement_host;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                  MemorySpaceType>::solve_nonlinear_system()
{
  using VectorType =
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>;
  using KINSOL = dealii::SUNDIALS::KINSOL<VectorType>;

  dealii::IndexSet const locally_owned_dofs = _dof_handler.locally_owned_dofs();
  dealii::IndexSet const locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
  MPI_Comm const communicator =
      _mechanical_operator->rhs().get_mpi_communicator();

  // Use the elastic solution as the initial guess. The residual at zero
  // increment includes the current stress and body forces.
  VectorType zero_increment(locally_owned_dofs, locally_relevant_dofs,
                            communicator);
  zero_increment.update_ghost_values();
  auto initial_rhs = evaluate_residual(zero_increment);
  initial_rhs *= -1.;
  double const linear_tolerance = 1e-12 * initial_rhs.l2_norm();
  auto displacement_increment =
      solve_linear_system(initial_rhs, linear_tolerance);
  auto displacement = displacement_increment;
  if (_old_displacement.size() > 0)
  {
    _old_displacement.update_ghost_values();
    displacement += _old_displacement;
  }
  displacement.update_ghost_values();

  typename KINSOL::AdditionalData additional_data;
  // Tolerance is scaled by the norm of the initial residual. We use the
  // infinity norm to match KINSOL stopping criteria for the non-linear solver.
  auto const initial_increment = compute_incremental_displacement(displacement);
  additional_data.function_tolerance = std::max(
      1.e-10 * evaluate_residual(initial_increment).linfty_norm(), 1.e-12);
  KINSOL nonlinear_solver(additional_data, communicator);

  nonlinear_solver.reinit_vector = [locally_owned_dofs, locally_relevant_dofs,
                                    communicator](VectorType &vector)
  { vector.reinit(locally_owned_dofs, locally_relevant_dofs, communicator); };

  nonlinear_solver.residual = [this](VectorType const &src, VectorType &dst)
  {
    auto const incremental_displacement = compute_incremental_displacement(src);
    dst = evaluate_residual(incremental_displacement);
  };

  nonlinear_solver.setup_jacobian =
      [this](VectorType const &current_u, VectorType const &)
  {
    auto const incremental_displacement =
        compute_incremental_displacement(current_u);
    evaluate_trial_state(incremental_displacement);
    _mechanical_operator->assemble_jacobian(_constitutive_tangent);
  };
  nonlinear_solver.solve_with_jacobian =
      [this](VectorType const &rhs, VectorType &dst, double const tolerance)
  { dst = solve_linear_system(rhs, std::max(tolerance, 1e-14)); };

  unsigned int const nonlinear_iterations =
      nonlinear_solver.solve(displacement);
  if (_verbose_output &&
      dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    std::cout << "Number of nonlinear iterations: " << nonlinear_iterations
              << std::endl;

  return displacement;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                  MemorySpaceType>::solve()
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("solve mechanical system");
#endif

  _mechanical_operator->ensure_elastic_matrix();

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      displacement_host;
  if (_is_purely_elastic)
  {
    dealii::IndexSet const locally_owned_dofs =
        _dof_handler.locally_owned_dofs();
    dealii::IndexSet const locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        zero_displacement(locally_owned_dofs, locally_relevant_dofs,
                          _mechanical_operator->rhs().get_mpi_communicator());
    zero_displacement.update_ghost_values();

    // For an elastic constitutive law the residual is affine in the
    // displacement increment. Evaluating it at zero displacement gives the
    // right-hand side for the unchanged elastic matrix without introducing a
    // separate thermal load vector.
    auto rhs = evaluate_residual(zero_displacement);
    rhs *= -1.;
    double const tolerance = 1e-12 * rhs.l2_norm();
    displacement_host = solve_linear_system(rhs, tolerance);
    if (_old_displacement.size() > 0)
    {
      _old_displacement.update_ghost_values();
      displacement_host += _old_displacement;
    }
    displacement_host.update_ghost_values();
  }
  else
  {
    displacement_host = solve_nonlinear_system();
  }

  // Evaluate the trial state using the displacement increment from the
  // previous time step. If the trial stress is under the yield criterion, the
  // deformation is elastic and we are done. Otherwise we need to use the
  // radial return algorithm to compute the plastic deformation.
  auto const incremental_displacement =
      compute_incremental_displacement(displacement_host);
  evaluate_trial_state(incremental_displacement);

  // Commit the trial material state and displacement history after the
  // mechanical solve has converged.
  _state = _trial_state;
  if (_old_displacement.size() == 0)
  {
    dealii::IndexSet const locally_owned_dofs =
        _dof_handler.locally_owned_dofs();
    dealii::IndexSet const locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
    _old_displacement.reinit(
        locally_owned_dofs, locally_relevant_dofs,
        _mechanical_operator->rhs().get_mpi_communicator());
  }
  _old_displacement = displacement_host;

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_END("solve mechanical system");
#endif

  return _old_displacement;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    compute_incremental_displacement(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &total_displacement) const
{
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      incremental_displacement = total_displacement;
  if (_old_displacement.size() > 0)
    incremental_displacement -= _old_displacement;
  incremental_displacement.update_ghost_values();
  return incremental_displacement;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    evaluate_residual(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &displacement_increment)
{
  evaluate_trial_state(displacement_increment);
  return _mechanical_operator->assemble_residual(_trial_stress);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    evaluate_trial_state(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &incremental_displacement)
{
  dealii::hp::FEValues<dim> displacement_hp_fe_values(
      _fe_collection, _q_collection, dealii::update_gradients);
  unsigned int const n_q_points = _q_collection.max_n_quadrature_points();
  std::vector<dealii::SymmetricTensor<2, dim>> strain_tensor(n_q_points);
  std::vector<dealii::SymmetricTensor<2, dim>> previous_strain_tensor(
      n_q_points);
  std::vector<double> temperature_values(n_q_points);
  const dealii::FEValuesExtractors::Vector displacement_extr(0);
  std::unique_ptr<dealii::hp::FEValues<dim>> temperature_hp_fe_values;
  std::vector<unsigned int> cell_indices;
  // When solving a thermomechanical problem, we create mapping between the
  // active cells of the mechanical problem and the active cells of the thermal
  // problem. Liquid cells are active in the thermal problem but not in the
  // mechanical problem.
  if (!_reference_temperatures.empty())
  {
    _temperature.update_ghost_values();
    temperature_hp_fe_values = std::make_unique<dealii::hp::FEValues<dim>>(
        _thermal_dof_handler->get_fe_collection(), _q_collection,
        dealii::update_values);

    auto &triangulation = _dof_handler.get_triangulation();
    cell_indices.resize(triangulation.n_active_cells());
    unsigned int thermal_cell_index = 0;
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
                              tria_cell->index(), &_dof_handler);
        if (displacement_cell->active_fe_index() == 0)
          cell_indices[displacement_cell->active_cell_index()] =
              thermal_cell_index;
        ++thermal_cell_index;
      }
    }
  }
  unsigned int cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned() && cell->active_fe_index() == 0)
    {
      // Compute the strain. We get the strain for all the quadrature points at
      // once.
      displacement_hp_fe_values.reinit(cell);
      auto const &fe_values = displacement_hp_fe_values.get_present_fe_values();

      fe_values[displacement_extr].get_function_symmetric_gradients(
          incremental_displacement, strain_tensor);
      if (_old_displacement.size() > 0)
        fe_values[displacement_extr].get_function_symmetric_gradients(
            _old_displacement, previous_strain_tensor);
      else
        std::fill(previous_strain_tensor.begin(), previous_strain_tensor.end(),
                  dealii::SymmetricTensor<2, dim>());

      double reference_temperature = 0.;
      if (!_reference_temperatures.empty())
      {
        auto &triangulation = _dof_handler.get_triangulation();
        dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
            temperature_cell(&triangulation, cell->level(), cell->index(),
                             _thermal_dof_handler);
        temperature_hp_fe_values->reinit(temperature_cell);
        auto const &temperature_fe_values =
            temperature_hp_fe_values->get_present_fe_values();
        temperature_fe_values.get_function_values(_temperature,
                                                  temperature_values);
        reference_temperature =
            _has_melted[cell_indices[cell->active_cell_index()]]
                ? _reference_temperatures[temperature_cell->material_id()]
                : _reference_temperatures.back();
      }

      double const lambda = _material_properties.get_mechanical_property(
          cell, StateProperty::lame_first_parameter);
      double const mu = _material_properties.get_mechanical_property(
          cell, StateProperty::lame_second_parameter);
      double const plastic_modulus =
          _material_properties.get_mechanical_property(
              cell, StateProperty::plastic_modulus);
      double const iso_hardening_coef =
          _material_properties.get_mechanical_property(
              cell, StateProperty::isotropic_hardening);
      double const alpha = _material_properties.get_mechanical_property(
          cell, StateProperty::thermal_expansion_coef);
      // Loop over the quadrature points.
      for (auto const q : fe_values.quadrature_point_indices())
      {
        double const temperature =
            _reference_temperatures.empty() ? 0. : temperature_values[q];
        auto const [trial_state, stress, constitutive_tangent] =
            evaluate_constitutive_response(
                previous_strain_tensor[q] + strain_tensor[q], temperature,
                reference_temperature, lambda, mu, plastic_modulus,
                iso_hardening_coef, alpha, !_reference_temperatures.empty(),
                _state[cell_id][q]);

        _trial_state[cell_id][q] = trial_state;
        _trial_stress[cell_id][q] = stress;
        _constitutive_tangent[cell_id][q] = constitutive_tangent;
      }
    }
    ++cell_id;
  }
}

} // namespace adamantine

INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_HOST(MechanicalPhysics)
INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_DEVICE(MechanicalPhysics)
