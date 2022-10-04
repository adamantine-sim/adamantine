/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperator.hh>
#include <instantiation.hh>
#include <utils.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace adamantine
{

template <int dim, int fe_degree, typename MemorySpaceType>
ThermalOperator<dim, fe_degree, MemorySpaceType>::ThermalOperator(
    MPI_Comm const &communicator, BoundaryType boundary_type,
    MaterialProperty<dim, MemorySpaceType> &material_properties,
    std::vector<std::shared_ptr<HeatSource<dim>>> const &heat_sources)
    : _communicator(communicator), _boundary_type(boundary_type),
      _material_properties(material_properties), _heat_sources(heat_sources),
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
  _matrix_free.reinit(dealii::StaticMappingQ1<dim>::mapping, dof_handler,
                      affine_constraints, q_collection, _matrix_free_data);
  _affine_constraints = &affine_constraints;

  // Compute mapping between DoFHandler cells and the MatrixFree cells
  _cell_it_to_mf_cell_map.clear();
  unsigned int const n_cells = _matrix_free.n_cell_batches();
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int i = 0;
         i < _matrix_free.n_active_entries_per_cell_batch(cell); ++i)
    {
      typename dealii::DoFHandler<dim>::cell_iterator cell_it =
          _matrix_free.get_cell_iterator(cell, i);
      _cell_it_to_mf_cell_map[cell_it] = std::make_pair(cell, i);
    }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::cell_local_mass(
    dealii::MatrixFree<dim, double> const &data,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
    std::pair<unsigned int, unsigned int> const &cell_range) const
{
  // Get the subrange of cells associated with the fe index 0
  std::pair<unsigned int, unsigned int> cell_subrange =
      data.create_cell_subrange_hp_by_index(cell_range, 0);
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(data);

  // Loop over the "cells". Note that we don't really work on a cell but on a
  // set of quadrature point.
  for (unsigned int cell = cell_subrange.first; cell < cell_subrange.second;
       ++cell)
  {
    // Reinit fe_eval on the current cell
    fe_eval.reinit(cell);
    // Store in a local vector the local values of src
    fe_eval.read_dof_values(src);
    // Evaluate the shape function on the reference cell
    fe_eval.evaluate(dealii::EvaluationFlags::values);
    // Apply the Jacobian of the transformation
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value(fe_eval.get_value(q), q);
    }
    // Sum over the quadrature points.
    fe_eval.integrate(dealii::EvaluationFlags::values);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    compute_inverse_mass_matrix(
        dealii::DoFHandler<dim> const &dof_handler,
        dealii::AffineConstraints<double> const &affine_constraints)
{
  // Compute the inverse of the mass matrix
  dealii::hp::QCollection<dim> mass_q_collection;
  mass_q_collection.push_back(dealii::QGaussLobatto<dim>(fe_degree + 1));
  mass_q_collection.push_back(dealii::QGaussLobatto<dim>(2));

  typename dealii::MatrixFree<dim, double>::AdditionalData
      mass_matrix_free_data;
  mass_matrix_free_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, double>::AdditionalData::partition_color;
  mass_matrix_free_data.mapping_update_flags =
      dealii::update_values | dealii::update_JxW_values;

  dealii::MatrixFree<dim, double> mass_matrix_free;
  mass_matrix_free.reinit(dealii::StaticMappingQ1<dim>::mapping, dof_handler,
                          affine_constraints, mass_q_collection,
                          mass_matrix_free_data);
  mass_matrix_free.initialize_dof_vector(*_inverse_mass_matrix);
  dealii::LA::distributed::Vector<double, MemorySpaceType> unit_vector;
  mass_matrix_free.initialize_dof_vector(unit_vector);
  unit_vector = 1.;
  mass_matrix_free.cell_loop(&ThermalOperator::cell_local_mass, this,
                             *_inverse_mass_matrix, unit_vector);
  // Because cell_loop resolves the constraints, the constrained dofs are not
  // called they stay at zero. Thus, we need to force the value on the
  // constrained dofs by hand.
  std::vector<unsigned int> const &constrained_dofs =
      mass_matrix_free.get_constrained_dofs();
  for (auto &dof : constrained_dofs)
    _inverse_mass_matrix->local_element(dof) += 1.;

  _inverse_mass_matrix->compress(dealii::VectorOperation::add);

  unsigned int const local_size = _inverse_mass_matrix->locally_owned_size();
  for (unsigned int k = 0; k < local_size; ++k)
  {
    _inverse_mass_matrix->local_element(k) =
        1. / _inverse_mass_matrix->local_element(k);
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::clear()
{
  _cell_it_to_mf_cell_map.clear();
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
            conv_temperature_infty = _material_properties.get_cell_value(
                cell, Property::convection_temperature_infty);
            conv_heat_transfer_coef = _material_properties.get_cell_value(
                cell, StateProperty::convection_heat_transfer_coef);
          }
          if (_boundary_type & BoundaryType::radiative)
          {
            rad_temperature_infty = _material_properties.get_cell_value(
                cell, Property::radiation_temperature_infty);
            rad_heat_transfer_coef = _material_properties.get_cell_value(
                cell, StateProperty::radiation_heat_transfer_coef);
          }

          fe_face_values.reinit(cell, face);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                // FIXME Need to be aware that we are using face quadrature
                // points not volumetric quadrature point. Right now we accept
                // this slight error.
                double const inv_rho_cp = get_inv_rho_cp(cell, q);

                cell_dst[i] -= inv_rho_cp *
                               (conv_heat_transfer_coef *
                                    (cell_src[j] - conv_temperature_infty) +
                                rad_heat_transfer_coef *
                                    (cell_src[j] - rad_temperature_infty)) *
                               fe_face_values.shape_value(i, q) *
                               fe_face_values.shape_value(j, q) *
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
void ThermalOperator<dim, fe_degree, MemorySpaceType>::update_state_ratios(
    unsigned int cell, unsigned int q,
    dealii::VectorizedArray<double> temperature,
    std::array<dealii::VectorizedArray<double>,
               static_cast<unsigned int>(MaterialState::SIZE)> &state_ratios)
    const
{
  unsigned int constexpr liquid =
      static_cast<unsigned int>(MaterialState::liquid);
  unsigned int constexpr powder =
      static_cast<unsigned int>(MaterialState::powder);
  unsigned int constexpr solid =
      static_cast<unsigned int>(MaterialState::solid);

  // Loop over the vectorized arrays
  for (unsigned int n = 0; n < temperature.size(); ++n)
  {
    // Get the material id at this point
    dealii::types::material_id const material_id = _material_id(cell, q)[n];

    // Get the material thermodynamic properties
    double const solidus =
        _material_properties.get(material_id, Property::solidus);
    double const liquidus =
        _material_properties.get(material_id, Property::liquidus);

    // Update the state ratios
    state_ratios[powder] = _powder_ratio(cell, q);

    if (temperature[n] < solidus)
      state_ratios[liquid][n] = 0.;
    else if (temperature[n] > liquidus)
      state_ratios[liquid][n] = 1.;
    else
    {
      state_ratios[liquid][n] =
          (temperature[n] - solidus) / (liquidus - solidus);
    }
    // Because the powder can only become liquid, the solid can only
    // become liquid, and the liquid can only become solid, the ratio of
    // powder can only decrease.
    state_ratios[powder][n] =
        std::min(1. - state_ratios[liquid][n], state_ratios[powder][n]);
    // Use max to make sure that we don't create matter because of
    // round-off.
    state_ratios[solid][n] =
        std::max(1. - state_ratios[liquid][n] - state_ratios[powder][n], 0.);
  }

  _liquid_ratio(cell, q) = state_ratios[liquid];
  _powder_ratio(cell, q) = state_ratios[powder];
}

template <int dim, int fe_degree, typename MemorySpaceType>
dealii::VectorizedArray<double>
ThermalOperator<dim, fe_degree, MemorySpaceType>::get_inv_rho_cp(
    unsigned int cell, unsigned int q,
    std::array<dealii::VectorizedArray<double>,
               static_cast<unsigned int>(MaterialState::SIZE)>
        state_ratios,
    dealii::VectorizedArray<double> temperature) const
{
  // Here we need the specific heat (including the latent heat contribution)
  // and the density

  auto material_id = _material_id(cell, q);
  // First, get the state-independent material properties
  dealii::VectorizedArray<double> solidus, liquidus, latent_heat;
  for (unsigned int n = 0; n < solidus.size(); ++n)
  {
    solidus[n] = _material_properties.get(material_id[n], Property::solidus);
    liquidus[n] = _material_properties.get(material_id[n], Property::liquidus);
    latent_heat[n] =
        _material_properties.get(material_id[n], Property::latent_heat);
  }

  // Now compute the state-dependent properties
  dealii::VectorizedArray<double> density =
      _material_properties.compute_material_property(
          StateProperty::density, material_id.data(), state_ratios.data(),
          temperature);

  dealii::VectorizedArray<double> specific_heat =
      _material_properties.compute_material_property(
          StateProperty::specific_heat, material_id.data(), state_ratios.data(),
          temperature);

  // Add in the latent heat contribution
  unsigned int constexpr liquid =
      static_cast<unsigned int>(MaterialState::liquid);

  for (unsigned int n = 0; n < specific_heat.size(); ++n)
  {
    if (state_ratios[liquid][n] > 0.0 && (state_ratios[liquid][n] < 1.0))
    {
      specific_heat[n] += latent_heat[n] / (liquidus[n] - solidus[n]);
    }
  }

  _inv_rho_cp(cell, q) = 1.0 / (density * specific_heat);

  return _inv_rho_cp(cell, q);
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

  std::array<dealii::VectorizedArray<double>,
             static_cast<unsigned int>(MaterialState::SIZE)>
      state_ratios = {{dealii::make_vectorized_array(-1.0),
                       dealii::make_vectorized_array(-1.0),
                       dealii::make_vectorized_array(-1.0)}};

  // Loop over the "cells". Note that we don't really work on a cell but on a
  // set of quadrature point.
  for (unsigned int cell = cell_subrange.first; cell < cell_subrange.second;
       ++cell)
  {
    // Reinit fe_eval on the current cell
    fe_eval.reinit(cell);
    // Store in a local vector the local values of src
    fe_eval.read_dof_values(src);
    // Evaluate the function and its gradient on the reference cell
    fe_eval.evaluate(dealii::EvaluationFlags::values |
                     dealii::EvaluationFlags::gradients);
    // Apply the Jacobian of the transformation, multiply by the variable
    // coefficients and the quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      auto temperature = fe_eval.get_value(q);

      // Calculate the local material properties
      update_state_ratios(cell, q, temperature, state_ratios);
      auto inv_rho_cp = get_inv_rho_cp(cell, q, state_ratios, temperature);
      auto mat_id = _material_id(cell, q);
      auto th_conductivity_grad = fe_eval.get_gradient(q);

      // In 2D we only use x and z, and there are no deposition angle
      if constexpr (dim == 2)
      {
        th_conductivity_grad[axis<dim>::x] *=
            _material_properties.compute_material_property(
                StateProperty::thermal_conductivity_x, mat_id.data(),
                state_ratios.data(), temperature);
        th_conductivity_grad[axis<dim>::z] *=
            _material_properties.compute_material_property(
                StateProperty::thermal_conductivity_z, mat_id.data(),
                state_ratios.data(), temperature);
      }

      if constexpr (dim == 3)
      {
        auto const th_conductivity_grad_x = th_conductivity_grad[axis<dim>::x];
        auto const th_conductivity_grad_y = th_conductivity_grad[axis<dim>::y];
        auto const thermal_conductivity_x =
            _material_properties.compute_material_property(
                StateProperty::thermal_conductivity_x, mat_id.data(),
                state_ratios.data(), temperature);
        auto const thermal_conductivity_y =
            _material_properties.compute_material_property(
                StateProperty::thermal_conductivity_y, mat_id.data(),
                state_ratios.data(), temperature);

        auto cos = _deposition_cos(cell, q);
        auto sin = _deposition_sin(cell, q);

        // The rotation is performed using the following formula
        //
        // (cos  -sin) (x  0) ( cos  sin)
        // (sin   cos) (0  y) (-sin  cos)
        // =
        // ((x*cos^2 + y*sin^2)  ((x-y) * (sin*cos)))
        // (((x-y) * (sin*cos))  (x*sin^2 + y*cos^2))

        th_conductivity_grad[axis<dim>::x] =
            (thermal_conductivity_x * cos * cos +
             thermal_conductivity_y * sin * sin) *
                th_conductivity_grad_x +
            ((thermal_conductivity_x - thermal_conductivity_y) * sin * cos) *
                th_conductivity_grad_y;
        th_conductivity_grad[axis<dim>::y] =
            ((thermal_conductivity_x - thermal_conductivity_y) * sin * cos) *
                th_conductivity_grad_x +
            (thermal_conductivity_x * sin * sin +
             thermal_conductivity_y * cos * cos) *
                th_conductivity_grad_y;

        // There is no deposition angle for the z axis
        th_conductivity_grad[axis<dim>::z] *=
            _material_properties.compute_material_property(
                StateProperty::thermal_conductivity_z, mat_id.data(),
                state_ratios.data(), temperature);
      }

      fe_eval.submit_gradient(-inv_rho_cp * th_conductivity_grad, q);

      // Compute source term
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
          quad_pt_source[i] += beam->value(q_point_loc, _current_source_height);
      }
      quad_pt_source *= inv_rho_cp;

      fe_eval.submit_value(quad_pt_source, q);
    }
    // Sum over the quadrature points.
    fe_eval.integrate(dealii::EvaluationFlags::values |
                      dealii::EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree,
                     MemorySpaceType>::get_state_from_material_properties()
{
  unsigned int const n_cells = _matrix_free.n_cell_batches();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      _matrix_free);

  _liquid_ratio.reinit(n_cells, fe_eval.n_q_points);
  _powder_ratio.reinit(n_cells, fe_eval.n_q_points);
  _material_id.reinit(n_cells, fe_eval.n_q_points);
  _inv_rho_cp.reinit(n_cells, fe_eval.n_q_points);

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

        _liquid_ratio(cell, q)[i] = _material_properties.get_state_ratio(
            cell_tria, MaterialState::liquid);
        _powder_ratio(cell, q)[i] = _material_properties.get_state_ratio(
            cell_tria, MaterialState::powder);
        _material_id(cell, q)[i] = cell_tria->material_id();
      }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree,
                     MemorySpaceType>::set_state_to_material_properties()
{
  _material_properties.set_state(_liquid_ratio, _powder_ratio,
                                 _cell_it_to_mf_cell_map,
                                 _matrix_free.get_dof_handler());
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    update_boundary_material_properties(
        dealii::LA::distributed::Vector<double, MemorySpaceType> const
            &temperature)
{
  if (!(_boundary_type & BoundaryType::adiabatic))
    _material_properties.update_boundary_material_properties(
        _matrix_free.get_dof_handler(), temperature);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    set_material_deposition_orientation(
        std::vector<double> const &deposition_cos,
        std::vector<double> const &deposition_sin)
{
  unsigned int const n_cells = _matrix_free.n_cell_batches();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      _matrix_free);

  _deposition_cos.reinit(n_cells, fe_eval.n_q_points);
  _deposition_sin.reinit(n_cells, fe_eval.n_q_points);

  using dof_cell_iterator = typename dealii::DoFHandler<dim>::cell_iterator;
  std::map<dof_cell_iterator, unsigned int> cell_mapping;
  unsigned int pos = 0;
  for (auto const &cell : dealii::filter_iterators(
           _matrix_free.get_dof_handler().active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    cell_mapping[cell] = pos;
    ++pos;
  }
  ASSERT((pos == 0) || (pos - 1 < deposition_cos.size()),
         "Out-of-bound access.");

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      for (unsigned int i = 0;
           i < _matrix_free.n_active_entries_per_cell_batch(cell); ++i)
      {
        dof_cell_iterator cell_it = _matrix_free.get_cell_iterator(cell, i);

        if (cell_it->active_fe_index() == 0)
        {
          unsigned int const j = cell_mapping[cell_it];
          _deposition_cos(cell, q)[i] = deposition_cos[j];
          _deposition_sin(cell, q)[i] = deposition_sin[j];
        }
      }
}

} // namespace adamantine

INSTANTIATE_DIM_FEDEGREE_HOST(TUPLE(ThermalOperator))
