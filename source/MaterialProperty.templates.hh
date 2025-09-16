/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MATERIAL_PROPERTY_TEMPLATES_HH
#define MATERIAL_PROPERTY_TEMPLATES_HH

#include <MaterialProperty.hh>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/algorithm/string/split.hpp>
#include <boost/optional.hpp>

#include <Kokkos_Core_fwd.hpp>

#include <algorithm>
#include <type_traits>

namespace adamantine
{
namespace internal
{
template <int dim>
void compute_average(
    unsigned int const n_q_points, unsigned int const dofs_per_cell,
    dealii::DoFHandler<dim> const &mp_dof_handler,
    dealii::DoFHandler<dim> const &temperature_dof_handler,
    dealii::hp::FEValues<dim> &hp_fe_values,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &temperature,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        &temperature_average)
{
  std::vector<dealii::types::global_dof_index> mp_dof_indices(1);
  std::vector<dealii::types::global_dof_index> enth_dof_indices(dofs_per_cell);
  auto mp_cell = mp_dof_handler.begin_active();
  auto mp_end_cell = mp_dof_handler.end();
  auto enth_cell = temperature_dof_handler.begin_active();
  for (; mp_cell != mp_end_cell; ++enth_cell, ++mp_cell)
  {
    ASSERT(mp_cell->is_locally_owned() == enth_cell->is_locally_owned(),
           "Internal Error");
    if ((mp_cell->is_locally_owned()) && (enth_cell->active_fe_index() == 0))
    {
      hp_fe_values.reinit(enth_cell);
      dealii::FEValues<dim> const &fe_values =
          hp_fe_values.get_present_fe_values();
      mp_cell->get_dof_indices(mp_dof_indices);
      dealii::types::global_dof_index const mp_dof_index = mp_dof_indices[0];
      enth_cell->get_dof_indices(enth_dof_indices);
      double volume = 0.;
      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          volume += fe_values.shape_value(i, q) * fe_values.JxW(q);
          temperature_average[mp_dof_index] +=
              fe_values.shape_value(i, q) * temperature[enth_dof_indices[i]] *
              fe_values.JxW(q);
        }
      temperature_average[mp_dof_index] /= volume;
    }
  }
}

template <typename ViewType,
          std::enable_if_t<
              std::is_same_v<typename ViewType::memory_space,
                             typename dealii::MemorySpace::Host::kokkos_space>,
              int> = 0>
double get_value(ViewType &view, unsigned int i, unsigned int j)
{
  return view(i, j);
}

template <int dim>
void compute_average(
    unsigned int const n_q_points, unsigned int const dofs_per_cell,
    dealii::DoFHandler<dim> const &mp_dof_handler,
    dealii::DoFHandler<dim> const &temperature_dof_handler,
    dealii::hp::FEValues<dim> &hp_fe_values,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> const
        &temperature,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default>
        &temperature_average)
{
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature_host(temperature.get_partitioner());
  temperature_host.import(temperature, dealii::VectorOperation::insert);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature_average_host(temperature_average.get_partitioner());
  temperature_average_host = 0.;
  compute_average(n_q_points, dofs_per_cell, mp_dof_handler,
                  temperature_dof_handler, hp_fe_values, temperature_host,
                  temperature_average_host);

  temperature_average.import(temperature_average_host,
                             dealii::VectorOperation::insert);
}

template <typename ViewType,
          std::enable_if_t<
              !std::is_same_v<typename ViewType::memory_space,
                              typename dealii::MemorySpace::Host::kokkos_space>,
              int> = 0>
double get_value(ViewType &view, unsigned int i, unsigned int j)
{
  auto view_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

  return view_host(i, j);
}
} // namespace internal

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    MaterialProperty(
        MPI_Comm const &communicator,
        dealii::parallel::distributed::Triangulation<dim> const &tria,
        boost::property_tree::ptree const &database)
    : _communicator(communicator), _fe(0), _mp_dof_handler(tria)
{
  // Because deal.II cannot easily attach data to a cell. We store the state
  // of the material in distributed::Vector. This allows to use deal.II to
  // compute the new state after refinement of the mesh. However, this
  // requires to use another DoFHandler.
  reinit_dofs();

  // Set the material state to the state defined in the geometry.
  set_initial_state();

  // Fill the _properties map
  fill_properties(database);
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
double
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::get_cell_value(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    StateProperty prop) const
{
  unsigned int property = static_cast<unsigned int>(prop);
  auto const mp_dof_index = get_dof_index(cell);

  // FIXME this is extremely slow on CUDA but this function should not exist in
  // the first place
  return internal::get_value(_property_values, property, mp_dof_index);
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
double
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::get_cell_value(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    Property prop) const
{
  dealii::types::material_id material_id = cell->material_id();
  unsigned int property = static_cast<unsigned int>(prop);

  // FIXME this is extremely slow on CUDA but this function should not exist in
  // the first place
  return internal::get_value(_properties, material_id, property);
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
double MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    get_mechanical_property(
        typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
        StateProperty prop) const
{
  unsigned int property =
      static_cast<unsigned int>(prop) - g_n_thermal_state_properties;
  ASSERT(property < g_n_mechanical_state_properties,
         "Unknown mechanical property requested.");
  return _mechanical_properties_host(cell->material_id(), property);
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
double MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    get_state_ratio(
        typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
        typename MaterialStates::State material_state) const
{
  auto const mp_dof_index = get_dof_index(cell);
  auto const mat_state = static_cast<unsigned int>(material_state);

  // FIXME this is extremely slow on CUDA but this function should not exist in
  // the first place
  return internal::get_value(_state, mat_state, mp_dof_index);
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates,
                      MemorySpaceType>::reinit_dofs()
{
  _mp_dof_handler.distribute_dofs(_fe);

  // Initialize _dofs_map
  _dofs_map.clear();
  unsigned int i = 0;
  std::vector<dealii::types::global_dof_index> mp_dof(1);
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    cell->get_dof_indices(mp_dof);
    _dofs_map[mp_dof[0]] = i;
    ++i;
  }

  _state = Kokkos::View<double **, typename MemorySpaceType::kokkos_space>(
      "state", MaterialStates::n_material_states, _dofs_map.size());
#ifdef ADAMANTINE_DEBUG
  if constexpr (std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>)
  {
    Kokkos::deep_copy(_state, std::numeric_limits<double>::signaling_NaN());
  }
#endif
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::update(
    dealii::DoFHandler<dim> const &temperature_dof_handler,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &temperature)
{
  auto temperature_average =
      compute_average_temperature(temperature_dof_handler, temperature);
  // Set View to zero in purpose
  _property_values =
      Kokkos::View<double **, typename MemorySpaceType::kokkos_space>(
          "property_values", g_n_thermal_state_properties, _dofs_map.size());

  std::vector<dealii::types::global_dof_index> mp_dofs_vec;
  std::vector<dealii::types::material_id> material_ids_vec;
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    std::vector<dealii::types::global_dof_index> mp_dof(1);
    cell->get_dof_indices(mp_dof);
    mp_dofs_vec.push_back(_dofs_map.at(mp_dof[0]));
    material_ids_vec.push_back(cell->material_id());
  }

  unsigned int const material_ids_size = material_ids_vec.size();
  Kokkos::View<dealii::types::material_id *, Kokkos::HostSpace> material_ids(
      material_ids_vec.data(), material_ids_size);
  auto material_ids_mirror = Kokkos::create_mirror_view_and_copy(
      typename MemorySpaceType::kokkos_space{}, material_ids);

  Kokkos::View<dealii::types::global_dof_index *, Kokkos::HostSpace> mp_dofs(
      mp_dofs_vec.data(), mp_dofs_vec.size());
  auto mp_dofs_mirror = Kokkos::create_mirror_view_and_copy(
      typename MemorySpaceType::kokkos_space{}, mp_dofs);

  double *temperature_average_local = temperature_average.get_values();

  using ExecutionSpace = std::conditional_t<
      std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
  auto properties = _properties;
  auto state = _state;
  auto use_table = _use_table;
  auto property_values = _property_values;
  auto state_property_tables = _state_property_tables;
  auto state_property_polynomials = _state_property_polynomials;
  Kokkos::parallel_for(
      "adamantine::update_material_properties",
      Kokkos::RangePolicy<ExecutionSpace>(0, material_ids_size),
      KOKKOS_LAMBDA(int i) {
        unsigned int constexpr solid =
            static_cast<unsigned int>(MaterialStates::State::solid);
        unsigned int constexpr prop_solidus =
            static_cast<unsigned int>(Property::solidus);
        unsigned int constexpr prop_liquidus =
            static_cast<unsigned int>(Property::liquidus);
        // Set solid_ratio to one, since that's the value when MaterialStates is
        // Solid
        double solid_ratio = 1.;
        double liquid_ratio = 0.;

        dealii::types::material_id material_id = material_ids_mirror(i);
        double const solidus = properties(material_id, prop_solidus);
        double const liquidus = properties(material_id, prop_liquidus);
        unsigned int const dof = mp_dofs_mirror(i);

        // Work-around CUDA compiler complaining that the first call to a
        // captured-variable is inside a constexpr.
        double *temp_average_local = temperature_average_local;
        auto local_state = state;

        if constexpr (!std::is_same_v<MaterialStates, Solid>)
        {
          unsigned int constexpr liquid =
              static_cast<unsigned int>(MaterialStates::State::liquid);
          // First determine the ratio of liquid.
          if (temp_average_local[dof] < solidus)
            liquid_ratio = 0.;
          else if (temp_average_local[dof] > liquidus)
            liquid_ratio = 1.;
          else
            liquid_ratio =
                (temp_average_local[dof] - solidus) / (liquidus - solidus);
          if constexpr (std::is_same_v<MaterialStates, SolidLiquid>)
          {
            solid_ratio = 1. - liquid_ratio;
          }
          else if constexpr (std::is_same_v<MaterialStates, SolidLiquidPowder>)
          {
            unsigned int constexpr powder =
                static_cast<unsigned int>(MaterialStates::State::powder);
            // Because the powder can only become liquid, the solid can only
            // become liquid, and the liquid can only become solid, the ratio of
            // powder can only decrease.
            double powder_ratio =
                Kokkos::min(1. - liquid_ratio, state(powder, dof));
            solid_ratio = 1. - liquid_ratio - powder_ratio;

            // Update _state
            state(powder, dof) = powder_ratio;
          }
          state(liquid, dof) = liquid_ratio;
        }

        // Update _state
        state(solid, dof) = solid_ratio;

        if (use_table)
        {
          for (unsigned int property = 0;
               property < g_n_thermal_state_properties; ++property)
          {
            for (unsigned int material_state = 0;
                 material_state < MaterialStates::n_material_states;
                 ++material_state)
            {
              property_values(property, dof) +=
                  state(material_state, dof) *
                  compute_property_from_table(
                      state_property_tables, material_id, property,
                      material_state, temp_average_local[dof]);
            }
          }
        }
        else
        {
          for (unsigned int property = 0;
               property < g_n_thermal_state_properties; ++property)
          {
            for (unsigned int material_state = 0;
                 material_state < MaterialStates::n_material_states;
                 ++material_state)
            {
              for (unsigned int i = 0; i <= p_order; ++i)
              {
                property_values(property, dof) +=
                    state(material_state, dof) *
                    state_property_polynomials(material_id, property,
                                               material_state, i) *
                    std::pow(temp_average_local[dof], i);
              }
            }
          }
        }

        // If we are in the mushy state, i.e., part liquid part solid, we need
        // to modify the rho C_p to take into account the latent heat.
        if constexpr (!std::is_same_v<MaterialStates, Solid>)
        {
          if ((liquid_ratio > 0.) && (liquid_ratio < 1.))
          {
            unsigned int const specific_heat_prop =
                static_cast<unsigned int>(StateProperty::specific_heat);
            unsigned int const latent_heat_prop =
                static_cast<unsigned int>(Property::latent_heat);
            for (unsigned int material_state = 0;
                 material_state < MaterialStates::n_material_states;
                 ++material_state)
            {
              property_values(specific_heat_prop, dof) +=
                  state(material_state, dof) *
                  properties(material_id, latent_heat_prop) /
                  (liquidus - solidus);
            }
          }
        }

        // The radiation heat transfer coefficient is not a real material
        // property but it is derived from other material properties: h_rad =
        // emissitivity * stefan-boltzmann constant * (T + T_infty) (T^2 +
        // T^2_infty).
        unsigned int const emissivity_prop =
            static_cast<unsigned int>(StateProperty::emissivity);
        unsigned int const radiation_heat_transfer_coef_prop =
            static_cast<unsigned int>(
                StateProperty::radiation_heat_transfer_coef);
        unsigned int const radiation_temperature_infty_prop =
            static_cast<unsigned int>(Property::radiation_temperature_infty);
        double const T = temp_average_local[dof];
        double const T_infty =
            properties(material_id, radiation_temperature_infty_prop);
        double const emissivity = property_values(emissivity_prop, dof);
        property_values(radiation_heat_transfer_coef_prop, dof) =
            emissivity * Constant::stefan_boltzmann * (T + T_infty) *
            (T * T + T_infty * T_infty);
      });
}

// TODO When we can get rid of this function, we can remove
// StateProperty::radiation_heat_transfer_coef
template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    update_boundary_material_properties(
        dealii::DoFHandler<dim> const &temperature_dof_handler,
        dealii::LA::distributed::Vector<double, MemorySpaceType> const
            &temperature)
{
  auto temperature_average =
      compute_average_temperature(temperature_dof_handler, temperature);
  // Initialize the View to zero in purpose
  _property_values =
      Kokkos::View<double **, typename MemorySpaceType::kokkos_space>(
          "property_values", g_n_thermal_state_properties, _dofs_map.size());

  std::vector<dealii::types::global_dof_index> mp_dof(1);
  // We don't need to loop over all the active cells. We only need to loop over
  // the cells at the boundary and at the interface with FE_Nothing. However, to
  // do this we need to use the temperature_dof_handler instead of the
  // _mp_dof_handler.
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    dealii::types::material_id material_id = cell->material_id();

    cell->get_dof_indices(mp_dof);
    unsigned int const dof = _dofs_map.at(mp_dof[0]);
    if (_use_table)
    {
      // We only care about properties that are used to compute the boundary
      // condition. So we start at 3.
      for (unsigned int property = 3; property < g_n_thermal_state_properties;
           ++property)
      {
        for (unsigned int material_state = 0;
             material_state < MaterialStates::n_material_states;
             ++material_state)
        {
          _property_values(property, dof) +=
              _state(material_state, dof) *
              compute_property_from_table(
                  _state_property_tables, material_id, property, material_state,
                  temperature_average.local_element(dof));
        }
      }
    }
    else
    {
      // We only care about properties that are used to compute the boundary
      // condition. So we start at 3.
      for (unsigned int property = 3; property < g_n_thermal_state_properties;
           ++property)
      {
        for (unsigned int material_state = 0;
             material_state < MaterialStates::n_material_states;
             ++material_state)
        {
          for (unsigned int i = 0; i <= p_order; ++i)
          {
            _property_values(property, dof) +=
                _state(material_state, dof) *
                _state_property_polynomials(material_id, property,
                                            material_state, i) *
                std::pow(temperature_average.local_element(dof), i);
          }
        }
      }
    }

    // The radiation heat transfer coefficient is not a real material property
    // but it is derived from other material properties:
    // h_rad = emissitivity * stefan-boltzmann constant * (T + T_infty) (T^2 +
    // T^2_infty).
    unsigned int const emissivity_prop =
        static_cast<unsigned int>(StateProperty::emissivity);
    unsigned int const radiation_heat_transfer_coef_prop =
        static_cast<unsigned int>(StateProperty::radiation_heat_transfer_coef);
    unsigned int const radiation_temperature_infty_prop =
        static_cast<unsigned int>(Property::radiation_temperature_infty);
    double const T = temperature_average.local_element(dof);
    double const T_infty =
        _properties(material_id, radiation_temperature_infty_prop);
    double const emissivity = _property_values(emissivity_prop, dof);
    _property_values(radiation_heat_transfer_coef_prop, dof) =
        emissivity * Constant::stefan_boltzmann * (T + T_infty) *
        (T * T + T_infty * T_infty);
  }
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::set_state(
    [[maybe_unused]] dealii::Table<2, dealii::VectorizedArray<double>> const
        &liquid_ratio,
    [[maybe_unused]] dealii::Table<2, dealii::VectorizedArray<double>> const
        &powder_ratio,
    [[maybe_unused]] std::map<typename dealii::DoFHandler<dim>::cell_iterator,
                              std::pair<unsigned int, unsigned int>>
        &cell_it_to_mf_cell_map,
    [[maybe_unused]] dealii::DoFHandler<dim> const &dof_handler)
{

  if constexpr (std::is_same_v<MaterialStates, Solid>)
  {
    // When there is only Solid, we know we can set all of _state to one.
    Kokkos::deep_copy(_state, 1.);
  }
  else
  {
    auto constexpr solid_state =
        static_cast<unsigned int>(MaterialStates::State::solid);
    auto constexpr liquid_state =
        static_cast<unsigned int>(MaterialStates::State::liquid);
    std::vector<dealii::types::global_dof_index> mp_dof(1.);

    if constexpr (std::is_same_v<MaterialStates, SolidLiquid>)
    {
      unsigned int const n_q_points = liquid_ratio.size(1);
      for (auto const &cell : dealii::filter_iterators(
               dof_handler.active_cell_iterators(),
               dealii::IteratorFilters::LocallyOwnedCell()))
      {
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell);
        auto mp_dof_index = get_dof_index(cell_tria);
        auto const &mf_cell_vector = cell_it_to_mf_cell_map[cell];
        double liquid_ratio_sum = 0.;
        // We should really use shape functions to compute the average. This is
        // an approximation for FE degree greater than one.
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          liquid_ratio_sum +=
              liquid_ratio(mf_cell_vector.first, q)[mf_cell_vector.second];
        }
        _state(liquid_state, mp_dof_index) = liquid_ratio_sum / n_q_points;
        _state(solid_state, mp_dof_index) =
            1. - _state(liquid_state, mp_dof_index);
      }
    }
    else if constexpr (std::is_same_v<MaterialStates, SolidLiquidPowder>)
    {
      unsigned int const n_q_points = liquid_ratio.size(1);
      auto constexpr powder_state =
          static_cast<unsigned int>(MaterialStates::State::powder);
      for (auto const &cell : dealii::filter_iterators(
               dof_handler.active_cell_iterators(),
               dealii::IteratorFilters::LocallyOwnedCell()))
      {
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell);
        auto mp_dof_index = get_dof_index(cell_tria);
        auto const &mf_cell_vector = cell_it_to_mf_cell_map[cell];
        double liquid_ratio_sum = 0.;
        double powder_ratio_sum = 0.;
        // We should really use shape functions to compute the average. This is
        // an approximation for FE degree greater than one.
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          liquid_ratio_sum +=
              liquid_ratio(mf_cell_vector.first, q)[mf_cell_vector.second];
          powder_ratio_sum +=
              powder_ratio(mf_cell_vector.first, q)[mf_cell_vector.second];
        }
        _state(liquid_state, mp_dof_index) = liquid_ratio_sum / n_q_points;
        _state(powder_state, mp_dof_index) = powder_ratio_sum / n_q_points;
        _state(solid_state, mp_dof_index) = 1. -
                                            _state(liquid_state, mp_dof_index) -
                                            _state(powder_state, mp_dof_index);
      }
    }
  }
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    set_state_device(
        Kokkos::View<double *, typename MemorySpaceType::kokkos_space>
            liquid_ratio,
        Kokkos::View<double *, typename MemorySpaceType::kokkos_space>
            powder_ratio,
        std::map<typename dealii::DoFHandler<dim>::cell_iterator,
                 std::vector<unsigned int>> const &_cell_it_to_mf_pos,
        dealii::DoFHandler<dim> const &dof_handler)
{
  // Create a mapping between the matrix free dofs and material property dofs
  unsigned int const n_q_points = dealii::Utilities::fixed_power<dim>(
      dof_handler.get_fe().tensor_degree() + 1);
  Kokkos::View<unsigned int **, dealii::MemorySpace::Default::kokkos_space>
      mapping(Kokkos::view_alloc("mapping", Kokkos::WithoutInitializing),
              _state.extent(1), n_q_points);
  auto mapping_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, mapping);
  Kokkos::View<dealii::types::global_dof_index *,
               dealii::MemorySpace::Host::kokkos_space>
      mp_dof_host("mp_dof_host", _state.extent(1));
  // We only loop over the part of the domain which has material, i.e., not over
  // FE_Nothing cell. This is because _cell_it_to_mf_pos does not exist for
  // FE_Nothing cells. However, we have set the state of the material on the
  // entire domain. This is not a problem since that state is unchanged and does
  // not need to be updated.
  unsigned int cell_i = 0;
  for (auto const &cell :
       dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::ActiveFEIndexEqualTo(
                                    0, /* locally owned */ true)))
  {
    typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(cell);
    auto mp_dof_index = get_dof_index(cell_tria);
    auto const &mf_cell_vector = _cell_it_to_mf_pos.at(cell);
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      mapping_host(cell_i, q) = mf_cell_vector[q];
    }
    mp_dof_host(cell_i) = mp_dof_index;
    ++cell_i;
  }

  Kokkos::deep_copy(mapping, mapping_host);

  Kokkos::View<dealii::types::global_dof_index *,
               dealii::MemorySpace::Default::kokkos_space>
      mp_dof(Kokkos::view_alloc("mp_dof", Kokkos::WithoutInitializing),
             mp_dof_host.extent(0));
  Kokkos::deep_copy(mp_dof, mp_dof_host);

  if constexpr (std::is_same_v<MaterialStates, Solid>)
  {
    // When there is only Solid, we can just set _state to one.
    Kokkos::deep_copy(_state, 1.);
  }
  else
  {
    auto const solid_state =
        static_cast<unsigned int>(MaterialStates::State::solid);
    auto const liquid_state =
        static_cast<unsigned int>(MaterialStates::State::liquid);

    if constexpr (std::is_same_v<MaterialStates, SolidLiquid>)
    {
      using ExecutionSpace = std::conditional_t<
          std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>,
          Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
      auto state = _state;
      Kokkos::parallel_for(
          "adamantine::set_state_device",
          Kokkos::RangePolicy<ExecutionSpace>(0, cell_i), KOKKOS_LAMBDA(int i) {
            double liquid_ratio_sum = 0.;
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              liquid_ratio_sum += liquid_ratio(mapping(i, q));
            }
            state(liquid_state, mp_dof(i)) = liquid_ratio_sum / n_q_points;
            state(solid_state, mp_dof(i)) = 1. - state(liquid_state, mp_dof(i));
          });
    }
    else if constexpr (std::is_same_v<MaterialStates, SolidLiquidPowder>)
    {
      auto const powder_state =
          static_cast<unsigned int>(MaterialStates::State::powder);
      using ExecutionSpace = std::conditional_t<
          std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>,
          Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
      auto state = _state;
      Kokkos::parallel_for(
          "adamantine::set_state_device",
          Kokkos::RangePolicy<ExecutionSpace>(0, cell_i), KOKKOS_LAMBDA(int i) {
            double liquid_ratio_sum = 0.;
            double powder_ratio_sum = 0.;
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              liquid_ratio_sum += liquid_ratio(mapping(i, q));
              powder_ratio_sum += powder_ratio(mapping(i, q));
            }
            state(liquid_state, mp_dof(i)) = liquid_ratio_sum / n_q_points;
            state(powder_state, mp_dof(i)) = powder_ratio_sum / n_q_points;
            state(solid_state, mp_dof(i)) = 1. -
                                            state(liquid_state, mp_dof(i)) -
                                            state(powder_state, mp_dof(i));
          });
    }
  }
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    set_cell_state(
        std::vector<std::array<double, MaterialStates::n_material_states>> const
            &cell_state)
{
  auto state_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _state);
  for (unsigned int i = 0; i < cell_state.size(); ++i)
  {
    for (unsigned int j = 0; j < MaterialStates::n_material_states; ++j)
    {
      state_host(j, i) = cell_state[i][j];
    }
  }
  Kokkos::deep_copy(_state, state_host);
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates,
                      MemorySpaceType>::set_initial_state()
{
  // Set the material state to the one defined by the user_index
  std::vector<dealii::types::global_dof_index> mp_dofs_vec;
  std::vector<unsigned int> user_indices_vec;
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    std::vector<dealii::types::global_dof_index> mp_dof(1);
    cell->get_dof_indices(mp_dof);
    mp_dofs_vec.push_back(_dofs_map.at(mp_dof[0]));
    user_indices_vec.push_back(cell->user_index());
  }

  typename MemorySpaceType::kokkos_space memory_space;

  Kokkos::View<dealii::types::global_dof_index *, Kokkos::HostSpace>
      mp_dofs_host(mp_dofs_vec.data(), mp_dofs_vec.size());
  auto mp_dofs =
      Kokkos::create_mirror_view_and_copy(memory_space, mp_dofs_host);

  Kokkos::View<unsigned int *, Kokkos::HostSpace> user_indices_host(
      user_indices_vec.data(), user_indices_vec.size());
  auto user_indices =
      Kokkos::create_mirror_view_and_copy(memory_space, user_indices_host);

  Kokkos::deep_copy(_state, 0.);
  using ExecutionSpace = std::conditional_t<
      std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
  auto state = _state;
  Kokkos::parallel_for(
      "adamantine::set_initial_state",
      Kokkos::RangePolicy<ExecutionSpace>(0, user_indices.extent(0)),
      KOKKOS_LAMBDA(int i) { state(user_indices(i), mp_dofs(i)) = 1.; });
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    fill_properties(boost::property_tree::ptree const &database)
{
  // PropertyTreeInput materials.property_format
  std::string property_format = database.get<std::string>("property_format");

  _use_table = (property_format == "table");
  // PropertyTreeInput materials.n_materials
  unsigned int const n_materials = database.get<unsigned int>("n_materials");
  // Find all the material_ids being used.
  std::vector<dealii::types::material_id> material_ids;
  for (dealii::types::material_id id = 0;
       id < dealii::numbers::invalid_material_id; ++id)
  {
    if (database.count("material_" + std::to_string(id)) != 0)
      material_ids.push_back(id);
    if (material_ids.size() == n_materials)
      break;
  }

  // When using the polynomial format we allocate one contiguous block of
  // memory. Thus, the largest material_id should be as small as possible
  unsigned int const n_material_ids =
      *std::max_element(material_ids.begin(), material_ids.end()) + 1;
  _properties = Kokkos::View<double *[g_n_properties],
                             typename MemorySpaceType::kokkos_space>(
      Kokkos::view_alloc("properties", Kokkos::WithoutInitializing),
      n_material_ids);
  auto properties_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _properties);

  if (_use_table)
  {
    // View is initialized to zero in purpose
    _state_property_tables =
        Kokkos::View<double *[g_n_thermal_state_properties]
                         [MaterialStates::n_material_states][table_size][2],
                     typename MemorySpaceType::kokkos_space>(
            "state_property_tables", n_material_ids);
    // Mechanical properties only exist for the solid state. View is initialized
    // to zero in purpose.
    _mechanical_properties_tables_host =
        Kokkos::View<double *[g_n_mechanical_state_properties][table_size][2],
                     typename dealii::MemorySpace::Host::kokkos_space>(
            "mechanical_properties_tables_host", n_material_ids);
  }
  else
  {
    // View is initialized to zero in purpose
    _state_property_polynomials =
        Kokkos::View<double *[g_n_thermal_state_properties]
                         [MaterialStates::n_material_states][p_order + 1],
                     typename MemorySpaceType::kokkos_space>(
            "state_property_polynomials", n_material_ids);
    // Mechanical properties only exist for the solid state. View is initialized
    // to zero in purpose
    _mechanical_properties_polynomials_host =
        Kokkos::View<double *[g_n_mechanical_state_properties][p_order + 1],
                     typename dealii::MemorySpace::Host::kokkos_space>(
            "mechanical_properties_polynomials_host", n_material_ids);
  }
  auto state_property_tables_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultHostExecutionSpace{}, _state_property_tables);
  auto state_property_polynomials_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultHostExecutionSpace{}, _state_property_polynomials);

  for (auto const material_id : material_ids)
  {
    // Get the material property tree.
    boost::property_tree::ptree const &material_database =
        database.get_child("material_" + std::to_string(material_id));
    // For each material, loop over the possible states.
    for (unsigned int state = 0; state < MaterialStates::n_material_states;
         ++state)
    {
      // The state may or may not exist for the material.
      boost::optional<boost::property_tree::ptree const &> state_database =
          material_database.get_child_optional(material_state_names[state]);
      if (state_database)
      {
        // For each state, loop over the possible properties.
        for (unsigned int p = 0; p < g_n_state_properties; ++p)
        {
          // The property may or may not exist for that state
          boost::optional<std::string> const property =
              state_database.get().get_optional<std::string>(
                  state_property_names[p]);
          // If the property exists, put it in the map. If the property does not
          // exist, we have a nullptr.
          if (property)
          {
            // Remove blank spaces
            std::string property_string = property.get();
            property_string.erase(
                std::remove_if(property_string.begin(), property_string.end(),
                               [](unsigned char x) { return std::isspace(x); }),
                property_string.end());
            if (_use_table)
            {
              std::vector<std::string> parsed_property;
              boost::split(parsed_property, property_string,
                           [](char c) { return c == '|'; });
              unsigned int const parsed_property_size = parsed_property.size();
              ASSERT_THROW(parsed_property_size <= table_size,
                           "Too many coefficients, increase the table size");
              for (unsigned int i = 0; i < parsed_property_size; ++i)
              {
                std::vector<std::string> t_v;
                boost::split(t_v, parsed_property[i],
                             [](char c) { return c == ','; });
                ASSERT(t_v.size() == 2, "Error reading material property.");
                if (p < g_n_thermal_state_properties)
                {
                  state_property_tables_host(material_id, p, state, i, 0) =
                      std::stod(t_v[0]);
                  state_property_tables_host(material_id, p, state, i, 1) =
                      std::stod(t_v[1]);
                }
                else
                {
                  if (state ==
                      static_cast<unsigned int>(MaterialStates::State::solid))
                  {
                    _mechanical_properties_tables_host(
                        material_id, p - g_n_thermal_state_properties, i, 0) =
                        std::stod(t_v[0]);
                    _mechanical_properties_tables_host(
                        material_id, p - g_n_thermal_state_properties, i, 1) =
                        std::stod(t_v[1]);
                  }
                }
              }
              // fill the rest  with the last value
              for (unsigned int i = parsed_property_size; i < table_size; ++i)
              {
                if (p < g_n_thermal_state_properties)
                {
                  state_property_tables_host(material_id, p, state, i, 0) =
                      state_property_tables_host(material_id, p, state, i - 1,
                                                 0);
                  state_property_tables_host(material_id, p, state, i, 1) =
                      state_property_tables_host(material_id, p, state, i - 1,
                                                 1);
                }
                else
                {
                  if (state ==
                      static_cast<unsigned int>(MaterialStates::State::solid))
                  {
                    _mechanical_properties_tables_host(
                        material_id, p - g_n_thermal_state_properties, i, 0) =
                        _mechanical_properties_tables_host(
                            material_id, p - g_n_thermal_state_properties,
                            i - 1, 0);
                    _mechanical_properties_tables_host(
                        material_id, p - g_n_thermal_state_properties, i, 1) =
                        _mechanical_properties_tables_host(
                            material_id, p - g_n_thermal_state_properties,
                            i - 1, 1);
                  }
                }
              }
            }
            else
            {
              std::vector<std::string> parsed_property;
              boost::split(parsed_property, property_string,
                           [](char c) { return c == ','; });
              unsigned int const parsed_property_size = parsed_property.size();
              ASSERT_THROW(
                  parsed_property_size <= p_order + 1,
                  "Too many coefficients, increase the polynomial order");
              for (unsigned int i = 0; i < parsed_property_size; ++i)
              {
                if (p < g_n_thermal_state_properties)
                {
                  state_property_polynomials_host(material_id, p, state, i) =
                      std::stod(parsed_property[i]);
                }
                else if (state == static_cast<unsigned int>(
                                      MaterialStates::State::solid))
                {
                  _mechanical_properties_polynomials_host(
                      material_id, p - g_n_thermal_state_properties, i) =
                      std::stod(parsed_property[i]);
                }
              }
            }
          }
          else if (state_property_names[p] == "elastic_limit" &&
                   state ==
                       static_cast<unsigned int>(MaterialStates::State::solid))
          {
            // If the elastic limit is not provided, we solve a purely elastic
            // problem. We set the elastic limit to infinity.
            double infinity = std::numeric_limits<double>::infinity();
            if (_use_table)
            {
              _mechanical_properties_tables_host(
                  material_id, p - g_n_thermal_state_properties, 0, 0) =
                  infinity;
              _mechanical_properties_tables_host(
                  material_id, p - g_n_thermal_state_properties, 0, 1) =
                  infinity;
            }
            else
            {
              _mechanical_properties_polynomials_host(
                  material_id, p - g_n_thermal_state_properties, 0) = infinity;
            }
          }
        }
      }
    }

    // Check for the properties that are associated to a material but that
    // are independent of an individual state. These properties are duplicated
    // for every state.
    for (unsigned int p = 0; p < g_n_properties; ++p)
    {
      // The property may or may not exist for that state
      boost::optional<double> const property =
          material_database.get_optional<double>(property_names[p]);
      // If the property exists, put it in the map. If the property does not
      // exist, we use the largest possible value. This is useful if the
      // liquidus and the solidus are not set.
      properties_host(material_id, p) =
          property ? property.get() : std::numeric_limits<double>::max();
    }
  }

  // FIXME for now we assume that the mechanical properties are independent of
  // the temperature.
  _mechanical_properties_host =
      Kokkos::View<double *[g_n_mechanical_state_properties],
                   typename dealii::MemorySpace::Host::kokkos_space>(
          "mechanical_properties_host", n_material_ids);
  if (_use_table)
  {
    // We only read the first element
    for (unsigned int i = 0; i < n_material_ids; ++i)
    {
      for (unsigned int j = 0; j < g_n_mechanical_state_properties; ++j)
      {
        _mechanical_properties_host(i, j) =
            _mechanical_properties_tables_host(i, j, 0, 1);
      }
    }
  }
  else
  {
    // We only read the first element
    for (unsigned int i = 0; i < n_material_ids; ++i)
    {
      for (unsigned int j = 0; j < g_n_mechanical_state_properties; ++j)
      {
        _mechanical_properties_host(i, j) =
            _mechanical_properties_polynomials_host(i, j, 0);
      }
    }
  }

  // Copy the data
  deep_copy(_state_property_polynomials, state_property_polynomials_host);
  deep_copy(_state_property_tables, state_property_tables_host);
  Kokkos::deep_copy(_properties, properties_host);
}

// We need to compute the average temperature on the cell because we need the
// material properties to be uniform over the cell. If there aren't then we have
// problems with the weak form discretization.
template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::LA::distributed::Vector<double, MemorySpaceType>
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    compute_average_temperature(
        dealii::DoFHandler<dim> const &temperature_dof_handler,
        dealii::LA::distributed::Vector<double, MemorySpaceType> const
            &temperature) const
{
  // TODO: this should probably done in a matrix-free fashion.
  // The triangulation is the same for both DoFHandler
  dealii::LA::distributed::Vector<double, MemorySpaceType> temperature_average(
      _mp_dof_handler.locally_owned_dofs(), temperature.get_mpi_communicator());
  temperature.update_ghost_values();
  temperature_average = 0.;
  dealii::hp::FECollection<dim> const &fe_collection =
      temperature_dof_handler.get_fe_collection();
  dealii::hp::QCollection<dim> q_collection;
  q_collection.push_back(dealii::QGauss<dim>(fe_collection.max_degree() + 1));
  q_collection.push_back(dealii::QGauss<dim>(1));
  dealii::hp::FEValues<dim> hp_fe_values(
      fe_collection, q_collection,
      dealii::UpdateFlags::update_values |
          dealii::UpdateFlags::update_quadrature_points |
          dealii::UpdateFlags::update_JxW_values);
  unsigned int const n_q_points = q_collection.max_n_quadrature_points();
  unsigned int const dofs_per_cell = fe_collection.max_dofs_per_cell();
  internal::compute_average(n_q_points, dofs_per_cell, _mp_dof_handler,
                            temperature_dof_handler, hp_fe_values, temperature,
                            temperature_average);

  return temperature_average;
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
KOKKOS_FUNCTION double
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    compute_property_from_table(
        Kokkos::View<double *[g_n_thermal_state_properties]
                         [MaterialStates::n_material_states][table_size][2],
                     typename MemorySpaceType::kokkos_space>
            state_property_tables,
        unsigned int const material_id, unsigned int const property,
        unsigned int const material_state, double const temperature)
{
  if (temperature <=
      state_property_tables(material_id, property, material_state, 0, 0))
  {
    return state_property_tables(material_id, property, material_state, 0, 1);
  }
  else
  {
    unsigned int i = 0;
    unsigned int const size = state_property_tables.extent(3);
    for (; i < size; ++i)
    {
      if (temperature <
          state_property_tables(material_id, property, material_state, i, 0))
      {
        break;
      }
    }

    if (i >= size - 1)
    {
      return state_property_tables(material_id, property, material_state,
                                   size - 1, 1);
    }
    else
    {
      auto tempertature_i =
          state_property_tables(material_id, property, material_state, i, 0);
      auto tempertature_im1 = state_property_tables(material_id, property,
                                                    material_state, i - 1, 0);
      auto property_i =
          state_property_tables(material_id, property, material_state, i, 1);
      auto property_im1 = state_property_tables(material_id, property,
                                                material_state, i - 1, 1);
      return property_im1 + (temperature - tempertature_im1) *
                                (property_i - property_im1) /
                                (tempertature_i - tempertature_im1);
    }
  }
}

} // namespace adamantine

#endif
