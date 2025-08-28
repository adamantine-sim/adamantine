/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MATERIAL_PROPERTY_HH
#define MATERIAL_PROPERTY_HH

#include <MaterialStates.hh>
#include <types.hh>
#include <utils.hh>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/property_tree/ptree.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <limits>
#include <unordered_map>

namespace adamantine
{
/**
 * This class stores the material properties for all the materials
 */
template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
class MaterialProperty
{
public:
  /**
   * Size of the table, i.e. number of temperature/property pairs, used to
   * describe the material properties.
   */
  static unsigned int constexpr table_size = 12;

  /**
   * Constructor.
   */
  MaterialProperty(
      MPI_Comm const &communicator,
      dealii::parallel::distributed::Triangulation<dim> const &tria,
      boost::property_tree::ptree const &database);

  /**
   * Return true if the material properties are given in table format.
   * Return false if they are given in polynomial format.
   */
  bool properties_use_table() const;

  /**
   * Return the value of the given StateProperty for a given cell.
   */
  double get_cell_value(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      StateProperty prop) const;

  /**
   * Return the value of the given Property for a given cell.
   */
  double get_cell_value(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      Property prop) const;

  /**
   * Return the value of a given Property for a given material id.
   */
  double get(dealii::types::material_id material_id, Property prop) const;

  /**
   * Return the values of the given mechanical StateProperty for a given cell.
   */
  double get_mechanical_property(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      StateProperty prop) const;

  /**
   * Return the properties of the material that are independent of the state of
   * the material.
   */
  Kokkos::View<double *[g_n_properties], typename MemorySpaceType::kokkos_space>
  get_properties();

  /**
   * Return the properties of the material that are dependent of the state of
   * the material and which have been set using tables.
   */
  Kokkos::View<double * [MaterialStates::n_material_states]
                            [g_n_thermal_state_properties][table_size][2],
               typename MemorySpaceType::kokkos_space>
  get_state_property_tables();

  /**
   * Return the properties of the material that are dependent of the state of
   * the material and which have beese set using polynomials.
   */
  Kokkos::View<double * [MaterialStates::n_material_states]
                            [g_n_thermal_state_properties][p_order + 1],
               typename MemorySpaceType::kokkos_space>
  get_state_property_polynomials();

  /**
   * Reinitialize the DoFHandler associated with MaterialProperty and resize the
   * state vectors.
   */
  void reinit_dofs();

  /**
   * Update the material state, i.e, the ratio of liquid, powder, and solid and
   * the material properties given the field of temperature.
   */
  void update(dealii::DoFHandler<dim> const &temperature_dof_handler,
              dealii::LA::distributed::Vector<double, MemorySpaceType> const
                  &temperature);

  /**
   * Update the material properties necessary to compute the radiative and
   * convective boundary conditions given the field of temperature.
   */
  void update_boundary_material_properties(
      dealii::DoFHandler<dim> const &temperature_dof_handler,
      dealii::LA::distributed::Vector<double, MemorySpaceType> const
          &temperature);

  /**
   * Compute a material property at a quadrature point for a mix of states.
   * @note This function is templated on @tparam because it is in a hot loop.
   */
  template <bool use_table>
  dealii::VectorizedArray<double> compute_material_property(
      StateProperty state_property,
      std::array<dealii::types::material_id,
                 dealii::VectorizedArray<double>::size()> const &material_id,
      std::array<dealii::VectorizedArray<double>,
                 MaterialStates::n_material_states> const &state_ratios,
      dealii::VectorizedArray<double> const &temperature,
      dealii::AlignedVector<dealii::VectorizedArray<double>> const
          &temperature_powers) const;

  /**
   * Compute a material property at a quadrature point for a mix of states.
   * @note This function is templated on @tparam because it is in a hot loop.
   */
  template <bool use_table>
  KOKKOS_FUNCTION double
  compute_material_property(StateProperty state_property,
                            dealii::types::material_id const material_id,
                            double const *state_ratios,
                            double temperature) const;

  /**
   * Get the array of material state vectors. The order of the different state
   * vectors is given by the MaterialState enum. Each entry in the vector
   * correspond to a cell in the mesh and has a value between 0 and 1. The sum
   * of the states for a given cell is equal to 1.
   */
  Kokkos::View<double **, typename MemorySpaceType::kokkos_space>
  get_state() const;

  /**
   * Get the ratio of a given MaterialState for a given cell. The sum
   * of the states for a given cell is equal to 1.
   */
  double get_state_ratio(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      typename MaterialStates::State material_state) const;

  /**
   * Set the values in _state from the values of the user index of the
   * Triangulation.
   */
  // This cannot be private due to limitation of lambda function with CUDA
  void set_initial_state();

  /**
   * Set the ratio of the material states from ThermalOperator.
   */
  void set_state(
      dealii::Table<2, dealii::VectorizedArray<double>> const &liquid_ratio,
      dealii::Table<2, dealii::VectorizedArray<double>> const &powder_ratio,
      std::map<typename dealii::DoFHandler<dim>::cell_iterator,
               std::pair<unsigned int, unsigned int>> &cell_it_to_mf_cell_map,
      dealii::DoFHandler<dim> const &dof_handler);

  /**
   * Set the ratio of the material states from ThermalOperatorDevice.
   */
  void set_state_device(
      Kokkos::View<double *, typename MemorySpaceType::kokkos_space>
          liquid_ratio,
      Kokkos::View<double *, typename MemorySpaceType::kokkos_space>
          powder_ratio,
      std::map<typename dealii::DoFHandler<dim>::cell_iterator,
               std::vector<unsigned int>> const &_cell_it_to_mf_pos,
      dealii::DoFHandler<dim> const &dof_handler);

  /**
   * Set the ratio of the material states at the cell level.
   */
  void set_cell_state(
      std::vector<std::array<double, MaterialStates::n_material_states>> const
          &cell_state);

  /**
   * Return the underlying the DoFHandler.
   */
  dealii::DoFHandler<dim> const &get_dof_handler() const;

  /**
   * Return the mapping between the degrees of freedom and the local index of
   * the cells.
   */
  std::unordered_map<dealii::types::global_dof_index, unsigned int>
  get_dofs_map() const
  {
    return _dofs_map;
  }

  /**
   * Compute a property from a table given the temperature.
   */
  static KOKKOS_FUNCTION double compute_property_from_table(
      Kokkos::View<double ****[2], typename MemorySpaceType::kokkos_space>
          state_property_tables,
      unsigned int const material_id, unsigned int const material_state,
      unsigned int const property, double const temperature);

private:
  /**
   * Fill the _properties map.
   */
  void fill_properties(boost::property_tree::ptree const &database);

  /**
   * Return the index of the dof associated to the cell.
   */
  dealii::types::global_dof_index get_dof_index(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
      const;

  /**
   * Compute the average of the temperature on every cell.
   */
  dealii::LA::distributed::Vector<double, MemorySpaceType>
  compute_average_temperature(
      dealii::DoFHandler<dim> const &temperature_dof_handler,
      dealii::LA::distributed::Vector<double, MemorySpaceType> const
          &temperature) const;

  /**
   * MPI communicator.
   */
  MPI_Comm _communicator;
  /**
   * If the flag is true the material properties are saved under a table.
   * Otherwise the material properties are saved as polynomials.
   */
  bool _use_table;
  /**
   * Thermal material properties which have been set using tables.
   */
  Kokkos::View<double * [MaterialStates::n_material_states]
                            [g_n_thermal_state_properties][table_size][2],
               typename MemorySpaceType::kokkos_space>
      _state_property_tables;
  /**
   * Thermal material properties which have been set
   * using polynomials.
   */
  Kokkos::View<double * [MaterialStates::n_material_states]
                            [g_n_thermal_state_properties][p_order + 1],
               typename MemorySpaceType::kokkos_space>
      _state_property_polynomials;
  /**
   * Properties of the material that are independent of the state of the
   * material.
   */
  Kokkos::View<double *[g_n_properties], typename MemorySpaceType::kokkos_space>
      _properties;
  /**
   * Ratio of each in MaterarialState in each cell.
   */
  // FIXME Change the order of the indices. Currently, the first index is the
  // state and the second is the cell.
  Kokkos::View<double **, typename MemorySpaceType::kokkos_space> _state;
  /**
   * Thermal properties of the material that are dependent of the state of the
   * material.
   */
  Kokkos::View<double **, typename MemorySpaceType::kokkos_space>
      _property_values;
  /**
   * Mechanical properties which have been set using tables.
   */
  // We cannot put the mechanical properties with the thermal properties because
  // the mechanical properties can only exist on the host while the thermal ones
  // can be on the host or the device.
  Kokkos::View<double *[g_n_mechanical_state_properties][table_size][2],
               dealii::MemorySpace::Host::kokkos_space>
      _mechanical_properties_tables_host;
  /**
   * Mechanical properties which have been set using polynomials.
   */
  Kokkos::View<double *[g_n_mechanical_state_properties][p_order + 1],
               dealii::MemorySpace::Host::kokkos_space>
      _mechanical_properties_polynomials_host;
  /**
   * Temperature independent mechanical properties.
   */
  Kokkos::View<double *[g_n_mechanical_state_properties],
               dealii::MemorySpace::Host::kokkos_space>
      _mechanical_properties_host;
  /**
   * Discontinuous piecewise constant finite element.
   */
  dealii::FE_DGQ<dim> _fe;
  /**
   * DoFHandler associated to the _state array.
   */
  dealii::DoFHandler<dim> _mp_dof_handler;
  /**
   * Mapping between the degrees of freedom and the local index of the cells.
   */
  std::unordered_map<dealii::types::global_dof_index, unsigned int> _dofs_map;
};

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline double
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::get(
    dealii::types::material_id material_id, Property property) const
{
  return _properties(material_id, static_cast<unsigned int>(property));
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline Kokkos::View<double *[g_n_properties],
                    typename MemorySpaceType::kokkos_space>
MaterialProperty<dim, p_order, MaterialStates,
                 MemorySpaceType>::get_properties() { return _properties; }

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline bool MaterialProperty<dim, p_order, MaterialStates,
                             MemorySpaceType>::properties_use_table() const
{
  return _use_table;
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline Kokkos::View<double *[MaterialStates::n_material_states]
                                [g_n_thermal_state_properties][MaterialProperty<
                                    dim, p_order, MaterialStates,
                                    MemorySpaceType>::table_size][2],
                    typename MemorySpaceType::kokkos_space>
MaterialProperty<dim, p_order, MaterialStates,
                 MemorySpaceType>::get_state_property_tables()
{ return _state_property_tables; }

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline Kokkos::View<
    double *[MaterialStates::n_material_states][g_n_thermal_state_properties]
                                               [p_order + 1],
    typename MemorySpaceType::kokkos_space> MaterialProperty<dim, p_order,
                                                             MaterialStates,
                                                             MemorySpaceType>::
    get_state_property_polynomials() { return _state_property_polynomials; }

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline Kokkos::
    View<double **, typename MemorySpaceType::kokkos_space> MaterialProperty<
        dim, p_order, MaterialStates, MemorySpaceType>::get_state() const
{
  return _state;
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline dealii::types::global_dof_index
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::get_dof_index(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell) const
{
  // Get a DoFCellAccessor from a Triangulation::active_cell_iterator.
  dealii::DoFAccessor<dim, dim, dim, false> dof_accessor(
      &_mp_dof_handler.get_triangulation(), cell->level(), cell->index(),
      &_mp_dof_handler);
  std::vector<dealii::types::global_dof_index> mp_dof(1.);
  dof_accessor.get_dof_indices(mp_dof);

  return _dofs_map.at(mp_dof[0]);
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
inline dealii::DoFHandler<dim> const &
MaterialProperty<dim, p_order, MaterialStates,
                 MemorySpaceType>::get_dof_handler() const
{
  return _mp_dof_handler;
}

// We define the two compute_material_property in the header to simplify, the
// instantiation. It also helps the compiler to inline the code.

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
template <bool use_table>
dealii::VectorizedArray<double>
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    compute_material_property(
        StateProperty state_property,
        std::array<dealii::types::material_id,
                   dealii::VectorizedArray<double>::size()> const &material_id,
        std::array<dealii::VectorizedArray<double>,
                   MaterialStates::n_material_states> const &state_ratios,
        dealii::VectorizedArray<double> const &temperature,
        dealii::AlignedVector<dealii::VectorizedArray<double>> const
            &temperature_powers) const
{
  dealii::VectorizedArray<double> value = 0.0;
  dealii::VectorizedArray<double> property;
  unsigned int const property_index = static_cast<unsigned int>(state_property);

  if constexpr (use_table)
  {
    for (unsigned int material_state = 0;
         material_state < MaterialStates::n_material_states; ++material_state)
    {
      for (unsigned int n = 0; n < dealii::VectorizedArray<double>::size(); ++n)
      {
        property[n] = compute_property_from_table(
            _state_property_tables, material_id[n], material_state,
            property_index, temperature[n]);
      }

      value += state_ratios[material_state] * property;
    }
  }
  else
  {
    for (unsigned int material_state = 0;
         material_state < MaterialStates::n_material_states; ++material_state)
    {
      for (unsigned int i = 0; i <= p_order; ++i)
      {
        for (unsigned int n = 0; n < dealii::VectorizedArray<double>::size();
             ++n)
        {
          property[n] = _state_property_polynomials(
              material_id[n], material_state, property_index, i);
        }

        value +=
            state_ratios[material_state] * property * temperature_powers[i];
      }
    }
  }

  return value;
}

template <int dim, int p_order, typename MaterialStates,
          typename MemorySpaceType>
template <bool use_table>
KOKKOS_FUNCTION double
MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>::
    compute_material_property(StateProperty state_property,
                              dealii::types::material_id const material_id,
                              double const *state_ratios,
                              double temperature) const
{
  double value = 0.0;
  unsigned int const property_index = static_cast<unsigned int>(state_property);

  if constexpr (use_table)
  {
    for (unsigned int material_state = 0;
         material_state < MaterialStates::n_material_states; ++material_state)
    {
      const dealii::types::material_id m_id = material_id;

      value += state_ratios[material_state] *
               compute_property_from_table(_state_property_tables, m_id,
                                           material_state, property_index,
                                           temperature);
    }
  }
  else
  {
    for (unsigned int material_state = 0;
         material_state < MaterialStates::n_material_states; ++material_state)
    {
      dealii::types::material_id m_id = material_id;

      for (unsigned int i = 0; i <= p_order; ++i)
      {
        value += state_ratios[material_state] *
                 _state_property_polynomials(m_id, material_state,
                                             property_index, i) *
                 std::pow(temperature, i);
      }
    }
  }

  return value;
}
} // namespace adamantine

#endif
