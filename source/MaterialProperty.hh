/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MATERIAL_PROPERTY_HH
#define MATERIAL_PROPERTY_HH

#include <MemoryBlock.hh>
#include <MemoryBlockView.hh>
#include <types.hh>
#include <utils.hh>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>

#include <boost/property_tree/ptree.hpp>

#include <array>
#include <limits>
#include <unordered_map>

namespace adamantine
{
/**
 * This class stores the material properties for all the materials
 */
template <int dim, typename MemorySpaceType>
class MaterialProperty
{
public:
  /**
   * Constructor.
   */
  MaterialProperty(
      MPI_Comm const &communicator,
      dealii::parallel::distributed::Triangulation<dim> const &tria,
      boost::property_tree::ptree const &database);

  /**
   * Return true if the material properties are given in table format. Return
   * false if they are given in polynomial format.
   */
  bool properties_use_table() const;

  /**
   * Return the order of the polynomial used to set the material properties
   */
  unsigned int polynomial_order() const;

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
   * Return a MemoryBlockView of the properties of the material that are
   * independent of the state of the material.
   */
  MemoryBlockView<double, MemorySpaceType> get_properties();

  /**
   * Return a MemoryBlockView of the properties of the material that are
   * dependent of the state of the material and which have been set using
   * tables.
   */
  MemoryBlockView<double, MemorySpaceType> get_state_property_tables();

  /**
   * Return a MemoryBlockView of the properties of the material that are
   * dependent of the state of the material and which have beese set using
   * polynomials.
   */
  MemoryBlockView<double, MemorySpaceType> get_state_property_polynomials();

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
   */
  dealii::VectorizedArray<double>
  compute_material_property(StateProperty state_property,
                            dealii::types::material_id const *material_id,
                            dealii::VectorizedArray<double> const *state_ratios,
                            dealii::VectorizedArray<double> temperature) const;

  /**
   * Compute a material property at a quadrature point for a mix of states.
   */
  ADAMANTINE_HOST_DEV
  double compute_material_property(StateProperty state_property,
                                   dealii::types::material_id const material_id,
                                   double const *state_ratios,
                                   double temperature) const;

  /**
   * Get the array of material state vectors. The order of the different state
   * vectors is given by the MaterialState enum. Each entry in the vector
   * correspond to a cell in the mesh and has a value between 0 and 1. The sum
   * of the states for a given cell is equal to 1.
   */
  MemoryBlockView<double, MemorySpaceType> get_state() const;

  /**
   * Get the ratio of a given MaterialState for a given cell. The sum
   * of the states for a given cell is equal to 1.
   */
  double get_state_ratio(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      MaterialState material_state) const;

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
      MemoryBlock<double, MemorySpaceType> const &liquid_ratio,
      MemoryBlock<double, MemorySpaceType> const &powder_ratio,
      std::map<typename dealii::DoFHandler<dim>::cell_iterator,
               std::vector<unsigned int>> const &_cell_it_to_mf_pos,
      dealii::DoFHandler<dim> const &dof_handler);

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
  static ADAMANTINE_HOST_DEV double compute_property_from_table(
      MemoryBlockView<double, MemorySpaceType> const
          &state_property_tables_view,
      unsigned int const material_id, unsigned int const material_state,
      unsigned int const property, double const temperature);

private:
  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr _n_material_states =
      static_cast<unsigned int>(MaterialState::SIZE);

  /**
   * Number of StateProperty defined.
   */
  static unsigned int constexpr _n_state_properties =
      static_cast<unsigned int>(StateProperty::SIZE);

  /**
   * Number of mechanical StateProperty.
   */
  static unsigned int constexpr _n_mechanical_state_properties =
      static_cast<unsigned int>(StateProperty::SIZE_MECHANICAL);

  /**
   * Number of thermal StateProperty..
   */
  static unsigned int constexpr _n_thermal_state_properties =
      _n_state_properties - _n_mechanical_state_properties;

  /**
   * Number of Property defined.
   */
  static unsigned int constexpr _n_properties =
      static_cast<unsigned int>(Property::SIZE);

  /**
   * Order of the polynomial used to describe the material properties.
   */
  unsigned int _polynomial_order = 0;

  /**
   * Size of the table, i.e. number of temperature/property pairs, used to
   * describe the material properties.
   */
  unsigned int _table_size = 0;

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
   * MemoryBlock that stores the thermal material properties which have been set
   * using tables.
   */
  MemoryBlock<double, MemorySpaceType> _state_property_tables;
  /**
   * MemoryBlock that stores the thermal material properties which have been set
   * using polynomials.
   */
  MemoryBlock<double, MemorySpaceType> _state_property_polynomials;
  /**
   * MemoryBlock that stores the properties of the material that are independent
   * of the state of the material.
   */
  MemoryBlock<double, MemorySpaceType> _properties;
  /**
   * MemoryBlockView associated with _properties.
   */
  MemoryBlockView<double, MemorySpaceType> _properties_view;
  /**
   * MemoryBlock that stores the ratio of each in MaterarialState in each cell.
   */
  MemoryBlock<double, MemorySpaceType> _state;
  /**
   * MemoryBlock that stores the thermal properties of the material that are
   * dependent of the state of the material.
   */
  MemoryBlock<double, MemorySpaceType> _property_values;
  /**
   * MemoryBlock that stores the mechanical properties which have been set using
   * tables.
   */
  // We cannot put the mechanical properties with the thermal properties because
  // the mechanical properties can only exist on the host while the thermal ones
  // can be on the host or the device.
  MemoryBlock<double, dealii::MemorySpace::Host>
      _mechanical_properties_tables_host;
  /**
   * MemoryBlock that stores the mechanical properties which have been set using
   * polynomials.
   */
  MemoryBlock<double, dealii::MemorySpace::Host>
      _mechanical_properties_polynomials_host;
  /**
   * MemoryBlock that stores the mechanical properties.
   */
  MemoryBlock<double, dealii::MemorySpace::Host> _mechanical_properties_host;
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

template <int dim, typename MemorySpaceType>
inline double MaterialProperty<dim, MemorySpaceType>::get(
    dealii::types::material_id material_id, Property property) const
{
  // This function works only on the host because the MemoryBlockView needs to
  // be created on the host. Using MemoryBlock directly doesn't work because the
  // _data ptr is on the host.
  return _properties_view(material_id, static_cast<unsigned int>(property));
}

template <int dim, typename MemorySpaceType>
inline MemoryBlockView<double, MemorySpaceType>
MaterialProperty<dim, MemorySpaceType>::get_properties()
{
  return _properties_view;
}

template <int dim, typename MemorySpaceType>
inline bool MaterialProperty<dim, MemorySpaceType>::properties_use_table() const
{
  return _use_table;
}

template <int dim, typename MemorySpaceType>
inline unsigned int
MaterialProperty<dim, MemorySpaceType>::polynomial_order() const
{
  return _polynomial_order;
}

template <int dim, typename MemorySpaceType>
inline MemoryBlockView<double, MemorySpaceType>
MaterialProperty<dim, MemorySpaceType>::get_state_property_tables()
{
  return MemoryBlockView<double, MemorySpaceType>(_state_property_tables);
}

template <int dim, typename MemorySpaceType>
inline MemoryBlockView<double, MemorySpaceType>
MaterialProperty<dim, MemorySpaceType>::get_state_property_polynomials()
{
  return MemoryBlockView<double, MemorySpaceType>(_state_property_polynomials);
}

template <int dim, typename MemorySpaceType>
inline MemoryBlockView<double, MemorySpaceType>
MaterialProperty<dim, MemorySpaceType>::get_state() const
{
  return MemoryBlockView<double, MemorySpaceType>(_state);
}

template <int dim, typename MemorySpaceType>
inline dealii::types::global_dof_index
MaterialProperty<dim, MemorySpaceType>::get_dof_index(
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

template <int dim, typename MemorySpaceType>
inline dealii::DoFHandler<dim> const &
MaterialProperty<dim, MemorySpaceType>::get_dof_handler() const
{
  return _mp_dof_handler;
}
} // namespace adamantine

#endif
