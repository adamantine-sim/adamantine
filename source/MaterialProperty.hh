/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MATERIAL_PROPERTY_HH
#define MATERIAL_PROPERTY_HH

#include <ArrayMD.hh>
#include <types.hh>
#include <utils.hh>

#include <deal.II/base/memory_space.h>
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
template <int dim>
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
   * Reinitialize the DoFHandler associated with MaterialProperty and resize the
   * state vectors.
   */
  void reinit_dofs();

  /**
   * Update the material state, i.e, the ratio of liquid, powder, and solid and
   * the material properties given the field of temperature.
   */
  void update(dealii::DoFHandler<dim> const &temperature_dof_handler,
              dealii::LA::distributed::Vector<double> const &temperature);

  /**
   * Update the material properties necessary to compute the radiative and
   * convective boundary conditions given the field of temperature.
   */
  void update_boundary_material_properties(
      dealii::DoFHandler<dim> const &temperature_dof_handler,
      dealii::LA::distributed::Vector<double> const &temperature);

  /**
   * Compute a material property at a quadrature point for a mix of states.
   */
  ADAMANTINE_HOST_DEV
  dealii::VectorizedArray<double>
  compute_material_property(StateProperty state_property,
                            dealii::types::material_id const *material_id,
                            dealii::VectorizedArray<double> const *state_ratios,
                            dealii::VectorizedArray<double> temperature) const;

  /**
   * Get the array of material state vectors. The order of the different state
   * vectors is given by the MaterialState enum. Each entry in the vector
   * correspond to a cell in the mesh and has a value between 0 and 1. The sum
   * of the states for a given cell is equal to 1.
   */
  std::array<dealii::LA::distributed::Vector<double>,
             static_cast<unsigned int>(MaterialState::SIZE)> &
  get_state();

  /**
   * Get the ratio of a given MaterialState for a given cell. The sum
   * of the states for a given cell is equal to 1.
   */
  double get_state_ratio(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      MaterialState material_state) const;

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
   * Return the underlying the DoFHandler.
   */
  dealii::DoFHandler<dim> const &get_dof_handler() const;

private:
  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr _n_material_states =
      static_cast<unsigned int>(MaterialState::SIZE);

  /**
   * Number of Stateproperty defined.
   */
  static unsigned int constexpr _n_state_properties =
      static_cast<unsigned int>(StateProperty::SIZE);

  /**
   * Number of Stateproperty defined.
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
   * Set the values in _state from the values of the user index of the
   * Triangulation.
   */
  void set_initial_state();

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
  dealii::LA::distributed::Vector<double> compute_average_temperature(
      dealii::DoFHandler<dim> const &temperature_dof_handler,
      dealii::LA::distributed::Vector<double> const &temperature) const;

  /**
   * Compute a property from a table given the temperature.
   */
  double compute_property_from_table(unsigned int const material_id,
                                     unsigned int const material_state,
                                     unsigned int const property,
                                     double const temperature) const;

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
   * Array that stores the material properties which have been set using tables.
   */
  Array5D<-1, _n_material_states, _n_state_properties, -1, 2,
          dealii::MemorySpace::Host>
      _state_property_tables;
  /**
   * Array that stores the material properties which have been set using
   * polynomials.
   */
  Array4D<-1, _n_material_states, _n_state_properties, -1,
          dealii::MemorySpace::Host>
      _state_property_polynomials;
  /**
   * Array that stores the properties of the material that are independent of
   * the state of the material.
   */
  Array2D<-1, _n_properties, dealii::MemorySpace::Host> _properties;
  /**
   * Array of vector describing the ratio of each state in each cell. Each
   * vector corresponds to a state defined in the MaterialState enum.
   */
  std::array<dealii::LA::distributed::Vector<double>, _n_material_states>
      _state;
  /**
   * Array of vector describing the property in each cell. Each vector
   * corresponds to a property in the StateProperty enum.
   */
  std::array<dealii::LA::distributed::Vector<double>, _n_state_properties>
      _property_values;
  /**
   * Discontinuous piecewise constant finite element.
   */
  dealii::FE_DGQ<dim> _fe;
  /**
   * DoFHandler associated to the _state array.
   */
  dealii::DoFHandler<dim> _mp_dof_handler;
};

template <int dim>
inline double MaterialProperty<dim>::get(dealii::types::material_id material_id,
                                         Property property) const
{
  return _properties(material_id, static_cast<unsigned int>(property));
}

template <int dim>
inline std::array<dealii::LA::distributed::Vector<double>,
                  static_cast<unsigned int>(MaterialState::SIZE)> &
MaterialProperty<dim>::get_state()
{
  return _state;
}

template <int dim>
inline double MaterialProperty<dim>::get_state_ratio(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    MaterialState material_state) const
{
  auto const mp_dof_index = get_dof_index(cell);
  auto const mat_state = static_cast<unsigned int>(material_state);

  return _state[mat_state][mp_dof_index];
}

template <int dim>
inline dealii::types::global_dof_index MaterialProperty<dim>::get_dof_index(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell) const
{
  // Get a DoFCellAccessor from a Triangulation::active_cell_iterator.
  dealii::DoFAccessor<dim, dim, dim, false> dof_accessor(
      &_mp_dof_handler.get_triangulation(), cell->level(), cell->index(),
      &_mp_dof_handler);
  std::vector<dealii::types::global_dof_index> mp_dof(1.);
  dof_accessor.get_dof_indices(mp_dof);

  return mp_dof[0];
}

template <int dim>
inline dealii::DoFHandler<dim> const &
MaterialProperty<dim>::get_dof_handler() const
{
  return _mp_dof_handler;
}
} // namespace adamantine

#endif
