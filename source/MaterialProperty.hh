/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MATERIAL_PROPERTY_HH
#define MATERIAL_PROPERTY_HH

#include <types.hh>
#include <utils.hh>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/cuda_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

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
   * \param[in] database requires the following entries:
   *   - <B>n_materials</B>: unsigned int in \f$(0,\infty)\f$
   *   - <B>material_X</B>: property tree associated with material_X
   *   where X is a number
   *   - <B>material_X.Y</B>: property_tree where Y is either liquid, powder, or
   *   solid [optional]
   *   - <B>material_X.Y.Z</B>: string where Z is either density, specific_heat,
   *   or thermal_conductivity, describe the behavior of the property as a
   *   function of the temperature (e.g. "2.*T") [optional]
   *   - <B>material.X.A</B>: A is either solidus, liquidus, or latent heat
   * [optional]
   */
  MaterialProperty(
      MPI_Comm const &communicator,
      dealii::parallel::distributed::Triangulation<dim> const &tria,
      boost::property_tree::ptree const &database);

  /**
   * Return the value of the given StateProperty for a given cell.
   */
  double
  get(typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      StateProperty prop) const;

  // TODO add a function to get tensor material properties
  /**
   * Return the value of the given Property for a given cell.
   */
  double
  get(typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      Property prop) const;

  /**
   * Return the material id for a given cell.
   */
  dealii::types::material_id get_material_id(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
      const;

  /**
   * Reinitialize the DoFHandler associated with MaterialProperty and resize
   * the state vectors.
   */
  void reinit_dofs();

  /**
   * Update the material state, i.e, the ratio of liquid, powder, and solid and
   * the material properties given the field of temperature.
   */
  void update(dealii::DoFHandler<dim> const &temperature_dof_handler,
              dealii::LA::distributed::Vector<double> const &temperature);

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
   * Return the underlying the DoFHandler.
   */
  dealii::DoFHandler<dim> const &get_dof_handler() const;

  // New public members of the reworked MaterialProperies
  /**
   * Update the state ratios at a quadrature point
   */
#ifdef ADAMANTINE_HAVE_CUDA
  void update_state_ratios(
      unsigned int pos, double temperature,
      std::array<double, static_cast<unsigned int>(MaterialState::SIZE)>
          &state_ratios);
#else
  void update_state_ratios(
      unsigned int cell, unsigned int q,
      dealii::VectorizedArray<double> temperature,
      std::array<dealii::VectorizedArray<double>,
                 static_cast<unsigned int>(MaterialState::SIZE)> &state_ratios);
#endif
  /**
   * Calculate inv_rho_cp at a quadrature point
   */
#ifdef ADAMANTINE_HAVE_CUDA
  double MaterialProperty<dim>::get_inv_rho_cp(
      unsigned int pos,
      std::array<double, static_cast<unsigned int>(MaterialState::SIZE)>
          state_ratios,
      double temperature);
#else
  dealii::VectorizedArray<double>
  get_inv_rho_cp(unsigned int cell, unsigned int q,
                 std::array<dealii::VectorizedArray<double>,
                            static_cast<unsigned int>(MaterialState::SIZE)>
                     state_ratios,
                 dealii::VectorizedArray<double> temperature);
#endif
  /**
   * Calculate the thermal conductivity at a quadrature point
   */
#ifdef ADAMANTINE_HAVE_CUDA
  double get_thermal_conductivity(
      unsigned int pos,
      std::array<double, static_cast<unsigned int>(MaterialState::SIZE)>
          state_ratios,
      double temperature);
#else
  dealii::VectorizedArray<double> get_thermal_conductivity(
      unsigned int cell, unsigned int q,
      std::array<dealii::VectorizedArray<double>,
                 static_cast<unsigned int>(MaterialState::SIZE)>
          state_ratios,
      dealii::VectorizedArray<double> temperature);
#endif

#ifdef ADAMANTINE_HAVE_CUDA
  dealii::LinearAlgebra::CUDAWrappers::Vector<double> _powder_ratio;

  dealii::LinearAlgebra::CUDAWrappers::Vector<dealii::types::material_id>
      _material_id;
#else
  /**
   * Table of the powder fraction, public to minimize the reimplementation of
   * methods for GPUs.
   */
  dealii::Table<2, dealii::VectorizedArray<double>> _powder_ratio;
  /**
   * Table of the material index, public to minimize the reimplementation of
   * methods for GPUs.
   */
  dealii::Table<2, std::array<dealii::types::material_id,
                              dealii::VectorizedArray<double>::size()>>
      _material_id;
#endif

private:
  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr _n_material_states =
      static_cast<unsigned int>(MaterialState::SIZE);

  /**
   * Number of state properties defined.
   */
  static unsigned int constexpr _n_state_properties =
      static_cast<unsigned int>(StateProperty::SIZE);

  /**
   * Number of properties defined.
   */
  static unsigned int constexpr _n_properties =
      static_cast<unsigned int>(Property::SIZE);

  /**
   * Set the values in _state from the values of the user index of the
   * Triangulation.
   */
  void set_state();

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
  double compute_property_from_table(
      std::vector<std::pair<double, double>> const &table,
      double const temperature) const;

  /**
   * Compute a property using a polynomial representation given the
   * temperature.
   */
  double compute_property_from_polynomial(std::vector<double> const &coef,
                                          double const temperature) const;
  /**
   * Compute a material property a quadrature point for a mix of states
   */
  dealii::VectorizedArray<double> compute_material_property(
      StateProperty state_property,
      std::array<dealii::types::material_id,
                 dealii::VectorizedArray<double>::size()>
          material_id,
      std::array<dealii::VectorizedArray<double>,
                 static_cast<unsigned int>(MaterialState::SIZE)>
          state_ratios,
      dealii::VectorizedArray<double> temperature) const;

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
   * Map that stores tables describing the properties of the material.
   */
  std::unordered_map<
      dealii::types::material_id,
      std::array<std::array<std::vector<std::pair<double, double>>,
                            _n_state_properties>,
                 _n_material_states>>
      _state_property_tables;
  /**
   * Map that stores polynomials describing the properties of the material.
   */
  std::unordered_map<
      dealii::types::material_id,
      std::array<std::array<std::vector<double>, _n_state_properties>,
                 _n_material_states>>
      _state_property_polynomials;
  /**
   * Map that stores the properties of the material that are independent of the
   * state of the material.
   */
  std::unordered_map<dealii::types::material_id,
                     std::array<double, _n_properties>>
      _properties;
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
