/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _MATERIAL_PROPERTY_HH_
#define _MATERIAL_PROPERTY_HH_

#include "types.hh"
#include "utils.hh"
#include <deal.II/base/function_parser.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <boost/mpi.hpp>
#include <boost/property_tree/ptree.hpp>
#include <array>
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
   *   function of the temperatur (e.g. "2.*T") [optional]
   */
  MaterialProperty(
      boost::mpi::communicator const &communicator,
      dealii::parallel::distributed::Triangulation<dim> const &tria,
      boost::property_tree::ptree const &database);

  /**
   * Return the value of the given property, for a given cell and a given field
   * state.
   */
  template <typename NumberType>
  double
  get(typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      Property prop,
      dealii::LA::distributed::Vector<NumberType> const &field_state) const;

  /**
   * Reinitialize the DoFHandler associated with MaterialProperty and resize the
   * state vectors.
   */
  void reinit_dofs();

  /**
   * Get the array of material state vectors. The order of the different state
   * vectos is given by the MaterialState enum. Each entry in the vector
   * correspond to a cell in the mesh and has a value between 0 and 1. The sum
   * of the states for a given cell is equal to 1.
   */
  std::array<dealii::LA::distributed::Vector<double>,
             static_cast<unsigned int>(MaterialState::SIZE)> &
  get_state();

private:
  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr _n_material_states =
      static_cast<unsigned int>(MaterialState::SIZE);

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
   * MPI communicator.
   */
  boost::mpi::communicator _communicator;
  /**
   * Map that stores functions describing the properties of the material.
   */
  std::unordered_map<
      dealii::types::material_id,
      std::array<
          std::array<std::unique_ptr<dealii::FunctionParser<1>>, _n_properties>,
          _n_material_states>> _properties;
  /**
   * Array of vector describing the ratio of each state in each cell. Each
   * vector corresponds to a state defined in the MateralState enum.
   */
  std::array<dealii::LA::distributed::Vector<double>, _n_material_states>
      _state;
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
template <typename NumberType>
double MaterialProperty<dim>::get(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    Property prop, dealii::LA::distributed::Vector<NumberType> const &) const
{
  // TODO: For now, ignore field_state since we have a linear problem.
  double value = 0.;
  dealii::types::material_id material_id = cell->material_id();
  unsigned int property = static_cast<unsigned int>(prop);

  // Get a DoFCellAccessor from a Triangulation::active_cell_iterator.
  dealii::DoFAccessor<dim, dealii::DoFHandler<dim>, false> dof_accessor(
      &_mp_dof_handler.get_triangulation(), cell->level(), cell->index(),
      &_mp_dof_handler);
  std::vector<dealii::types::global_dof_index> mp_dof(1.);
  dof_accessor.get_dof_indices(mp_dof);

  for (unsigned int i = 0; i < _n_material_states; ++i)
  {
    // We cannot use operator[] because the function is constant.
    auto const tmp = _properties.find(material_id);
    ASSERT(tmp != _properties.end(), "Material not found.");
    if ((tmp->second)[i][property] != nullptr)
      value += _state[i][mp_dof[0]] *
               (tmp->second)[i][property]->value(dealii::Point<1>());
  }

  return value;
}

template <int dim>
inline std::array<dealii::LA::distributed::Vector<double>,
                  static_cast<unsigned int>(MaterialState::SIZE)> &
MaterialProperty<dim>::get_state()
{
  return _state;
}
}

#endif
