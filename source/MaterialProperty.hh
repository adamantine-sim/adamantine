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
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <boost/property_tree/ptree.hpp>
#include <array>
#include <unordered_map>

namespace adamantine
{
/**
 * This class stores the material properties for all the materials
 */
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
  MaterialProperty(boost::property_tree::ptree const &database);

  /**
   * Return the value of the given property, for a given cell and a given field
   * state.
   */
  template <int dim, typename NumberType>
  double
  get(typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      Property prop,
      dealii::LA::distributed::Vector<NumberType> const &field_state) const;

private:
  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr _n_material_states = 3;
  /**
   * Number of properties defined.
   */
  static unsigned int constexpr _n_properties = 3;
  /**
   * Map that stores functions describing the properties of the material.
   */
  std::unordered_map<
      dealii::types::material_id,
      std::array<
          std::array<std::unique_ptr<dealii::FunctionParser<1>>, _n_properties>,
          _n_material_states>> _properties;
};

template <int dim, typename NumberType>
double MaterialProperty::get(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    Property prop,
    dealii::LA::distributed::Vector<NumberType> const &field_state) const
{
  // TODO: For now, ignore field_state since we have a linear problem. Also for
  // now can only be in one state. It can't be half powder and half liquid.
  (void)field_state;
  dealii::types::material_id material_id = cell->material_id();
  MaterialState state = static_cast<MaterialState>(cell->user_index());

  // We cannot use operator[] because the function is constant.
  auto const tmp = _properties.find(material_id);
  ASSERT(tmp != _properties.end(), "Material not found.");
  ASSERT((tmp->second)[state][prop] != nullptr, "Property not found.");
  return (tmp->second)[state][prop]->value(dealii::Point<1>());
}
}

#endif
