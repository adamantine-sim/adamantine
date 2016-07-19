/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _MATERIAL_PROPERTY_HH_
#define _MATERIAL_PROPERTY_HH_

#include "types.hh"
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <boost/property_tree/ptree.hpp>
#include <array>
#include <unordered_map>

namespace adamantine
{

class MaterialProperty
{
public:
  MaterialProperty(boost::property_tree::ptree const &database);

  template <int dim, typename NumberType>
  double
  get(typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      Property prop,
      dealii::LA::distributed::Vector<NumberType> const &field_state) const;

private:
  // material_id - (powder/solid/liquid, property)
  std::unordered_map<
      dealii::types::material_id,
      std::array<std::array<std::unique_ptr<dealii::FunctionParser<1>>, 1>, 3>>
      _properties;
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
  return (tmp->second)[state][prop]->value(dealii::Point<1>());
}
}

#endif
