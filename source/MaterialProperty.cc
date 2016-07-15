/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "MaterialProperty.hh"
#include "utils.hh"
#include <deal.II/base/point.h>
#include <boost/optional.hpp>

namespace adamantine
{

MaterialProperty::MaterialProperty(boost::property_tree::ptree const &database)
{
  unsigned int constexpr n_material_states = 3;
  std::array<std::string, n_material_states> material_state = {
      {"powder", "solid", "liquid"}};
  unsigned int constexpr n_properties = 1;
  std::array<std::string, n_properties> properties = {{"themal_conductivity"}};

  unsigned int const n_materials = database.get<unsigned int>("n_materials");
  dealii::types::material_id next_material_id = 0;
  for (unsigned int i = 0; i < n_materials; ++i)
  {
    // Try to find the material_id by checking every possible number.
    dealii::types::material_id material_id = next_material_id;
    while (material_id < dealii::numbers::invalid_material_id)
    {
      // If the child exists, exit the loop.
      if (database.count("material_" + std::to_string(material_id)) != 0)
        break;

      ++material_id;
    }
    ASSERT_THROW(material_id != dealii::numbers::invalid_material_id,
                 "Invalid material ID. Choose a smaller number");
    // Update the next possible material id
    ++next_material_id;

    // Get the material property tree.
    std::string variable = "T";
    boost::property_tree::ptree const &material_database =
        database.get_child("material_" + std::to_string(material_id));
    // For each material, loop over the possible states.
    for (unsigned int state = MaterialState::powder; state < n_material_states;
         ++state)
    {
      boost::property_tree::ptree const &state_database =
          material_database.get_child(material_state[state]);
      // For each state, loop overt the possible properties.
      for (unsigned int p = Property::thermal_conductivity; p < n_properties;
           ++p)
      {
        // The property may or may not exist for that state
        boost::optional<std::string> const property =
            state_database.get<std::string>(properties[p]);
        // If the property exists, put it in the map. If the property does not
        // exist, we have a nullptr.
        if (property)
        {
          _properties[material_id][state][p] =
              std::make_unique<dealii::FunctionParser<1>>(1);
          _properties[material_id][state][p]->initialize(
              variable, property.get(), std::map<std::string, double>());
        }
      }
    }
  }
}

//  // Instantiate the templates
//  template double MaterialProperty::get<2, float>(
//      typename dealii::Triangulation<2>::active_cell_iterator const &cell,
//      Property prop, dealii::LA::distributed::Vector<float> const
//      &field_state) const;
//  template double MaterialProperty::get<2, double>(
//      typename dealii::Triangulation<2>::active_cell_iterator const &cell,
//      Property prop, dealii::LA::distributed::Vector<double> const
//      &field_state) const;
//  template double MaterialProperty::get<3, float>(
//      typename dealii::Triangulation<3>::active_cell_iterator const &cell,
//      Property prop, dealii::LA::distributed::Vector<float> const
//      &field_state) const;
//  template double MaterialProperty::get<3, double>(
//      typename dealii::Triangulation<3>::active_cell_iterator const &cell,
//      Property prop, dealii::LA::distributed::Vector<double> const
//      &field_state) const;
}
