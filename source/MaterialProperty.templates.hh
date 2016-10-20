/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _MATERIAL_PROPERTY_TEMPLATES_HH
#define _MATERIAL_PROPERTY_TEMPLATES_HH

#include "MaterialProperty.hh"
#include <deal.II/base/point.h>
#include <deal.II/grid/filtered_iterator.h>
#include <boost/optional.hpp>

namespace adamantine
{

template <int dim>
MaterialProperty<dim>::MaterialProperty(
    boost::mpi::communicator const &communicator,
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
  set_state();

  // Fill the _properties map
  fill_properties(database);
}

template <int dim>
void MaterialProperty<dim>::reinit_dofs()
{
  _mp_dof_handler.distribute_dofs(_fe);
  // Initialize the state vectors
  for (auto &vec : _state)
    vec.reinit(_mp_dof_handler.locally_owned_dofs(), _communicator);
}

template <int dim>
void MaterialProperty<dim>::set_state()
{
  // Set the material state to the one defined by the user_index
  std::vector<dealii::types::global_dof_index> mp_dof(1);
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    cell->get_dof_indices(mp_dof);
    _state[cell->user_index()][mp_dof[0]] = 1.;
  }
}

template <int dim>
void MaterialProperty<dim>::fill_properties(
    boost::property_tree::ptree const &database)
{
  std::array<std::string, _n_material_states> material_state = {
      {"powder", "solid", "liquid"}};
  std::array<std::string, _n_properties> properties = {
      {"density", "latent_heat", "liquidus", "solidus", "specific_heat",
       "thermal_conductivity"}};

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
    bool valid_state = false;
    for (unsigned int state = 0; state < _n_material_states; ++state)
    {
      // The state may or may not exist for the material.
      boost::optional<boost::property_tree::ptree const &> state_database =
          material_database.get_child_optional(material_state[state]);
      if (state_database)
      {
        valid_state = true;
        // For each state, loop over the possible properties.
        for (unsigned int p = 0; p < _n_properties; ++p)
        {
          // The property may or may not exist for that state
          boost::optional<std::string> const property =
              state_database.get().get_optional<std::string>(properties[p]);
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
    // Check that there is at least one valid MaterialState
    ASSERT_THROW(
        valid_state == true,
        "Material without any valid state (solid, powder, or liquid).");

    // Check for the properties that are associated to a material but that
    // are independent of an individual state. These properties are duplicated
    // for every state.
    for (unsigned int p = 0; p < _n_properties; ++p)
    {
      // The property may or may not exist for that state
      boost::optional<std::string> const property =
          material_database.get_optional<std::string>(properties[p]);
      // If the property exists, put it in the map. If the property does not
      // exist, we have a nullptr.
      if (property)
      {
        for (unsigned int state = 0; state < _n_material_states; ++state)
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
}

#endif
