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

  // Compute the alpha and beta constants
  compute_constants();
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
void MaterialProperty<dim>::update_state(
    dealii::LA::Vector<double> const &enthalpy)
{
  unsigned int pos = 0;
  std::vector<dealii::types::global_dof_index> mp_dof(1);
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    dealii::types::material_id material_id = cell->material_id();
    auto const tmp = _properties.find(material_id);
    ASSERT(tmp != _properties.end(), "Material not found.");

    // TODO check that assumption enthalpy of solidus powder = enthalpy of
    // solidus solid
    unsigned int constexpr liquid =
        static_cast<unsigned int>(MaterialState::liquid);
    unsigned int constexpr powder =
        static_cast<unsigned int>(MaterialState::powder);
    unsigned int constexpr solid =
        static_cast<unsigned int>(MaterialState::solid);
    unsigned int constexpr prop_latent_heat =
        static_cast<unsigned int>(Property::latent_heat);
    unsigned int constexpr solidus =
        static_cast<unsigned int>(Property::solidus);
    unsigned int constexpr density =
        static_cast<unsigned int>(Property::density);
    unsigned int constexpr specific_heat =
        static_cast<unsigned int>(Property::specific_heat);

    dealii::Point<1> const empty_pt;

    double const latent_heat =
        (tmp->second)[solid][prop_latent_heat]->value(empty_pt);
    double const solidus_enthalpy =
        (tmp->second)[solid][solidus]->value(empty_pt) *
        (tmp->second)[solid][density]->value(empty_pt) *
        (tmp->second)[solid][specific_heat]->value(empty_pt);
    double const liquidus_enthalpy = solidus_enthalpy + latent_heat;
    cell->get_dof_indices(mp_dof);
    unsigned int const dof = mp_dof[0];

    // First determine the ratio of liquid.
    double liquid_ratio = -1.;
    double powder_ratio = -1.;
    double solid_ratio = -1.;
    if (enthalpy[pos] < solidus_enthalpy)
      liquid_ratio = 0.;
    else if (enthalpy[pos] > liquidus_enthalpy)
      liquid_ratio = 1.;
    else
      liquid_ratio = (enthalpy[pos] - solidus_enthalpy) / latent_heat;
    // Because the powder can only become liquid, the solid can only become
    // liquid, and the liquid can only become solid, the ratio of powder can
    // only decrease.
    powder_ratio = std::min(1. - liquid_ratio, _state[powder][dof]);
    // Use max to make sure that we don't create matter because of round-off.
    solid_ratio = std::max(1 - liquid_ratio - powder_ratio, 0.);

    // Update the value
    _state[liquid][dof] = liquid_ratio;
    _state[powder][dof] = powder_ratio;
    _state[solid][dof] = solid_ratio;

    ++pos;
  }
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

template <int dim>
void MaterialProperty<dim>::compute_constants()
{
  unsigned int constexpr liquid =
      static_cast<unsigned int>(MaterialState::liquid);
  unsigned int constexpr solid =
      static_cast<unsigned int>(MaterialState::solid);
  unsigned int constexpr prop_latent_heat =
      static_cast<unsigned int>(Property::latent_heat);
  unsigned int constexpr solidus = static_cast<unsigned int>(Property::solidus);
  unsigned int constexpr liquidus =
      static_cast<unsigned int>(Property::liquidus);
  unsigned int constexpr density = static_cast<unsigned int>(Property::density);
  unsigned int constexpr specific_heat =
      static_cast<unsigned int>(Property::specific_heat);

  dealii::Point<1> const empty_pt;

  for (auto prop = _properties.begin(); prop != _properties.end(); ++prop)
  {
    dealii::types::material_id const material_id = prop->first;
    bool const liquidus_exist =
        (prop->second[solid][liquidus] == nullptr) ? false : true;
    bool const solidus_exist =
        (prop->second[solid][solidus] == nullptr) ? false : true;
    bool const latent_heat_exist =
        (prop->second[solid][prop_latent_heat] == nullptr) ? false : true;
    bool const density_exist =
        (prop->second[solid][density] == nullptr) ? false : true;
    bool const specific_heat_exist =
        (prop->second[solid][specific_heat] == nullptr) ? false : true;

    if (liquidus_exist && solidus_exist && latent_heat_exist)
      _mushy_alpha[material_id] =
          (prop->second[solid][liquidus]->value(empty_pt) -
           prop->second[solid][solidus]->value(empty_pt)) /
          prop->second[solid][prop_latent_heat]->value(empty_pt);

    if (liquidus_exist && solidus_exist && density_exist && specific_heat_exist)
    {
      double const solidus_enthalpy =
          prop->second[solid][solidus]->value(empty_pt) *
          prop->second[solid][density]->value(empty_pt) *
          prop->second[solid][specific_heat]->value(empty_pt);
      _mushy_beta[material_id] =
          -solidus_enthalpy /
              prop->second[solid][prop_latent_heat]->value(empty_pt) *
              (prop->second[solid][liquidus]->value(empty_pt) -
               prop->second[solid][solidus]->value(empty_pt)) +
          prop->second[solid][solidus]->value(empty_pt);
      // TODO this is true only if density and heat capacity are independent of
      // the temperature
      _liquid_beta[material_id] =
          -(solidus_enthalpy +
            prop->second[liquid][prop_latent_heat]->value(empty_pt)) /
              (prop->second[liquid][density]->value(empty_pt) *
               prop->second[liquid][specific_heat]->value(empty_pt)) +
          prop->second[liquid][liquidus]->value(empty_pt);
    }
  }
}
}

#endif
