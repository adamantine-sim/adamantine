/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <MaterialProperty.hh>
#include <instantiation.hh>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <boost/algorithm/string/split.hpp>
#include <boost/optional.hpp>

namespace adamantine
{

template <int dim>
MaterialProperty<dim>::MaterialProperty(
    MPI_Comm const &communicator,
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
double MaterialProperty<dim>::get(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    StateProperty prop) const
{
  unsigned int property = static_cast<unsigned int>(prop);
  auto const mp_dof_index = get_dof_index(cell);

  return _property_values[property][mp_dof_index];
}

template <int dim>
double MaterialProperty<dim>::get(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    Property prop) const
{
  dealii::types::material_id material_id = cell->material_id();
  unsigned int property = static_cast<unsigned int>(prop);

  return _properties.find(material_id)->second[property];
}

template <int dim>
dealii::types::material_id MaterialProperty<dim>::get_material_id(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell) const
{
  return cell->material_id();
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
void MaterialProperty<dim>::update(
    dealii::DoFHandler<dim> const &temperature_dof_handler,
    dealii::LA::distributed::Vector<double> const &temperature)
{
  auto temperature_average =
      compute_average_temperature(temperature_dof_handler, temperature);
  for (auto &val : _property_values)
    val.reinit(temperature_average.get_partitioner());

  std::vector<dealii::types::global_dof_index> mp_dof(1);
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    dealii::types::material_id material_id = cell->material_id();
    auto const tmp = _properties.find(material_id);
    ASSERT(tmp != _properties.end(), "Material not found.");

    unsigned int constexpr liquid =
        static_cast<unsigned int>(MaterialState::liquid);
    unsigned int constexpr powder =
        static_cast<unsigned int>(MaterialState::powder);
    unsigned int constexpr solid =
        static_cast<unsigned int>(MaterialState::solid);
    unsigned int constexpr prop_solidus =
        static_cast<unsigned int>(Property::solidus);
    unsigned int constexpr prop_liquidus =
        static_cast<unsigned int>(Property::liquidus);

    double const solidus = tmp->second[prop_solidus];
    double const liquidus = tmp->second[prop_liquidus];
    cell->get_dof_indices(mp_dof);
    unsigned int const dof = mp_dof[0];

    // First determine the ratio of liquid.
    double liquid_ratio = -1.;
    double powder_ratio = -1.;
    double solid_ratio = -1.;
    if (temperature_average[dof] < solidus)
      liquid_ratio = 0.;
    else if (temperature_average[dof] > liquidus)
      liquid_ratio = 1.;
    else
      liquid_ratio =
          (temperature_average[dof] - solidus) / (liquidus - solidus);
    // Because the powder can only become liquid, the solid can only become
    // liquid, and the liquid can only become solid, the ratio of powder can
    // only decrease.
    powder_ratio = std::min(1. - liquid_ratio, _state[powder][dof]);
    // Use max to make sure that we don't create matter because of round-off.
    solid_ratio = std::max(1. - liquid_ratio - powder_ratio, 0.);

    // Update the value
    _state[liquid][dof] = liquid_ratio;
    _state[powder][dof] = powder_ratio;
    _state[solid][dof] = solid_ratio;

    if (_use_table)
    {
      for (unsigned int property = 0; property < _n_state_properties;
           ++property)
      {
        for (unsigned int material_state = 0;
             material_state < _n_material_states; ++material_state)
        {
          _property_values[property][dof] +=
              _state[material_state][dof] *
              compute_property_from_table(
                  _state_property_tables[material_id][material_state][property],
                  temperature_average[dof]);
        }
      }
    }
    else
    {
      for (unsigned int property = 0; property < _n_state_properties;
           ++property)
      {
        for (unsigned int material_state = 0;
             material_state < _n_material_states; ++material_state)
        {
          _property_values[property][dof] +=
              _state[material_state][dof] *
              compute_property_from_polynomial(
                  _state_property_polynomials[material_id][material_state]
                                             [property],
                  temperature_average[dof]);
        }
      }
    }

    // If we are in the mushy state, i.e., part liquid part solid, we need to
    // modify the rho C_p to take into account the latent heat.
    if ((liquid_ratio > 0.) && (liquid_ratio < 1.))
    {
      unsigned int specific_heat_prop =
          static_cast<unsigned int>(StateProperty::specific_heat);
      unsigned int latent_heat_prop =
          static_cast<unsigned int>(Property::latent_heat);
      for (unsigned int material_state = 0; material_state < _n_material_states;
           ++material_state)
      {
        _property_values[specific_heat_prop][dof] +=
            _state[material_state][dof] *
            _properties[material_id][latent_heat_prop] / (liquidus - solidus);
      }
    }
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
      {"liquidus", "solidus", "latent_heat"}};
  std::array<std::string, _n_state_properties> state_properties = {
      {"density", "specific_heat", "thermal_conductivity"}};

  // PropertyTreeInput materials.property_format
  std::string property_format = database.get<std::string>("property_format");
  ASSERT_THROW((property_format == "table") ||
                   (property_format == "polynomial"),
               "property_format should be table or polynomial.");
  _use_table = (property_format == "table");
  // PropertyTreeInput materials.n_materials
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
                 "Invalid material ID. Choose a smaller number.");
    // Update the next possible material id
    ++next_material_id;

    // Get the material property tree.
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
        for (unsigned int p = 0; p < _n_state_properties; ++p)
        {
          // The property may or may not exist for that state
          boost::optional<std::string> const property =
              state_database.get().get_optional<std::string>(
                  state_properties[p]);
          // If the property exists, put it in the map. If the property does not
          // exist, we have a nullptr.
          if (property)
          {
            // Remove blank spaces
            std::string property_string = property.get();
            property_string.erase(
                std::remove_if(property_string.begin(), property_string.end(),
                               [](unsigned char x) { return std::isspace(x); }),
                property_string.end());
            if (_use_table)
            {
              std::vector<std::string> parsed_property;
              boost::split(parsed_property, property_string,
                           [](char c) { return c == ';'; });
              for (auto const &p_property : parsed_property)
              {
                std::vector<std::string> t_v;
                boost::split(t_v, p_property, [](char c) { return c == ','; });
                ASSERT(t_v.size() == 2, "Error reading material property.");
                _state_property_tables[material_id][state][p].emplace_back(
                    std::stod(t_v[0]), std::stod(t_v[1]));
              }
            }
            else
            {
              std::vector<std::string> parsed_property;
              boost::split(parsed_property, property_string,
                           [](char c) { return c == ','; });
              for (auto const &p_property : parsed_property)
              {
                _state_property_polynomials[material_id][state][p].emplace_back(
                    std::stod(p_property));
              }
            }
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
      boost::optional<double> const property =
          material_database.get_optional<double>(properties[p]);
      // If the property exists, put it in the map. If the property does not
      // exist, we use the largest. This is useful if the liquidus and the
      // solidus are not set.
      _properties[material_id][p] =
          property ? property.get() : std::numeric_limits<double>::max();
    }
  }
}

// We need to compute the average temperature on the cell because we need the
// material properties to be uniform over the cell. If there aren't then we have
// problems with the weak form discretization.
template <int dim>
dealii::LA::distributed::Vector<double>
MaterialProperty<dim>::compute_average_temperature(
    dealii::DoFHandler<dim> const &temperature_dof_handler,
    dealii::LA::distributed::Vector<double> const &temperature) const
{
  // TODO: this should probably done in a matrix-free fashion.
  // The triangulation is the same for both DoFHandler
  dealii::LA::distributed::Vector<double> temperature_average(
      _mp_dof_handler.locally_owned_dofs(), temperature.get_mpi_communicator());
  temperature_average = 0.;
  auto mp_cell = _mp_dof_handler.begin_active();
  auto mp_end_cell = _mp_dof_handler.end();
  auto enth_cell = temperature_dof_handler.begin_active();
  dealii::hp::FECollection<dim> const &fe_collection =
      temperature_dof_handler.get_fe_collection();
  dealii::hp::QCollection<dim> q_collection;
  q_collection.push_back(dealii::QGauss<dim>(fe_collection.max_degree() + 1));
  q_collection.push_back(dealii::QGauss<dim>(1));
  dealii::hp::FEValues<dim> hp_fe_values(
      fe_collection, q_collection,
      dealii::UpdateFlags::update_values |
          dealii::UpdateFlags::update_quadrature_points |
          dealii::UpdateFlags::update_JxW_values);
  std::vector<dealii::types::global_dof_index> mp_dof_indices(1);
  unsigned int const n_q_points = q_collection.max_n_quadrature_points();
  unsigned int const dofs_per_cell = fe_collection.max_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> enth_dof_indices(dofs_per_cell);
  for (; mp_cell != mp_end_cell; ++enth_cell, ++mp_cell)
    if ((mp_cell->is_locally_owned()) && (enth_cell->active_fe_index() == 0))
    {
      hp_fe_values.reinit(enth_cell);
      dealii::FEValues<dim> const &fe_values =
          hp_fe_values.get_present_fe_values();
      mp_cell->get_dof_indices(mp_dof_indices);
      dealii::types::global_dof_index const mp_dof_index = mp_dof_indices[0];
      enth_cell->get_dof_indices(enth_dof_indices);
      double volume = 0.;
      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          volume += fe_values.shape_value(i, q) * fe_values.JxW(q);
          temperature_average[mp_dof_index] +=
              fe_values.shape_value(i, q) * temperature[enth_dof_indices[i]] *
              fe_values.JxW(q);
        }
      temperature_average[mp_dof_index] /= volume;
    }

  return temperature_average;
}

template <int dim>
double MaterialProperty<dim>::compute_property_from_table(
    std::vector<std::pair<double, double>> const &table,
    double const temperature) const
{
  if (table.size() == 0)
    return 0.;
  else
  {
    if (temperature <= table.front().first)
      return table.front().second;
    else if (temperature >= table.back().first)
      return table.back().second;
    else
    {
      auto it = std::find_if(table.begin(), table.end(),
                             [=](std::pair<double, double> const &t_v) {
                               return t_v.first > temperature;
                             });
      auto prev_it = it - 1;
      return prev_it->second + (temperature - prev_it->first) *
                                   (it->second - prev_it->second) /
                                   (it->first - prev_it->first);
    }
  }
}

template <int dim>
double MaterialProperty<dim>::compute_property_from_polynomial(
    std::vector<double> const &coef, double const temperature) const
{
  double value = 0.;
  unsigned int const n_coef = coef.size();
  for (unsigned int i = 0; i < n_coef; ++i)
  {
    value += coef[i] * std::pow(temperature, i);
  }

  return value;
}

template <int dim>
void MaterialProperty<dim>::update_state_ratios(
    unsigned int cell, unsigned int q,
    dealii::VectorizedArray<double> temperature,
    std::array<dealii::VectorizedArray<double>,
               static_cast<unsigned int>(MaterialState::SIZE)> &state_ratios)
{

  unsigned int constexpr liquid =
      static_cast<unsigned int>(MaterialState::liquid);
  unsigned int constexpr powder =
      static_cast<unsigned int>(MaterialState::powder);
  unsigned int constexpr solid =
      static_cast<unsigned int>(MaterialState::solid);
  unsigned int constexpr prop_solidus =
      static_cast<unsigned int>(Property::solidus);
  unsigned int constexpr prop_liquidus =
      static_cast<unsigned int>(Property::liquidus);

  // Loop over the vectorized arrays
  for (unsigned int n = 0; n < temperature.size(); ++n)
  {
    // Get the material id at this point
    dealii::types::material_id material_id = _material_id(cell, q)[n];

    // Get the material thermodynamic properties

    auto const tmp = _properties.find(material_id);
    ASSERT(tmp != _properties.end(), "Material not found.");

    double const solidus = tmp->second[prop_solidus];
    double const liquidus = tmp->second[prop_liquidus];

    // Update the state ratios
    state_ratios[powder] = _powder_ratio(cell, q);

    if (temperature[n] < solidus)
      state_ratios[liquid][n] = 0.;
    else if (temperature[n] > liquidus)
      state_ratios[liquid][n] = 1.;
    else
    {
      state_ratios[liquid][n] =
          (temperature[n] - solidus) / (liquidus - solidus);
    }
    // Because the powder can only become liquid, the solid can only
    // become liquid, and the liquid can only become solid, the ratio of
    // powder can only decrease.
    state_ratios[powder][n] =
        std::min(1. - state_ratios[liquid][n], state_ratios[powder][n]);
    // Use max to make sure that we don't create matter because of
    // round-off.
    state_ratios[solid][n] =
        std::max(1. - state_ratios[liquid][n] - state_ratios[powder][n], 0.);
  }

  _powder_ratio(cell, q) = state_ratios[powder];
}

template <int dim>
dealii::VectorizedArray<double> MaterialProperty<dim>::get_inv_rho_cp(
    unsigned int cell, unsigned int q,
    std::array<dealii::VectorizedArray<double>,
               static_cast<unsigned int>(MaterialState::SIZE)>
        state_ratios,
    dealii::VectorizedArray<double> temperature)
{
  // Here we need the specific heat (including the latent heat contribution)
  // and the density

  // First, get the state-independent material properties
  unsigned int constexpr prop_solidus =
      static_cast<unsigned int>(Property::solidus);
  unsigned int constexpr prop_liquidus =
      static_cast<unsigned int>(Property::liquidus);
  unsigned int constexpr prop_latent_heat =
      static_cast<unsigned int>(Property::latent_heat);

  dealii::VectorizedArray<double> solidus, liquidus, latent_heat;

  for (unsigned int n = 0; n < solidus.size(); ++n)
  {
    auto const tmp = _properties.find(_material_id(cell, q)[n]);
    ASSERT(tmp != _properties.end(), "Material not found.");

    solidus[n] = tmp->second[prop_solidus];
    liquidus[n] = tmp->second[prop_liquidus];
    latent_heat[n] = tmp->second[prop_latent_heat];
  }

  // Now compute the state-dependent properties
  // std::cout << "density" << std::endl;
  dealii::VectorizedArray<double> density = compute_material_property(
      StateProperty::density, _material_id(cell, q), state_ratios, temperature);

  // std::cout << "specific_heat" << std::endl;
  dealii::VectorizedArray<double> specific_heat = compute_material_property(
      StateProperty::specific_heat, _material_id(cell, q), state_ratios,
      temperature);

  // Add in the latent heat contribution
  unsigned int constexpr liquid =
      static_cast<unsigned int>(MaterialState::liquid);

  for (unsigned int n = 0; n < specific_heat.size(); ++n)
  {
    if (state_ratios[liquid][n] > 0.0 && (state_ratios[liquid][n] < 1.0))
    {
      specific_heat[n] += latent_heat[n] / (liquidus[n] - solidus[n]);
    }
  }

  return 1.0 / (density * specific_heat);
}

template <int dim>
dealii::VectorizedArray<double> MaterialProperty<dim>::get_thermal_conductivity(
    unsigned int cell, unsigned int q,
    std::array<dealii::VectorizedArray<double>,
               static_cast<unsigned int>(MaterialState::SIZE)>
        state_ratios,
    dealii::VectorizedArray<double> temperature)
{
  return compute_material_property(StateProperty::thermal_conductivity,
                                   _material_id(cell, q), state_ratios,
                                   temperature);
}

template <int dim>
dealii::VectorizedArray<double>
MaterialProperty<dim>::compute_material_property(
    StateProperty state_property,
    std::array<dealii::types::material_id,
               dealii::VectorizedArray<double>::size()>
        material_id,
    std::array<dealii::VectorizedArray<double>,
               static_cast<unsigned int>(MaterialState::SIZE)>
        state_ratios,
    dealii::VectorizedArray<double> temperature) const
{
  dealii::VectorizedArray<double> value = 0.0;
  unsigned int property_index = static_cast<unsigned int>(state_property);

  if (_use_table)
  {
    for (unsigned int material_state = 0; material_state < _n_material_states;
         ++material_state)
    {
      for (unsigned int n = 0; n < temperature.size(); ++n)
      {

        const dealii::types::material_id m_id = material_id[n];

        value[n] +=
            state_ratios[material_state][n] *
            compute_property_from_table(
                _state_property_tables.at(m_id)[material_state][property_index],
                temperature[n]);
      }
    }
  }
  else
  {
    for (unsigned int material_state = 0; material_state < _n_material_states;
         ++material_state)
    {
      for (unsigned int n = 0; n < temperature.size(); ++n)
      {

        dealii::types::material_id m_id = material_id[n];

        value[n] += state_ratios[material_state][n] *
                    compute_property_from_polynomial(
                        _state_property_polynomials.at(
                            m_id)[material_state][property_index],
                        temperature[n]);
      }
    }
  }
  return value;
}

} // namespace adamantine

INSTANTIATE_DIM(MaterialProperty)
