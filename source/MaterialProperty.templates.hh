/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <MaterialProperty.hh>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <boost/algorithm/string/split.hpp>
#include <boost/optional.hpp>

#include <algorithm>

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
  set_initial_state();

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

  return _properties(material_id, property);
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

    double const solidus = _properties(material_id, prop_solidus);
    double const liquidus = _properties(material_id, prop_liquidus);
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
    solid_ratio = std::max(1 - liquid_ratio - powder_ratio, 0.);

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
              compute_property_from_table(material_id, material_state, property,
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
          for (unsigned int i = 0; i <= _polynomial_order; ++i)
          {
            _property_values[property][dof] +=
                _state[material_state][dof] *
                _state_property_polynomials(material_id, material_state,
                                            property, i) *
                std::pow(temperature_average[dof], i);
          }
        }
      }
    }

    // If we are in the mushy state, i.e., part liquid part solid, we need to
    // modify the rho C_p to take into account the latent heat.
    if ((liquid_ratio > 0.) && (liquid_ratio < 1.))
    {
      unsigned int const specific_heat_prop =
          static_cast<unsigned int>(StateProperty::specific_heat);
      unsigned int const latent_heat_prop =
          static_cast<unsigned int>(Property::latent_heat);
      for (unsigned int material_state = 0; material_state < _n_material_states;
           ++material_state)
      {
        _property_values[specific_heat_prop][dof] +=
            _state[material_state][dof] *
            _properties(material_id, latent_heat_prop) / (liquidus - solidus);
      }
    }

    // The radiation heat transfer coefficient is not a real material property
    // but it is derived from other material properties:
    // h_rad = emissitivity * stefan-boltzmann constant * (T + T_infty) (T^2 +
    // T^2_infty).
    unsigned int const emissivity_prop =
        static_cast<unsigned int>(StateProperty::emissivity);
    unsigned int const radiation_heat_transfer_coef_prop =
        static_cast<unsigned int>(StateProperty::radiation_heat_transfer_coef);
    unsigned int const radiation_temperature_infty_prop =
        static_cast<unsigned int>(Property::radiation_temperature_infty);
    double const T = temperature_average[dof];
    double const T_infty =
        _properties(material_id, radiation_temperature_infty_prop);
    double const emissivity = _property_values[emissivity_prop][dof];
    _property_values[radiation_heat_transfer_coef_prop][dof] =
        emissivity * Constant::stefan_boltzmann * (T + T_infty) *
        (T * T + T_infty * T_infty);
  }
}

template <int dim>
void MaterialProperty<dim>::update_boundary_material_properties(
    dealii::DoFHandler<dim> const &temperature_dof_handler,
    dealii::LA::distributed::Vector<double> const &temperature)
{
  auto temperature_average =
      compute_average_temperature(temperature_dof_handler, temperature);
  for (auto &val : _property_values)
    val.reinit(temperature_average.get_partitioner());

  std::vector<dealii::types::global_dof_index> mp_dof(1);
  // We don't need to loop over all the active cells. We only need to loop over
  // the cells at the boundary and at the interface with FE_Nothing. However, to
  // do this we need to use the temperature_dof_handler instead of the
  // _mp_dof_handler.
  for (auto cell :
       dealii::filter_iterators(_mp_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    dealii::types::material_id material_id = cell->material_id();

    cell->get_dof_indices(mp_dof);
    unsigned int const dof = mp_dof[0];
    if (_use_table)
    {
      // We only care about properties that are used to compute the boundary
      // condition. So we start at 3.
      for (unsigned int property = 3; property < _n_state_properties;
           ++property)
      {
        for (unsigned int material_state = 0;
             material_state < _n_material_states; ++material_state)
        {
          _property_values[property][dof] +=
              _state[material_state][dof] *
              compute_property_from_table(material_id, material_state, property,
                                          temperature_average[dof]);
        }
      }
    }
    else
    {
      // We only care about properties that are used to compute the boundary
      // condition. So we start at  3.
      for (unsigned int property = 3; property < _n_state_properties;
           ++property)
      {
        for (unsigned int material_state = 0;
             material_state < _n_material_states; ++material_state)
        {
          for (unsigned int i = 0; i <= _polynomial_order; ++i)
          {
            _property_values[property][dof] +=
                _state[material_state][dof] *
                _state_property_polynomials(material_id, material_state,
                                            property, i) *
                std::pow(temperature_average[dof], i);
          }
        }
      }
    }

    // The radiation heat transfer coefficient is not a real material property
    // but it is derived from other material properties:
    // h_rad = emissitivity * stefan-boltzmann constant * (T + T_infty) (T^2 +
    // T^2_infty).
    unsigned int const emissivity_prop =
        static_cast<unsigned int>(StateProperty::emissivity);
    unsigned int const radiation_heat_transfer_coef_prop =
        static_cast<unsigned int>(StateProperty::radiation_heat_transfer_coef);
    unsigned int const radiation_temperature_infty_prop =
        static_cast<unsigned int>(Property::radiation_temperature_infty);
    double const T = temperature_average[dof];
    double const T_infty =
        _properties(material_id, radiation_temperature_infty_prop);
    double const emissivity = _property_values[emissivity_prop][dof];
    _property_values[radiation_heat_transfer_coef_prop][dof] =
        emissivity * Constant::stefan_boltzmann * (T + T_infty) *
        (T * T + T_infty * T_infty);
  }
}

template <int dim>
ADAMANTINE_HOST_DEV dealii::VectorizedArray<double>
MaterialProperty<dim>::compute_material_property(
    StateProperty state_property, dealii::types::material_id const *material_id,
    dealii::VectorizedArray<double> const *state_ratios,
    dealii::VectorizedArray<double> temperature) const
{
  dealii::VectorizedArray<double> value = 0.0;
  unsigned int const property_index = static_cast<unsigned int>(state_property);

  if (_use_table)
  {
    for (unsigned int material_state = 0; material_state < _n_material_states;
         ++material_state)
    {
      for (unsigned int n = 0; n < dealii::VectorizedArray<double>::size(); ++n)
      {

        const dealii::types::material_id m_id = material_id[n];

        value[n] += state_ratios[material_state][n] *
                    compute_property_from_table(m_id, material_state,
                                                property_index, temperature[n]);
      }
    }
  }
  else
  {
    for (unsigned int material_state = 0; material_state < _n_material_states;
         ++material_state)
    {
      for (unsigned int n = 0; n < dealii::VectorizedArray<double>::size(); ++n)
      {

        dealii::types::material_id m_id = material_id[n];

        for (unsigned int i = 0; i <= _polynomial_order; ++i)
        {
          value[n] += state_ratios[material_state][n] *
                      _state_property_polynomials(m_id, material_state,
                                                  property_index, i) *
                      std::pow(temperature[n], i);
        }
      }
    }
  }

  return value;
}

template <int dim>
void MaterialProperty<dim>::set_state(
    dealii::Table<2, dealii::VectorizedArray<double>> const &liquid_ratio,
    dealii::Table<2, dealii::VectorizedArray<double>> const &powder_ratio,
    std::map<typename dealii::DoFHandler<dim>::cell_iterator,
             std::pair<unsigned int, unsigned int>> &cell_it_to_mf_cell_map,
    dealii::DoFHandler<dim> const &dof_handler)
{
  auto const powder_state = static_cast<unsigned int>(MaterialState::powder);
  auto const liquid_state = static_cast<unsigned int>(MaterialState::liquid);
  auto const solid_state = static_cast<unsigned int>(MaterialState::solid);
  std::vector<dealii::types::global_dof_index> mp_dof(1.);

  for (auto cell :
       dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(cell);
    auto mp_dof_index = get_dof_index(cell_tria);
    auto const &mf_cell_vector = cell_it_to_mf_cell_map[cell];
    unsigned int const n_q_points = dof_handler.get_fe().tensor_degree() + 1;
    double liquid_ratio_sum = 0.;
    double powder_ratio_sum = 0.;
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      liquid_ratio_sum +=
          liquid_ratio(mf_cell_vector.first, q)[mf_cell_vector.second];
      powder_ratio_sum +=
          powder_ratio(mf_cell_vector.first, q)[mf_cell_vector.second];
    }
    _state[liquid_state][mp_dof_index] = liquid_ratio_sum / n_q_points;
    _state[powder_state][mp_dof_index] = powder_ratio_sum / n_q_points;
    _state[solid_state][mp_dof_index] =
        std::max(1. - _state[liquid_state][mp_dof_index] -
                     _state[powder_state][mp_dof_index],
                 0.);
  }
}

template <int dim>
void MaterialProperty<dim>::set_initial_state()
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
      {"liquidus", "solidus", "latent_heat", "radiation_temperature_infty",
       "convection_temperature_infty"}};
  std::array<std::string, _n_state_properties> state_properties = {
      {"density", "specific_heat", "thermal_conductivity", "emissivity",
       "radiation_heat_transfer_coef", "convection_heat_transfer_coef"}};

  // PropertyTreeInput materials.property_format
  std::string property_format = database.get<std::string>("property_format");
  ASSERT_THROW((property_format == "table") ||
                   (property_format == "polynomial"),
               "property_format should be table or polynomial.");
  _use_table = (property_format == "table");
  // PropertyTreeInput materials.n_materials
  unsigned int const n_materials = database.get<unsigned int>("n_materials");
  // Find all the material_ids being used.
  std::vector<dealii::types::material_id> material_ids;
  for (dealii::types::material_id id = 0;
       id < dealii::numbers::invalid_material_id; ++id)
  {
    if (database.count("material_" + std::to_string(id)) != 0)
      material_ids.push_back(id);
    if (material_ids.size() == n_materials)
      break;
  }
  ASSERT_THROW(material_ids.size() == n_materials,
               "Could not find all the material_ids.");

  // When using the polynomial format we allocate one contiguous block of
  // memory. Thus, the largest material_id should be as small as possible
  unsigned int const n_material_ids =
      *std::max_element(material_ids.begin(), material_ids.end()) + 1;
  _properties.reinit(n_material_ids);
  if (_use_table)
  {
    // TODO read table size from input file
    _table_size = 4;
    _state_property_tables.reinit(n_material_ids, _table_size);
    _state_property_tables.set_zero();
  }
  else
  {
    // TODO read polynomial_order from input file
    _polynomial_order = 4;
    _state_property_polynomials.reinit(n_material_ids + 1,
                                       _polynomial_order + 1);
    _state_property_polynomials.set_zero();
  }

  for (auto const material_id : material_ids)
  {
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
              unsigned int const parsed_property_size = parsed_property.size();
              ASSERT_THROW(parsed_property_size <= _table_size,
                           "Too many coefficients, increase the table size");
              for (unsigned int i = 0; i < parsed_property_size; ++i)
              {
                std::vector<std::string> t_v;
                boost::split(t_v, parsed_property[i],
                             [](char c) { return c == ','; });
                ASSERT(t_v.size() == 2, "Error reading material property.");
                _state_property_tables(material_id, state, p, i, 0) =
                    std::stod(t_v[0]);
                _state_property_tables(material_id, state, p, i, 1) =
                    std::stod(t_v[1]);
              }
              // fill the rest  with the last value
              for (unsigned int i = parsed_property_size; i < _table_size; ++i)
              {
                _state_property_tables(material_id, state, p, i, 0) =
                    _state_property_tables(material_id, state, p, i - 1, 0);
                _state_property_tables(material_id, state, p, i, 1) =
                    _state_property_tables(material_id, state, p, i - 1, 1);
              }
            }
            else
            {
              std::vector<std::string> parsed_property;
              boost::split(parsed_property, property_string,
                           [](char c) { return c == ','; });
              unsigned int const parsed_property_size = parsed_property.size();
              ASSERT_THROW(
                  parsed_property_size <= _polynomial_order,
                  "Too many coefficients, increase the polynomial order");
              for (unsigned int i = 0; i < parsed_property_size; ++i)
              {
                _state_property_polynomials(material_id, state, p, i) =
                    std::stod(parsed_property[i]);
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
      // exist, we use the largest possible value. This is useful if the
      // liquidus and the solidus are not set.
      _properties(material_id, p) =
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
  temperature.update_ghost_values();
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
  {
    ASSERT(mp_cell->is_locally_owned() == enth_cell->is_locally_owned(),
           "Internal Error");
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
  }

  return temperature_average;
}

template <int dim>
double MaterialProperty<dim>::compute_property_from_table(
    unsigned int const material_id, unsigned int const material_state,
    unsigned int const property, double const temperature) const
{
  if (temperature <=
      _state_property_tables(material_id, material_state, property, 0, 0))
  {
    return _state_property_tables(material_id, material_state, property, 0, 1);
  }
  else
  {
    unsigned int i = 0;
    unsigned int const size = _state_property_tables.extent(3);
    for (; i < size; ++i)
    {
      if (temperature <
          _state_property_tables(material_id, material_state, property, i, 0))
      {
        break;
      }
    }

    if (i >= size - 1)
    {
      return _state_property_tables(material_id, material_state, property,
                                    size - 1, 1);
    }
    else
    {
      auto tempertature_i =
          _state_property_tables(material_id, material_state, property, i, 0);
      auto tempertature_im1 = _state_property_tables(
          material_id, material_state, property, i - 1, 0);
      auto property_i =
          _state_property_tables(material_id, material_state, property, i, 1);
      auto property_im1 = _state_property_tables(material_id, material_state,
                                                 property, i - 1, 1);
      return property_im1 + (temperature - tempertature_im1) *
                                (property_i - property_im1) /
                                (tempertature_i - tempertature_im1);
    }
  }
}

} // namespace adamantine
