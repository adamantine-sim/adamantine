/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "MaterialProperty.hh"

#include "instantiation.hh"

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

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

  // Compute the alpha and beta constants
  compute_constants();
}

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

  double const mp_dof_index = get_dof_index(cell);

  for (unsigned int i = 0; i < _n_material_states; ++i)
  {
    // We cannot use operator[] because the function is constant.
    auto const tmp = _properties.find(material_id);
    ASSERT(tmp != _properties.end(), "Material not found.");
    if ((tmp->second)[i][property] != nullptr)
      value += _state[i][mp_dof_index] *
               (tmp->second)[i][property]->value(dealii::Point<1>());
  }

  return value;
}

template <int dim>
template <typename NumberType>
dealii::LA::distributed::Vector<NumberType>
MaterialProperty<dim>::enthalpy_to_temperature(
    dealii::DoFHandler<dim> const &enthalpy_dof_handler,
    dealii::LA::distributed::Vector<NumberType> const &enthalpy)
{
  dealii::LA::distributed::Vector<NumberType> temperature(
      enthalpy.get_partitioner());
  dealii::LA::distributed::Vector<NumberType> dummy;

  update_state(enthalpy_dof_handler, enthalpy);

  unsigned int const dofs_per_cell =
      enthalpy_dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (auto cell :
       dealii::filter_iterators(enthalpy_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    cell->get_dof_indices(local_dof_indices);

    double const liquid_ratio = get_state_ratio(cell, MaterialState::liquid);
    double const density = get(cell, Property::density, dummy);
    double const specific_heat = get(cell, Property::specific_heat, dummy);
    double const liquidus = get(cell, Property::liquidus, dummy);
    double const solidus = get(cell, Property::solidus, dummy);
    double const solidus_enthalpy = solidus * density * specific_heat;
    double const latent_heat = get(cell, Property::latent_heat, dummy);
    double const liquidus_enthalpy = solidus_enthalpy + latent_heat;

    NumberType enth_to_temp = [liquid_ratio, liquidus, solidus,
                               liquidus_enthalpy, solidus_enthalpy, density,
                               specific_heat](double const enthalpy) {
      if (liquid_ratio > 0.)
      {
        if (liquid_ratio == 1.)
        {
          return liquidus +
                 (enthalpy - liquidus_enthalpy) / (density * specific_heat);
        }
        else
        {
          return solidus + (liquidus - solidus) *
                               (enthalpy - solidus_enthalpy) /
                               (liquidus_enthalpy - solidus_enthalpy);
        }
      }
      else
      {
        return enthalpy / (density * specific_heat);
      }
    };

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      temperature[local_dof_indices[i]] =
          enth_to_temp(enthalpy[local_dof_indices[i]]);
  }

  return temperature;
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
template <typename NumberType>
void MaterialProperty<dim>::update_state(
    dealii::DoFHandler<dim> const &enthalpy_dof_handler,
    dealii::LA::distributed::Vector<NumberType> const &enthalpy)
{
  dealii::LA::distributed::Vector<NumberType> enthalpy_average =
      compute_average_enthalpy(enthalpy_dof_handler, enthalpy);

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
    unsigned int constexpr prop_solidus =
        static_cast<unsigned int>(Property::solidus);
    unsigned int constexpr prop_density =
        static_cast<unsigned int>(Property::density);
    unsigned int constexpr prop_specific_heat =
        static_cast<unsigned int>(Property::specific_heat);

    dealii::Point<1> const empty_pt;

    double const latent_heat =
        ((tmp->second)[solid][prop_latent_heat] == nullptr)
            ? std::numeric_limits<double>::max()
            : (tmp->second)[solid][prop_latent_heat]->value(empty_pt);
    double const solidus =
        ((tmp->second)[solid][prop_solidus] == nullptr)
            ? std::numeric_limits<double>::max()
            : (tmp->second)[solid][prop_solidus]->value(empty_pt);
    double const solidus_enthalpy =
        solidus * (tmp->second)[solid][prop_density]->value(empty_pt) *
        (tmp->second)[solid][prop_specific_heat]->value(empty_pt);
    double const liquidus_enthalpy = solidus_enthalpy + latent_heat;
    cell->get_dof_indices(mp_dof);
    unsigned int const dof = mp_dof[0];

    // First determine the ratio of liquid.
    double liquid_ratio = -1.;
    double powder_ratio = -1.;
    double solid_ratio = -1.;
    if (enthalpy_average[dof] < solidus_enthalpy)
      liquid_ratio = 0.;
    else if (enthalpy_average[dof] > liquidus_enthalpy)
      liquid_ratio = 1.;
    else
      liquid_ratio = (enthalpy_average[dof] - solidus_enthalpy) / latent_heat;
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

template <int dim>
template <typename NumberType>
dealii::LA::distributed::Vector<NumberType>
MaterialProperty<dim>::compute_average_enthalpy(
    dealii::DoFHandler<dim> const &enthalpy_dof_handler,
    dealii::LA::distributed::Vector<NumberType> const &enthalpy) const
{
  // TODO: this should probably done in a matrix-free fashion.
  // The triangulation is the same for both DoFHandler
  dealii::LA::distributed::Vector<NumberType> enthalpy_average(
      _mp_dof_handler.locally_owned_dofs(), enthalpy.get_mpi_communicator());
  enthalpy_average = 0.;
  auto mp_cell = _mp_dof_handler.begin_active();
  auto mp_end_cell = _mp_dof_handler.end();
  auto enth_cell = enthalpy_dof_handler.begin_active();
  dealii::FiniteElement<dim> const &fe = enthalpy_dof_handler.get_fe();
  // We can use a lower degree of quadrature since we are projecting on a
  // piecewise constant space
  dealii::QGauss<dim> quadrature(fe.degree);
  dealii::FEValues<dim> fe_values(
      fe, quadrature,
      dealii::UpdateFlags::update_values |
          dealii::UpdateFlags::update_quadrature_points |
          dealii::UpdateFlags::update_JxW_values);
  unsigned int const dofs_per_cell = fe.dofs_per_cell;
  std::vector<dealii::types::global_dof_index> mp_dof_indices(1);
  std::vector<dealii::types::global_dof_index> enth_dof_indices(dofs_per_cell);
  unsigned int const n_q_points = quadrature.size();
  for (; mp_cell != mp_end_cell; ++enth_cell, ++mp_cell)
    if (mp_cell->is_locally_owned())
    {
      fe_values.reinit(enth_cell);
      mp_cell->get_dof_indices(mp_dof_indices);
      dealii::types::global_dof_index const mp_dof_index = mp_dof_indices[0];
      enth_cell->get_dof_indices(enth_dof_indices);
      double area = 0.;
      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          area += fe_values.shape_value(i, q) * fe_values.JxW(q);
          enthalpy_average[mp_dof_index] += fe_values.shape_value(i, q) *
                                            enthalpy[enth_dof_indices[i]] *
                                            fe_values.JxW(q);
        }
      enthalpy_average[mp_dof_index] /= area;
    }

  return enthalpy_average;
}
} // namespace adamantine

INSTANTIATE_DIM(MaterialProperty)

namespace adamantine
{
// Instantiate templated function: get
template double MaterialProperty<2>::get(
    dealii::Triangulation<2>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<float> const &) const;
template double MaterialProperty<2>::get(
    dealii::Triangulation<2>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<double> const &) const;
template double MaterialProperty<3>::get(
    dealii::Triangulation<3>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<float> const &) const;
template double MaterialProperty<3>::get(
    dealii::Triangulation<3>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<double> const &) const;

// Instantiate templated function: update_state
template void MaterialProperty<2>::update_state(
    dealii::DoFHandler<2> const &,
    dealii::LA::distributed::Vector<float> const &);
template void MaterialProperty<2>::update_state(
    dealii::DoFHandler<2> const &,
    dealii::LA::distributed::Vector<double> const &);
template void MaterialProperty<3>::update_state(
    dealii::DoFHandler<3> const &,
    dealii::LA::distributed::Vector<float> const &);
template void MaterialProperty<3>::update_state(
    dealii::DoFHandler<3> const &,
    dealii::LA::distributed::Vector<double> const &);
} // namespace adamantine
