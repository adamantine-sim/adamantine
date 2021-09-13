/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE MaterialProperty

#include <Geometry.hh>
#include <MaterialProperty.hh>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/property_tree/ptree.hpp>

#include "main.cc"

BOOST_AUTO_TEST_CASE(material_property)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  auto const &triangulation = geometry.get_triangulation();

  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
  }

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "polynomial");
  database.put("n_materials", 1);
  database.put("material_0.solid.density", 1.);
  database.put("material_0.solid.thermal_conductivity", 10.);
  database.put("material_0.powder.conductivity", 10.);
  database.put("material_0.liquid", "");
  database.put("material_0.liquidus", "100");
  adamantine::MaterialProperty<2> mat_prop(communicator, triangulation,
                                           database);
  // Evaluate the material property at the given temperature
  dealii::FE_Q<2> fe(4);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  mat_prop.update(dof_handler, temperature);

  // This check only one cell
  for (auto cell : triangulation.active_cell_iterators())
  {
    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_CHECK(density == 1.);
    double const th_conduc = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity);
    BOOST_CHECK(th_conduc == 10.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_CHECK(liquidus == 100.);
  }
}

BOOST_AUTO_TEST_CASE(ratios)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  auto const &triangulation = geometry.get_triangulation();

  for (auto cell : triangulation.active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(static_cast<int>(adamantine::MaterialState::powder));
  }
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "polynomial");
  database.put("n_materials", 1);
  database.put("material_0.solid.density", 10.);
  database.put("material_0.solid.thermal_conductivity", 10.);
  database.put("material_0.solid.specific_heat", 10.);
  database.put("material_0.powder.density", 1.);
  database.put("material_0.powder.thermal_conductivity", 1.);
  database.put("material_0.powder.specific_heat", 1.);
  database.put("material_0.liquid.density", 1.);
  database.put("material_0.liquid.thermal_conductivity", 20.);
  database.put("material_0.liquid.specific_heat", 20.);
  database.put("material_0.solidus", "50");
  database.put("material_0.liquidus", "100");
  database.put("material_0.latent_heat", "1000");
  adamantine::MaterialProperty<2> mat_prop(communicator, triangulation,
                                           database);
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  mat_prop.update(dof_handler, temperature);

  // Check the material properties of the powder
  for (auto cell : triangulation.active_cell_iterators())
  {
    double powder_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::powder);
    BOOST_CHECK(powder_ratio == 1.);
    double solid_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::solid);
    BOOST_CHECK(solid_ratio == 0.);
    double liquid_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::liquid);
    BOOST_CHECK(liquid_ratio == 0.);

    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_CHECK(density == 1.);
    double const th_conduc = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity);
    BOOST_CHECK(th_conduc == 1.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_CHECK(liquidus == 100.);
    double const solidus =
        mat_prop.get_cell_value(cell, adamantine::Property::solidus);
    BOOST_CHECK(solidus == 50.);
    double const latent_heat =
        mat_prop.get_cell_value(cell, adamantine::Property::latent_heat);
    BOOST_CHECK(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::specific_heat);
    BOOST_CHECK(specific_heat == 1.);
  }

  // Check the material properties of the liquid
  dealii::types::global_dof_index const n_dofs = dof_handler.n_dofs();
  dealii::LA::distributed::Vector<double> avg_temperature(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    avg_temperature[i] = 100000.;
  mat_prop.update(dof_handler, avg_temperature);
  for (auto cell : triangulation.active_cell_iterators())
  {
    double powder_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::powder);
    BOOST_CHECK(powder_ratio == 0.);
    double solid_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::solid);
    BOOST_CHECK(solid_ratio == 0.);
    double liquid_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::liquid);
    BOOST_CHECK(liquid_ratio == 1.);

    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_CHECK(density == 1.);
    double const th_conduc = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity);
    BOOST_CHECK(th_conduc == 20.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_CHECK(liquidus == 100.);
    double const solidus =
        mat_prop.get_cell_value(cell, adamantine::Property::solidus);
    BOOST_CHECK(solidus == 50.);
    double const latent_heat =
        mat_prop.get_cell_value(cell, adamantine::Property::latent_heat);
    BOOST_CHECK(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::specific_heat);
    BOOST_CHECK(specific_heat == 20.);
  }

  // Check the material properties of the solid
  avg_temperature = 0.;
  mat_prop.update(dof_handler, avg_temperature);
  for (auto cell : triangulation.active_cell_iterators())
  {
    double powder_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::powder);
    BOOST_CHECK(powder_ratio == 0.);
    double solid_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::solid);
    BOOST_CHECK(solid_ratio == 1.);
    double liquid_ratio =
        mat_prop.get_state_ratio(cell, adamantine::MaterialState::liquid);
    BOOST_CHECK(liquid_ratio == 0.);

    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_CHECK(density == 10.);
    double const th_conduc = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity);
    BOOST_CHECK(th_conduc == 10.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_CHECK(liquidus == 100.);
    double const solidus =
        mat_prop.get_cell_value(cell, adamantine::Property::solidus);
    BOOST_CHECK(solidus == 50.);
    double const latent_heat =
        mat_prop.get_cell_value(cell, adamantine::Property::latent_heat);
    BOOST_CHECK(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::specific_heat);
    BOOST_CHECK(specific_heat == 10.);
  }
}

BOOST_AUTO_TEST_CASE(material_property_table)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  auto const &triangulation = geometry.get_triangulation();

  unsigned int n = 0;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
      cell->set_material_id(0);
    else
      cell->set_material_id(1);

    if (n < 15)
      cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
    else
      cell->set_user_index(static_cast<int>(adamantine::MaterialState::powder));

    ++n;
  }

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "table");
  database.put("n_materials", 2);
  database.put("material_0.solid.density", "0., 1.");
  database.put("material_0.solid.thermal_conductivity", "0., 10.; 10., 100.");
  database.put("material_1.solid.density", "0., 1.; 20., 2.; 30., 3.");
  database.put("material_1.solid.thermal_conductivity",
               "0., 10.; 10., 100.; 20., 200.");
  database.put("material_1.powder.density", "0., 1.; 15., 2.; 30., 3.");
  database.put("material_1.powder.thermal_conductivity",
               "0., 10.; 10., 100.; 18., 200.");
  adamantine::MaterialProperty<2> mat_prop(communicator, triangulation,
                                           database);
  // Evaluate the material property at the given temperature
  dealii::FE_Q<2> fe(4);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
    temperature.local_element(i) = 15;
  mat_prop.update(dof_handler, temperature);

  n = 0;
  double constexpr tolerance = 1e-10;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
    {
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(cell, adamantine::StateProperty::density), 1.,
          tolerance);
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(
              cell, adamantine::StateProperty::thermal_conductivity),
          100., tolerance);
    }
    else if (n < 15)
    {
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(cell, adamantine::StateProperty::density),
          1.75, tolerance);
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(
              cell, adamantine::StateProperty::thermal_conductivity),
          150., tolerance);
    }
    else
    {
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(cell, adamantine::StateProperty::density), 2.,
          tolerance);
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(
              cell, adamantine::StateProperty::thermal_conductivity),
          162.5, tolerance);
    }
    ++n;
  }
}

BOOST_AUTO_TEST_CASE(material_property_polynomials)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  auto const &triangulation = geometry.get_triangulation();

  unsigned int n = 0;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
      cell->set_material_id(0);
    else
      cell->set_material_id(1);

    if (n < 15)
      cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
    else
      cell->set_user_index(static_cast<int>(adamantine::MaterialState::powder));

    ++n;
  }

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "polynomial");
  database.put("n_materials", 2);
  database.put("material_0.solid.density", "0., 1.");
  database.put("material_0.solid.thermal_conductivity", "0., 1., 2.");
  database.put("material_1.solid.density", " 1., 2., 3.");
  database.put("material_1.solid.thermal_conductivity", "1.,  100., 20., 200.");
  database.put("material_1.powder.density", "15., 2., 3.");
  database.put("material_1.powder.thermal_conductivity", " 10., 18., 200.");
  adamantine::MaterialProperty<2> mat_prop(communicator, triangulation,
                                           database);
  // Evaluate the material property at the given temperature
  dealii::FE_Q<2> fe(4);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
    temperature.local_element(i) = 15;
  mat_prop.update(dof_handler, temperature);

  n = 0;
  double constexpr tolerance = 1e-10;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
    {
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(cell, adamantine::StateProperty::density),
          15., tolerance);
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(
              cell, adamantine::StateProperty::thermal_conductivity),
          465., tolerance);
    }
    else if (n < 15)
    {
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(cell, adamantine::StateProperty::density),
          706., tolerance);
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(
              cell, adamantine::StateProperty::thermal_conductivity),
          681001., tolerance);
    }
    else
    {
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(cell, adamantine::StateProperty::density),
          720., tolerance);
      BOOST_CHECK_CLOSE(
          mat_prop.get_cell_value(
              cell, adamantine::StateProperty::thermal_conductivity),
          45280., tolerance);
    }
    ++n;
  }
}
