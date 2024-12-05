/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "MaterialStates.hh"
#include <Geometry.hh>
#include <MaterialProperty.hh>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/property_tree/ptree.hpp>

namespace tt = boost::test_tools;

template <typename MemorySpaceType>
void material_property()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);
  auto const &triangulation = geometry.get_triangulation();

  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
  }

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "polynomial");
  database.put("n_materials", 1);
  database.put("material_0.solid.density", 1.);
  database.put("material_0.solid.thermal_conductivity_x", 10.);
  database.put("material_0.solid.thermal_conductivity_z", 10.);
  database.put("material_0.powder.thermal_conductivity_x", 10.);
  database.put("material_0.powder.thermal_conductivity_z", 10.);
  database.put("material_0.liquid", "");
  database.put("material_0.liquidus", "100");
  database.put("material_0.solid.lame_first_parameter", 2.);
  database.put("material_0.solid.lame_second_parameter", 3.);
  adamantine::MaterialProperty<2, 2, adamantine::SolidLiquidPowder,
                               MemorySpaceType>
      mat_prop(communicator, triangulation, database);
  // Evaluate the material property at the given temperature
  dealii::FE_Q<2> fe(4);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  mat_prop.update(dof_handler, temperature);

  // This checks only one cell
  for (auto cell : triangulation.active_cell_iterators())
  {
    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_TEST(density == 1.);
    double const th_conduc_x = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_x);
    BOOST_TEST(th_conduc_x == 10.);
    double const th_conduc_z = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_z);
    BOOST_TEST(th_conduc_z == 10.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_TEST(liquidus == 100.);
    double const lambda = mat_prop.get_mechanical_property(
        cell, adamantine::StateProperty::lame_first_parameter);
    BOOST_TEST(lambda == 2.);
    double const mu = mat_prop.get_mechanical_property(
        cell, adamantine::StateProperty::lame_second_parameter);
    BOOST_TEST(mu == 3.);
  }
}

template <typename MemorySpaceType>
void ratios()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);
  auto const &triangulation = geometry.get_triangulation();

  for (auto cell : triangulation.active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::powder));
  }
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "polynomial");
  database.put("n_materials", 1);
  database.put("material_0.solid.density", 10.);
  database.put("material_0.solid.thermal_conductivity_x", 10.);
  database.put("material_0.solid.thermal_conductivity_z", 10.);
  database.put("material_0.solid.specific_heat", 10.);
  database.put("material_0.powder.density", 1.);
  database.put("material_0.powder.thermal_conductivity_x", 1.);
  database.put("material_0.powder.thermal_conductivity_z", 1.);
  database.put("material_0.powder.specific_heat", 1.);
  database.put("material_0.liquid.density", 1.);
  database.put("material_0.liquid.thermal_conductivity_x", 20.);
  database.put("material_0.liquid.thermal_conductivity_z", 20.);
  database.put("material_0.liquid.specific_heat", 20.);
  database.put("material_0.solidus", "50");
  database.put("material_0.liquidus", "100");
  database.put("material_0.latent_heat", "1000");
  adamantine::MaterialProperty<2, 0, adamantine::SolidLiquidPowder,
                               MemorySpaceType>
      mat_prop(communicator, triangulation, database);
  dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  mat_prop.update(dof_handler, temperature);

  // Check the material properties of the powder
  for (auto cell : triangulation.active_cell_iterators())
  {
    double powder_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::powder);
    BOOST_TEST(powder_ratio == 1.);
    double solid_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::solid);
    BOOST_TEST(solid_ratio == 0.);
    double liquid_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::liquid);
    BOOST_TEST(liquid_ratio == 0.);

    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_TEST(density == 1.);
    double const th_conduc_x = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_x);
    BOOST_TEST(th_conduc_x == 1.);
    double const th_conduc_z = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_z);
    BOOST_TEST(th_conduc_z == 1.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_TEST(liquidus == 100.);
    double const solidus =
        mat_prop.get_cell_value(cell, adamantine::Property::solidus);
    BOOST_TEST(solidus == 50.);
    double const latent_heat =
        mat_prop.get_cell_value(cell, adamantine::Property::latent_heat);
    BOOST_TEST(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::specific_heat);
    BOOST_TEST(specific_heat == 1.);
  }

  // Check the material properties of the liquid
  dealii::types::global_dof_index const n_dofs = dof_handler.n_dofs();
  dealii::LA::distributed::Vector<double, MemorySpaceType> avg_temperature(
      n_dofs);
  dealii::LA::ReadWriteVector<double> rw_vector(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    rw_vector[i] = 100000.;
  avg_temperature.import(rw_vector, dealii::VectorOperation::insert);
  mat_prop.update(dof_handler, avg_temperature);
  for (auto cell : triangulation.active_cell_iterators())
  {
    double powder_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::powder);
    BOOST_TEST(powder_ratio == 0.);
    double solid_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::solid);
    BOOST_TEST(solid_ratio == 0.);
    double liquid_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::liquid);
    BOOST_TEST(liquid_ratio == 1.);

    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_TEST(density == 1.);
    double const th_conduc_x = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_x);
    BOOST_TEST(th_conduc_x == 20.);
    double const th_conduc_z = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_z);
    BOOST_TEST(th_conduc_z == 20.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_TEST(liquidus == 100.);
    double const solidus =
        mat_prop.get_cell_value(cell, adamantine::Property::solidus);
    BOOST_TEST(solidus == 50.);
    double const latent_heat =
        mat_prop.get_cell_value(cell, adamantine::Property::latent_heat);
    BOOST_TEST(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::specific_heat);
    BOOST_TEST(specific_heat == 20.);
  }

  // Check the material properties of the solid
  avg_temperature = 0.;
  mat_prop.update(dof_handler, avg_temperature);
  for (auto cell : triangulation.active_cell_iterators())
  {
    double powder_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::powder);
    BOOST_TEST(powder_ratio == 0.);
    double solid_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::solid);
    BOOST_TEST(solid_ratio == 1.);
    double liquid_ratio = mat_prop.get_state_ratio(
        cell, adamantine::SolidLiquidPowder::State::liquid);
    BOOST_TEST(liquid_ratio == 0.);

    double const density =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::density);
    BOOST_TEST(density == 10.);
    double const th_conduc_x = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_x);
    BOOST_TEST(th_conduc_x == 10.);
    double const th_conduc_z = mat_prop.get_cell_value(
        cell, adamantine::StateProperty::thermal_conductivity_z);
    BOOST_TEST(th_conduc_z == 10.);
    double const liquidus =
        mat_prop.get_cell_value(cell, adamantine::Property::liquidus);
    BOOST_TEST(liquidus == 100.);
    double const solidus =
        mat_prop.get_cell_value(cell, adamantine::Property::solidus);
    BOOST_TEST(solidus == 50.);
    double const latent_heat =
        mat_prop.get_cell_value(cell, adamantine::Property::latent_heat);
    BOOST_TEST(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get_cell_value(cell, adamantine::StateProperty::specific_heat);
    BOOST_TEST(specific_heat == 10.);
  }
}

template <typename MemorySpaceType>
void material_property_table()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);
  auto const &triangulation = geometry.get_triangulation();

  unsigned int n = 0;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
      cell->set_material_id(0);
    else
      cell->set_material_id(1);

    if (n < 15)
      cell->set_user_index(
          static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
    else
      cell->set_user_index(
          static_cast<int>(adamantine::SolidLiquidPowder::State::powder));

    ++n;
  }

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "table");
  database.put("n_materials", 2);
  database.put("material_0.solid.density", "0., 1.");
  database.put("material_0.solid.thermal_conductivity_x", "0., 10.; 10., 100.");
  database.put("material_0.solid.thermal_conductivity_z", "0., 10.; 10., 100.");
  database.put("material_1.solid.density", "0., 1.; 20., 2.; 30., 3.");
  database.put("material_1.solid.thermal_conductivity_x",
               "0., 10.; 10., 100.; 20., 200.");
  database.put("material_1.solid.thermal_conductivity_z",
               "0., 10.; 10., 100.; 20., 200.");
  database.put("material_1.powder.density", "0., 1.; 15., 2.; 30., 3.");
  database.put("material_1.powder.thermal_conductivity_x",
               "0., 10.; 10., 100.; 18., 200.");
  database.put("material_1.powder.thermal_conductivity_z",
               "0., 10.; 10., 100.; 18., 200.");
  adamantine::MaterialProperty<2, 0, adamantine::SolidLiquidPowder,
                               MemorySpaceType>
      mat_prop(communicator, triangulation, database);
  // Evaluate the material property at the given temperature
  dealii::FE_Q<2> fe(4);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  dealii::LA::ReadWriteVector<double> rw_vector(
      dof_handler.locally_owned_dofs());
  for (unsigned int i = 0; i < rw_vector.locally_owned_size(); ++i)
    rw_vector.local_element(i) = 15;
  temperature.import(rw_vector, dealii::VectorOperation::insert);
  mat_prop.update(dof_handler, temperature);

  n = 0;
  double constexpr tolerance = 1e-10;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
    {
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::density) == 1.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_x) ==
                     100.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_z) ==
                     100.,
                 tt::tolerance(tolerance));
    }
    else if (n < 15)
    {
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::density) == 1.75,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_x) ==
                     150.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_z) ==
                     150.,
                 tt::tolerance(tolerance));
    }
    else
    {
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::density) == 2.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_x) ==
                     162.5,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_z) ==
                     162.5,
                 tt::tolerance(tolerance));
    }
    ++n;
  }
}

template <typename MemorySpaceType>
void material_property_polynomials()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);
  auto const &triangulation = geometry.get_triangulation();

  unsigned int n = 0;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
      cell->set_material_id(0);
    else
      cell->set_material_id(1);

    if (n < 15)
      cell->set_user_index(
          static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
    else
      cell->set_user_index(
          static_cast<int>(adamantine::SolidLiquidPowder::State::powder));

    ++n;
  }

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("property_format", "polynomial");
  database.put("n_materials", 2);
  database.put("material_0.solid.density", "0., 1.");
  database.put("material_0.solid.thermal_conductivity_x", "0., 1., 2.");
  database.put("material_0.solid.thermal_conductivity_z", "0., 1., 2.");
  database.put("material_1.solid.density", " 1., 2., 3.");
  database.put("material_1.solid.thermal_conductivity_x",
               "1.,  100., 20., 200.");
  database.put("material_1.solid.thermal_conductivity_z",
               "1.,  100., 20., 200.");
  database.put("material_1.powder.density", "15., 2., 3.");
  database.put("material_1.powder.thermal_conductivity_x", " 10., 18., 200.");
  database.put("material_1.powder.thermal_conductivity_z", " 10., 18., 200.");
  adamantine::MaterialProperty<2, 4, adamantine::SolidLiquidPowder,
                               MemorySpaceType>
      mat_prop(communicator, triangulation, database);
  // Evaluate the material property at the given temperature
  dealii::FE_Q<2> fe(4);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
      temperature(dof_handler.locally_owned_dofs(), communicator);
  dealii::LA::ReadWriteVector<double> rw_vector(
      dof_handler.locally_owned_dofs());
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
    rw_vector.local_element(i) = 15;
  temperature.import(rw_vector, dealii::VectorOperation::insert);
  mat_prop.update(dof_handler, temperature);

  n = 0;
  double constexpr tolerance = 1e-10;
  for (auto cell : triangulation.active_cell_iterators())
  {
    if (n < 10)
    {
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::density) == 15.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_x) ==
                     465.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_z) ==
                     465.,
                 tt::tolerance(tolerance));
    }
    else if (n < 15)
    {
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::density) == 706.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_x) ==
                     681001.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_z) ==
                     681001.,
                 tt::tolerance(tolerance));
    }
    else
    {
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::density) == 720.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_x) ==
                     45280.,
                 tt::tolerance(tolerance));
      BOOST_TEST(mat_prop.get_cell_value(
                     cell, adamantine::StateProperty::thermal_conductivity_z) ==
                     45280.,
                 tt::tolerance(tolerance));
    }
    ++n;
  }
}
