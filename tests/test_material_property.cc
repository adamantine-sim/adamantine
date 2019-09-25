/* Copyright (c) 2016, the adamantine authors.
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
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);

  dealii::Triangulation<2> tria;
  dealii::GridGenerator::hyper_cube(tria);
  for (auto cell : tria.active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
  }

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("n_materials", 1);
  database.put("material_0.solid.density", 1.);
  database.put("material_0.solid.thermal_conductivity", 10.);
  database.put("material_0.powder.conductivity", 10.);
  database.put("material_0.liquid", "");
  database.put("material_0.liquidus", "100");
  adamantine::MaterialProperty<2> mat_prop(
      communicator, geometry.get_triangulation(), database);
  dealii::LA::distributed::Vector<double> dummy;

  // This check only one cell
  for (auto cell : tria.active_cell_iterators())
  {
    double const density =
        mat_prop.get(cell, adamantine::Property::density, dummy);
    BOOST_CHECK(density == 1.);
    double const th_conduc =
        mat_prop.get(cell, adamantine::Property::thermal_conductivity, dummy);
    BOOST_CHECK(th_conduc == 10.);
    double const liquidus =
        mat_prop.get(cell, adamantine::Property::liquidus, dummy);
    BOOST_CHECK(liquidus == 100.);
  }
}

BOOST_AUTO_TEST_CASE(ratios)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);

  dealii::parallel::distributed::Triangulation<2> &tria =
      geometry.get_triangulation();
  for (auto cell : tria.active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(static_cast<int>(adamantine::MaterialState::powder));
  }
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  dealii::LA::distributed::Vector<double> dummy;

  // Create the MaterialProperty
  boost::property_tree::ptree database;
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
  database.put("material_0.latent_heat", 1000);
  adamantine::MaterialProperty<2> mat_prop(
      communicator, geometry.get_triangulation(), database);

  // Check the material properties of the powder
  for (auto cell : tria.active_cell_iterators())
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
        mat_prop.get(cell, adamantine::Property::density, dummy);
    BOOST_CHECK(density == 1.);
    double const th_conduc =
        mat_prop.get(cell, adamantine::Property::thermal_conductivity, dummy);
    BOOST_CHECK(th_conduc == 1.);
    double const liquidus =
        mat_prop.get(cell, adamantine::Property::liquidus, dummy);
    BOOST_CHECK(liquidus == 100.);
    double const solidus =
        mat_prop.get(cell, adamantine::Property::solidus, dummy);
    BOOST_CHECK(solidus == 50.);
    double const latent_heat =
        mat_prop.get(cell, adamantine::Property::latent_heat, dummy);
    BOOST_CHECK(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get(cell, adamantine::Property::specific_heat, dummy);
    BOOST_CHECK(specific_heat == 1.);

    double const tolerance = 1e-10;
    double liquid_beta = mat_prop.get_liquid_beta(cell);
    BOOST_CHECK_CLOSE(liquid_beta, -200., tolerance);
    double mushy_alpha = mat_prop.get_mushy_alpha(cell);
    BOOST_CHECK_CLOSE(mushy_alpha, 0.05, tolerance);
    double mushy_beta = mat_prop.get_mushy_beta(cell);
    BOOST_CHECK_CLOSE(mushy_beta, -200., tolerance);
  }

  // Check the material properties of the liquid
  dealii::types::global_dof_index const n_dofs = dof_handler.n_dofs();
  dealii::LA::distributed::Vector<double> avg_enthalpy(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    avg_enthalpy[i] = 100000.;
  mat_prop.update_state(dof_handler, avg_enthalpy);
  for (auto cell : tria.active_cell_iterators())
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
        mat_prop.get(cell, adamantine::Property::density, dummy);
    BOOST_CHECK(density == 1.);
    double const th_conduc =
        mat_prop.get(cell, adamantine::Property::thermal_conductivity, dummy);
    BOOST_CHECK(th_conduc == 20.);
    double const liquidus =
        mat_prop.get(cell, adamantine::Property::liquidus, dummy);
    BOOST_CHECK(liquidus == 100.);
    double const solidus =
        mat_prop.get(cell, adamantine::Property::solidus, dummy);
    BOOST_CHECK(solidus == 50.);
    double const latent_heat =
        mat_prop.get(cell, adamantine::Property::latent_heat, dummy);
    BOOST_CHECK(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get(cell, adamantine::Property::specific_heat, dummy);
    BOOST_CHECK(specific_heat == 20.);
  }

  // Check the material properties of the solid
  avg_enthalpy = 0.;
  mat_prop.update_state(dof_handler, avg_enthalpy);
  for (auto cell : tria.active_cell_iterators())
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
        mat_prop.get(cell, adamantine::Property::density, dummy);
    BOOST_CHECK(density == 10.);
    double const th_conduc =
        mat_prop.get(cell, adamantine::Property::thermal_conductivity, dummy);
    BOOST_CHECK(th_conduc == 10.);
    double const liquidus =
        mat_prop.get(cell, adamantine::Property::liquidus, dummy);
    BOOST_CHECK(liquidus == 100.);
    double const solidus =
        mat_prop.get(cell, adamantine::Property::solidus, dummy);
    BOOST_CHECK(solidus == 50.);
    double const latent_heat =
        mat_prop.get(cell, adamantine::Property::latent_heat, dummy);
    BOOST_CHECK(latent_heat == 1000.);
    double const specific_heat =
        mat_prop.get(cell, adamantine::Property::specific_heat, dummy);
    BOOST_CHECK(specific_heat == 10.);
  }
}
