/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ThermalPhysics

#include "main.cc"

#include "ThermalPhysics.hh"

BOOST_AUTO_TEST_CASE(thermal_2d)
{
  boost::mpi::communicator communicator;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 12e-3);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6e-3);
  geometry_database.put("height_divisions", 5);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);

  boost::property_tree::ptree database;
  // Material property
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 1.);
  database.put("materials.material_0.powder.density", 1.);
  database.put("materials.material_0.liquid.density", 1.);
  database.put("materials.material_0.solid.specific_heat", 1.);
  database.put("materials.material_0.powder.specific_heat", 1.);
  database.put("materials.material_0.liquid.specific_heat", 1.);
  database.put("materials.material_0.solid.thermal_conductivity", 10.);
  database.put("materials.material_0.powder.thermal_conductivity", 10.);
  database.put("materials.material_0.liquid.thermal_conductivity", 10.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 0.1);
  database.put("sources.beam_0.energy_conversion_efficiency", 0.1);
  database.put("sources.beam_0.control_efficiency", 1.0);
  database.put("sources.beam_0.diameter", 1.0);
  database.put("sources.beam_0.max_power", 10.);
  database.put("sources.beam_0.abscissa", "t");
  // Time-stepping database
  database.put("time_stepping.method", "rk_fourth_order");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, double, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.reinit();

  dealii::LA::distributed::Vector<double> solution;
  physics.initialize_dof_vector(solution);
  double time = physics.evolve_one_time_step(0., 0.1, solution);

  double const tolerance = 1e-5;
  BOOST_CHECK(time == 0.1);
  BOOST_CHECK_CLOSE(solution.l2_norm(), 6.4802244e18, tolerance);
}

BOOST_AUTO_TEST_CASE(thermal_2d_manufactured_solution)
{
  boost::mpi::communicator communicator;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 1e3);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6e3);
  geometry_database.put("height_divisions", 5);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);

  boost::property_tree::ptree database;
  // Material property
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 1.);
  database.put("materials.material_0.powder.density", 1.);
  database.put("materials.material_0.liquid.density", 1.);
  database.put("materials.material_0.solid.specific_heat", 1.);
  database.put("materials.material_0.powder.specific_heat", 1.);
  database.put("materials.material_0.liquid.specific_heat", 1.);
  database.put("materials.material_0.solid.thermal_conductivity", 1.);
  database.put("materials.material_0.powder.thermal_conductivity", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity", 1.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.energy_conversion_efficiency",
               0.1 / 0.29317423955177113);
  database.put("sources.beam_0.control_efficiency", 1.0);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.abscissa", "1");
  // Time-stepping database
  database.put("time_stepping.method", "rk_fourth_order");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, double, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.reinit();

  dealii::LA::distributed::Vector<double> solution;
  physics.initialize_dof_vector(solution);
  double time = physics.evolve_one_time_step(0., 0.1, solution);

  double const tolerance = 1e-5;
  BOOST_CHECK(time == 0.1);
  for (unsigned int i = 0; i < solution.size(); ++i)
    BOOST_CHECK_CLOSE(solution[i], 0.1, tolerance);
}
