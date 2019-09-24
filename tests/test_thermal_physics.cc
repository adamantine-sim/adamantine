/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ThermalPhysics

#include "main.cc"

#include "Geometry.hh"
#include "ThermalPhysics.hh"

void thermal_2d(boost::property_tree::ptree &database, double time_step)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 12e-3);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6e-3);
  geometry_database.put("height_divisions", 5);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);
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
  database.put("sources.beam_0.energy_conversion_efficiency", 0.1);
  database.put("sources.beam_0.control_efficiency", 1.0);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.abscissa", "t");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, double, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.reinit();

  dealii::LA::distributed::Vector<double> solution;
  physics.initialize_dof_vector(solution);
  std::vector<adamantine::Timer> timers(6);
  double time = 0;
  while (time < 0.1)
    time = physics.evolve_one_time_step(time, time_step, solution, timers);

  double const tolerance = 1e-3;
  BOOST_CHECK(time == 0.1);
  BOOST_CHECK_CLOSE(solution.l2_norm(), 0.291705, tolerance);

  physics.initialize_dof_vector(1000., solution);
  BOOST_CHECK(solution.l1_norm() == 1000. * solution.size());
}

BOOST_AUTO_TEST_CASE(thermal_2d_explicit)
{
  boost::property_tree::ptree database;
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  thermal_2d(database, 0.05);
}

BOOST_AUTO_TEST_CASE(thermal_2d_implicit)
{
  boost::property_tree::ptree database;
  // Time-stepping database
  database.put("time_stepping.method", "backward_euler");
  database.put("time_stepping.max_iteration", 100);
  database.put("time_stepping.tolerance", 1e-6);
  database.put("time_stepping.n_tmp_vectors", 100);

  thermal_2d(database, 0.025);
}

BOOST_AUTO_TEST_CASE(thermal_2d_manufactured_solution)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

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
  physics.setup_dofs();
  physics.reinit();

  dealii::LA::distributed::Vector<double> solution;
  std::vector<adamantine::Timer> timers(6);
  physics.initialize_dof_vector(solution);
  double time = physics.evolve_one_time_step(0., 0.1, solution, timers);

  double const tolerance = 1e-5;
  BOOST_CHECK(time == 0.1);
  for (unsigned int i = 0; i < solution.size(); ++i)
    BOOST_CHECK_CLOSE(solution[i], 0.1, tolerance);
}

BOOST_AUTO_TEST_CASE(initial_temperature)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 12e-3);
  geometry_database.put("length_divisions", 1);
  geometry_database.put("height", 6e-3);
  geometry_database.put("height_divisions", 1);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  boost::property_tree::ptree database;
  // Material property
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 1.);
  database.put("materials.material_0.powder.density", 10.);
  database.put("materials.material_0.liquid.density", 1.);
  database.put("materials.material_0.solid.specific_heat", 1.);
  database.put("materials.material_0.powder.specific_heat", 2.);
  database.put("materials.material_0.liquid.specific_heat", 1.);
  database.put("materials.material_0.solid.thermal_conductivity", 1.);
  database.put("materials.material_0.powder.thermal_conductivity", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity", 1.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.energy_conversion_efficiency", 0.1);
  database.put("sources.beam_0.control_efficiency", 1.0);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.abscissa", "t");
  // Time-stepping database
  database.put("time_stepping.method", "rk_fourth_order");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, double, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.reinit();

  dealii::LA::distributed::Vector<double> solution;
  physics.initialize_dof_vector(1000., solution);
  BOOST_CHECK(solution.l1_norm() == 20000. * solution.size());
}
