/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE ThermalPhysicsDevice

// clang-format off
#include "main.cc"

#include "test_thermal_physics.hh"
// clang-format on

BOOST_AUTO_TEST_CASE(thermal_2d_explicit_device)
{
  boost::property_tree::ptree database;
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.type", "electron_beam");
  database.put("sources.beam_0.scan_path_file_format", "segment");

  thermal_2d<dealii::MemorySpace::Default>(database, 0.05);
}

BOOST_AUTO_TEST_CASE(thermal_2d_implicit_device)
{
  boost::property_tree::ptree database;
  // Time-stepping database
  database.put("time_stepping.method", "backward_euler");
  database.put("time_stepping.max_iteration", 100);
  database.put("time_stepping.tolerance", 1e-6);
  database.put("time_stepping.n_tmp_vectors", 100);
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.type", "electron_beam");
  database.put("sources.beam_0.scan_path_file_format", "segment");

  thermal_2d<dealii::MemorySpace::Default>(database, 0.025);
}

BOOST_AUTO_TEST_CASE(thermal_2d_manufactured_solution_device)
{
  thermal_2d_manufactured_solution<dealii::MemorySpace::Default>();
}

BOOST_AUTO_TEST_CASE(initial_temperature_device)
{
  initial_temperature<dealii::MemorySpace::Default>();
}

BOOST_AUTO_TEST_CASE(energy_conservation_device)
{
  energy_conservation<dealii::MemorySpace::Default>();
}
