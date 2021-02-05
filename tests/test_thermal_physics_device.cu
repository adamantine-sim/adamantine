/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
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
  thermal_2d<dealii::MemorySpace::CUDA>(database, 0.05);
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

  thermal_2d<dealii::MemorySpace::CUDA>(database, 0.025);
}

BOOST_AUTO_TEST_CASE(thermal_2d_manufactured_solution_device)
{
  thermal_2d_manufactured_solution<dealii::MemorySpace::CUDA>();
}

BOOST_AUTO_TEST_CASE(initial_temperature_device)
{
  initial_temperature<dealii::MemorySpace::CUDA>();
}

BOOST_AUTO_TEST_CASE(energy_conservation_device)
{
  energy_conservation<dealii::MemorySpace::CUDA>();
}
