/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE ThermalPhysics

// clang-format off
#include "main.cc"

#include "test_thermal_physics.hh"
// clang-format on

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(thermal_2d_explicit_host)
{
  boost::property_tree::ptree database;
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.type", "electron_beam");
  database.put("sources.beam_0.scan_path_file_format", "segment");

  thermal_2d<dealii::MemorySpace::Host>(database, 0.05);
}

BOOST_AUTO_TEST_CASE(thermal_2d_manufactured_solution_host)
{
  thermal_2d_manufactured_solution<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(initial_temperature_host)
{
  initial_temperature<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(energy_conservation_host)
{
  energy_conservation<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(radiation_bcs_host)
{
  radiation_bcs<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(convection_bcs_host, *utf::tolerance(1e-6))
{
  convection_bcs<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(reference_temperature_host)
{
  reference_temperature<dealii::MemorySpace::Host>();
}
