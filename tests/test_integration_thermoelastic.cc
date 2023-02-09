/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_Thermoelastic

#include "../application/adamantine.hh"

#include <fstream>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(integration_thermoelastic, *utf::tolerance(1.0e-5))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "thermoelastic_bare_plate.info";
  adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);
  database.put("materials.material_0.solid.thermal_expansion_coef", 17.2e-3);

  auto [temperature, displacement] =
      run<3, dealii::MemorySpace::Host>(communicator, database, timers);

  // For now doing a simple regression test. Without a dof handler, it's hard to
  // do something more meaningful with the vector.

  // To generate a new gold solution
  // std::cout << "dis l2:" << displacement.l2_norm() << std::endl;

  BOOST_TEST(displacement.l2_norm() == 0.21537566016824577);
}

BOOST_AUTO_TEST_CASE(integration_thermoelastic_add_material,
                     *utf::tolerance(1.0e-5))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "thermoelastic_bare_plate.info";
  adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);
  database.put("geometry.height", 8.0e-3);
  database.put("geometry.height_division", 4);
  database.put("sources.beam_0.scan_path_file",
               "thermoelastic_bare_plate_add_material_scan_path.txt");
  database.put("materials.material_0.solid.thermal_expansion_coef", 17.2e-3);

  auto [temperature, displacement] =
      run<3, dealii::MemorySpace::Host>(communicator, database, timers);

  // For now doing a simple regression test. Without a dof handler, it's hard to
  // do something more meaningful with the vector.

  BOOST_TEST(displacement.l2_norm() == 0.21134574383060864);
}
