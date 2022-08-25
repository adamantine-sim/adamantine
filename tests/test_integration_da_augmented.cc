/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_Data_Assimilation_Augmented

#include "../application/adamantine.hh"

#include <fstream>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(integration_3D_data_assimilation_augmented,
                     *utf::tolerance(5.))
{

  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "bare_plate_L_da_augmented.info";
  adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  // Run the simulation
  auto result = run_ensemble<3, dealii::MemorySpace::Host>(communicator,
                                                           database, timers);

  // Three ensemble members expected
  BOOST_TEST(result.size() == 3);

  // Get the average absorption value for each ensemble member
  double sum = 0.0;
  for (unsigned int member = 0; member < result.size(); ++member)
  {
    sum += result.at(member).block(1).local_element(0);
  }
  double average_value = sum / result.size();

  // Based on the reference solution, the expected absorption efficiency is 0.3
  double gold_solution = 0.3;
  BOOST_TEST(average_value == gold_solution);
}

BOOST_AUTO_TEST_CASE(integration_3D, *utf::tolerance(0.1))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "demo_316_short_anisotropic.info";
  adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto [temperature, displacement] =
      run<3, dealii::MemorySpace::Host>(communicator, database, timers);

  std::ifstream gold_file("integration_3d_gold.txt");
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
  {
    double gold_value = -1.;
    gold_file >> gold_value;
    BOOST_TEST(temperature.local_element(i) == gold_value);
  }
}
