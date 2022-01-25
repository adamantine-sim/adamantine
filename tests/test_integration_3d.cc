/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_3D

#include "../application/adamantine.hh"

#include <fstream>

#include "main.cc"

BOOST_AUTO_TEST_CASE(integration_3D_data_assimilation)
{

  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "bare_plate_L_da.info";
  adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  // Run the simulation
  auto result = run_ensemble<3, dealii::MemorySpace::Host>(communicator,
                                                           database, timers);

  // Three ensemble members expected
  BOOST_CHECK(result.size() == 3);

  // Get the average minimum value for each ensemble member, which is very close
  // to the initial temperature
  double sum = 0.0;
  for (unsigned int member = 0; member < result.size(); ++member)
  {
    double min_val = std::numeric_limits<double>::max();
    for (unsigned int i = 0;
         i < result.at(member).block(0).locally_owned_size(); ++i)
    {
      if (result.at(member).block(0).local_element(i) < min_val)
        min_val = result.at(member).block(0).local_element(i);
    }
    sum += min_val;
  }
  double average_minimum_value = sum / result.size();

  // Based on the experimental data, the expected temperature is ~200.0
  BOOST_CHECK((average_minimum_value >= 200.0) &&
              (average_minimum_value < 300.0));
}

BOOST_AUTO_TEST_CASE(integration_3D)
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

  auto result =
      run<3, dealii::MemorySpace::Host>(communicator, database, timers);

  std::ifstream gold_file("integration_3d_gold.txt");
  double const tolerance = 0.1;
  for (unsigned int i = 0; i < result.locally_owned_size(); ++i)
  {
    double gold_value = -1.;
    gold_file >> gold_value;
    BOOST_CHECK_CLOSE(result.local_element(i), gold_value, tolerance);
  }
}
