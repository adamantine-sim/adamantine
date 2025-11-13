/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "MaterialStates.hh"
#define BOOST_TEST_MODULE Integration_2D

#include "../application/adamantine.hh"

#include <boost/property_tree/info_parser.hpp>

#include <filesystem>
#include <fstream>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(integration_2D, *utf::tolerance(0.1))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "integration_2d.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto [temperature, displacement] =
      run<2, 1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  std::ifstream gold_file("integration_2d_gold.txt");
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
  {
    double gold_value = -1.;
    gold_file >> gold_value;
    BOOST_TEST(temperature.local_element(i) == gold_value);
  }
}

BOOST_AUTO_TEST_CASE(integration_2D_ensemble, *utf::tolerance(0.1))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "integration_2d_ensemble.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto result_ensemble =
      run_ensemble<2, -1, 3, adamantine::SolidLiquidPowder,
                   dealii::MemorySpace::Host>(communicator, database, timers);

  for (auto &result_member : result_ensemble)
  {
    std::ifstream gold_file("integration_2d_gold.txt");
    for (unsigned int i = 0; i < result_member.block(0).locally_owned_size();
         ++i)
    {
      double gold_value = -1.;
      gold_file >> gold_value;
      BOOST_TEST(result_member.block(0).local_element(i) == gold_value);
    }
  }
}

// This is the same test as integration_2D but using different units
BOOST_AUTO_TEST_CASE(integration_2D_units, *utf::tolerance(0.1))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "integration_2d_units.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto [temperature, displacement] =
      run<2, 1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  std::ifstream gold_file("integration_2d_gold.txt");
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
  {
    double gold_value = -1.;
    gold_file >> gold_value;
    BOOST_TEST(temperature.local_element(i) == gold_value);
  }
}
