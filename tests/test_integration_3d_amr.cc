/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "MaterialStates.hh"
#define BOOST_TEST_MODULE Integration_3D_AMR

#include "../application/adamantine.hh"

#include <boost/property_tree/info_parser.hpp>

#include <filesystem>
#include <fstream>
#include <limits>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(integration_3D_amr, *utf::tolerance(0.1))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "amr_test.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto [temperature, displacement] =
      run<3, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  double min_val = std::numeric_limits<double>::max();
  double max_val = std::numeric_limits<double>::min();
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
  {
    if (temperature.local_element(i) < min_val)
      min_val = temperature.local_element(i);

    if (temperature.local_element(i) > max_val)
      max_val = temperature.local_element(i);
  }

  double global_max =
      dealii::Utilities::MPI::max(max_val, temperature.get_mpi_communicator());
  double global_min =
      dealii::Utilities::MPI::min(min_val, temperature.get_mpi_communicator());

  double expected_max = 329.5;
  double expected_min = 296.1;

  BOOST_TEST(expected_max == global_max);
  BOOST_TEST(expected_min == global_min);
}

BOOST_AUTO_TEST_CASE(integration_3D_amr_refine_coarsen, *utf::tolerance(0.1))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "demo_316_short_amr.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto [temperature, displacement] =
      run<3, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  auto global_size = temperature.size();
  double l1_norm = temperature.l1_norm();

  // Check for the expected number of degrees of freedom
  BOOST_TEST(global_size == 13005);

  // Check that AMR hasn't caused any NaNs
  // NOTE: This check is only relevant in some cases. In debug mode the test
  // should fail earlier through an assert. In release model with --ffast-math
  // all NaN checks (including isnan and isfinite) are skipped.
  BOOST_TEST(std::isfinite(l1_norm));
}
