/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_3D

#include "../application/adamantine.hh"

#include <boost/property_tree/info_parser.hpp>

#include <filesystem>
#include <fstream>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(integration_3D, *utf::tolerance(0.1))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "demo_316_short_anisotropic.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto [temperature, displacement] =
      run<3, 4, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
          communicator, database, timers);

  int num_ranks = 0;
  MPI_Comm_size(communicator, &num_ranks);

  // Limits for a weak non-pointwise check
  double max_expected = 500.0;
  double min_expected = 285.0;

  if (num_ranks == 1)
  {
    std::ifstream gold_file("integration_3d_gold.txt");

    for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
    {
      BOOST_CHECK(temperature.local_element(i) > min_expected);
      BOOST_CHECK(temperature.local_element(i) < max_expected);

      double gold_value = -1.;
      gold_file >> gold_value;
      BOOST_TEST(temperature.local_element(i) == gold_value);
    }
  }
  else if (num_ranks == 2)
  {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);

    // To write the gold file
    /*
    std::ofstream gold_file_writer("integration_3d_gold_" +
                                   std::to_string(rank) + ".txt");
    for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
      gold_file_writer << temperature.local_element(i) << " ";

    gold_file_writer.close();
    */

    std::ifstream gold_file("integration_3d_gold_" + std::to_string(rank) +
                            ".txt");

    for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
    {
      BOOST_CHECK(temperature.local_element(i) > min_expected);
      BOOST_CHECK(temperature.local_element(i) < max_expected);

      double gold_value = -1.;
      gold_file >> gold_value;
      BOOST_TEST(temperature.local_element(i) == gold_value);
    }
  }
  else
  {
    // Only the weaker non-pointwise check for more than 2 cores
    for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
    {
      BOOST_CHECK(temperature.local_element(i) > min_expected);
      BOOST_CHECK(temperature.local_element(i) < max_expected);
    }
  }
}

BOOST_AUTO_TEST_CASE(integration_3D_checkpoint_restart)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const input_filename = "demo_316_short_anisotropic.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(input_filename) == true,
                           "The file " + input_filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(input_filename, database);
  std::string const checkpoint_filename = "test_checkpoint";

  // First run with checkpoint of the solution halfway through the solution
  database.put("checkpoint.filename_prefix", checkpoint_filename);
  database.put("checkpoint.overwrite_files", true);
  database.put("checkpoint.time_steps_between_checkpoint", 80);
  auto [temperature_1, displacement_1] =
      run<3, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);
  // Restart of the simulation
  database.put("restart.filename_prefix", checkpoint_filename);
  auto [temperature_2, displacement_2] =
      run<3, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  // Compare the temperatures. When using more than one processor, the
  // partitioning is different and so the distribution of the dofs are
  // different. This makes it very difficult to compare the temperatures, so
  // when using more than one processor we only compare the L2 norm.
  if (dealii::Utilities::MPI::n_mpi_processes(communicator) == 1)
  {
    for (unsigned int i = 0; i < temperature_1.size(); ++i)
      BOOST_TEST(temperature_1[i] == temperature_2[i]);
  }
  else
  {
    BOOST_TEST(temperature_1.l2_norm() == temperature_2.l2_norm());
  }

  // Remove the files created during the test
  std::filesystem::remove(checkpoint_filename);
  std::filesystem::remove(checkpoint_filename + "_fixed.data");
  std::filesystem::remove(checkpoint_filename + ".info");
  std::filesystem::remove(checkpoint_filename + "_variable.data");
  std::filesystem::remove(checkpoint_filename + "_time.txt");
}

BOOST_AUTO_TEST_CASE(hourglass, *utf::tolerance(0.1))
{
  // The HourGlass test would crash in parallel. This test makes sure it
  // doesn't happen anymore. We do not check the result, we just make sure that
  // the code runs to completion.

  MPI_Comm communicator = MPI_COMM_WORLD;

  if (dealii::Utilities::MPI::n_mpi_processes(communicator) == 2)
  {

    std::vector<adamantine::Timer> timers;
    initialize_timers(communicator, timers);

    // Read the input.
    std::string const filename = "HourGlass_AOP.info";
    adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                             "The file " + filename + " does not exist.");
    boost::property_tree::ptree database;
    boost::property_tree::info_parser::read_info(filename, database);

    run<3, 1, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
        communicator, database, timers);
  }
}
