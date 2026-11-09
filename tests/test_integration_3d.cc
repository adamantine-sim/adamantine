/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE Integration_3D

#include "../applications/adamantine.hh"

#include <boost/property_tree/info_parser.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "main.cc"

namespace utf = boost::unit_test;

namespace
{
// Keep the original cell sizes, FE degrees, scan paths, and time steps, but
// remove the large part of the substrate that is far from the heat source.
void configure_small_domain(boost::property_tree::ptree &database,
                            bool const five_axis = false)
{
  unsigned int const divisions =
      database.get<unsigned int>("geometry.length_divisions");
  database.put("geometry.length_origin", five_axis ? 4.6e-3 : 1.e-3);
  database.put("geometry.width_origin", five_axis ? 2.e-3 : 1.e-3);
  database.put("geometry.length", five_axis ? 1.8e-3 : 3.e-3);
  database.put("geometry.width", five_axis ? 6.e-3 : 4.e-3);
  database.put("geometry.length_divisions", five_axis ? 9 : divisions * 3 / 10);
  database.put("geometry.width_divisions", five_axis ? 30 : divisions * 4 / 10);
}

void test(std::string const &filename, std::string const &gold_filename_prefix,
          double const min_expected, double const max_expected,
          bool const five_axis = false)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  unsigned int const num_ranks =
      dealii::Utilities::MPI::n_mpi_processes(communicator);
  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);
  configure_small_domain(database, five_axis);
  if (five_axis)
    database.put("post_processor.time_steps_between_output", 4250);
  database.put("post_processor.filename_prefix",
               gold_filename_prefix + "_" + std::to_string(num_ranks));

  auto [temperature, displacement] =
      run<3, -1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  BOOST_REQUIRE_GT(temperature.size(), 0);
  // Initially only the bottom three cell layers contain material. There is
  // no AMR in these cases, so more DoFs prove that deposition activated cells.
  unsigned int const degree =
      database.get<unsigned int>("discretization.thermal.fe_degree");
  unsigned int const initial_dofs =
      (degree * database.get<unsigned int>("geometry.length_divisions") + 1) *
      (degree * database.get<unsigned int>("geometry.width_divisions") + 1) *
      (degree * 3 + 1);
  BOOST_CHECK_GT(temperature.size(), initial_dofs);
  double local_max = 0.;
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
  {
    double const value = temperature.local_element(i);
    BOOST_CHECK(std::isfinite(value));
    BOOST_CHECK_GT(value, min_expected);
    BOOST_CHECK_LT(value, max_expected);
    local_max = std::max(local_max, value);
  }
  // A uniform initial-temperature field must not pass the regression check.
  BOOST_CHECK_GT(dealii::Utilities::MPI::max(local_max, communicator), 301.);

  if (num_ranks <= 2)
  {
    std::string const gold_filename =
        gold_filename_prefix +
        (num_ranks == 2 ? "_" + std::to_string(rank) : "") + ".txt";
    std::ifstream gold_file(gold_filename);
    BOOST_REQUIRE_MESSAGE(gold_file.is_open(), "Cannot open " + gold_filename);
    for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
    {
      double gold_value = 0.;
      BOOST_REQUIRE_MESSAGE(static_cast<bool>(gold_file >> gold_value),
                            "Missing reference value in " + gold_filename);
      BOOST_TEST(temperature.local_element(i) == gold_value);
    }
    double extra_value = 0.;
    BOOST_CHECK_MESSAGE(!(gold_file >> extra_value),
                        "Extra reference values in " + gold_filename);
    BOOST_CHECK(gold_file.eof());
  }
}
} // namespace

BOOST_AUTO_TEST_CASE(integration_3D, *utf::tolerance(0.1))
{
  test("demo_316_short_anisotropic.info", "integration_3d_small_gold", 285.0,
       500.0);
}

BOOST_AUTO_TEST_CASE(integration_3D_short, *utf::tolerance(0.1))
{
  test("demo_316_short.info", "integration_3d_gold_short", 285.0, 580.0);
}

BOOST_AUTO_TEST_CASE(integration_3D_five_axis, *utf::tolerance(0.1))
{
  test("five_axis.info", "integration_3d_gold_five_axis", 299.9, 5415.0, true);
}

BOOST_AUTO_TEST_CASE(integration_3D_checkpoint_restart, *utf::tolerance(1e-10))
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
  configure_small_domain(database);
  database.put("post_processor.filename_prefix",
               "integration_3d_restart_" +
                   std::to_string(
                       dealii::Utilities::MPI::n_mpi_processes(communicator)));
  std::string const checkpoint_filename =
      "test_checkpoint_" +
      std::to_string(dealii::Utilities::MPI::n_mpi_processes(communicator));

  // Save at step 80 of 100, leaving thermal evolution after the restart.
  database.put("checkpoint.filename_prefix", checkpoint_filename);
  database.put("checkpoint.overwrite_files", true);
  database.put("checkpoint.time_steps_between_checkpoint", 80);
  auto [temperature_1, displacement_1] =
      run<3, -1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);
  // Restart of the simulation
  database.put("restart.filename_prefix", checkpoint_filename);
  auto [temperature_2, displacement_2] =
      run<3, -1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  BOOST_REQUIRE_GT(temperature_1.size(), 0);
  BOOST_REQUIRE_EQUAL(temperature_1.size(), temperature_2.size());

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
    // Preserve the complete distributed-deposition crash reproducer, but avoid
    // writing a large visualization every ten time steps.
    database.put("post_processor.time_steps_between_output", 1000);
    database.put("post_processor.filename_prefix", "integration_hourglass");

    run<3, 1, 1, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
        communicator, database, timers);
  }
}
