/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "MaterialStates.hh"
#define BOOST_TEST_MODULE Integration_Thermoelastic

#include "../applications/adamantine.hh"

#include <boost/property_tree/info_parser.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "main.cc"

namespace utf = boost::unit_test;

namespace
{
void configure_short_run(boost::property_tree::ptree &database,
                         std::string const &name)
{
  // The powered scan ends at approximately 0.12 s. Keep the full scan,
  // subsequent cooling, and refinement at steps 1, 100, 200, and 300.
  database.put("time_stepping.duration", 0.3);
  database.put("post_processor.time_steps_between_output", 300);
  database.put("post_processor.filename_prefix",
               name + "_" +
                   std::to_string(dealii::Utilities::MPI::n_mpi_processes(
                       MPI_COMM_WORLD)));
  database.put("materials.material_0.solid.thermal_expansion_coef", 17.2e-3);
}

template <typename VectorType>
void check_solution(VectorType const &temperature,
                    VectorType const &displacement)
{
  BOOST_REQUIRE_GT(temperature.size(), 0);
  BOOST_REQUIRE_GT(displacement.size(), 0);
  BOOST_CHECK(std::isfinite(temperature.l2_norm()));
  BOOST_CHECK(std::isfinite(displacement.l2_norm()));
  BOOST_CHECK_GT(temperature.linfty_norm(), 301.);
  BOOST_CHECK_GT(displacement.l2_norm(), 0.);
}
} // namespace

BOOST_AUTO_TEST_CASE(integration_thermoelastic, *utf::tolerance(1.0e-5))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "thermoelastic_bare_plate.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);
  configure_short_run(database, "integration_thermoelastic");

  auto [temperature, displacement] =
      run<3, -1, 3, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  // For now doing a simple regression test. Without a dof handler, it's hard to
  // do something more meaningful with the vector.

  // To generate a new gold solution
  // std::cout << "dis l2:" << displacement.l2_norm() << std::endl;

  check_solution(temperature, displacement);
  BOOST_TEST(displacement.l2_norm() == 0.62889786271626713);
}

BOOST_AUTO_TEST_CASE(integration_thermoelastic_add_material,
                     *utf::tolerance(1.0e-5))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "thermoelastic_bare_plate.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);
  database.put("geometry.height", 8.0e-3);
  // Keep the original three vertical divisions in the taller domain.
  database.put("geometry.height_divisions", 3);
  database.put("sources.beam_0.scan_path_file",
               "thermoelastic_bare_plate_add_material_scan_path.txt");
  configure_short_run(database, "integration_thermoelastic_add_material");

  auto [temperature, displacement] =
      run<3, 1, 2, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>(
          communicator, database, timers);

  // For now doing a simple regression test. Without a dof handler, it's hard to
  // do something more meaningful with the vector.

  check_solution(temperature, displacement);
  BOOST_TEST(displacement.l2_norm() == 0.11831992436508011);
}
