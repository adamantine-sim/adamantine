/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_3D_AMR

#include "../application/adamantine.hh"

#include <fstream>

#include "main.cc"

BOOST_AUTO_TEST_CASE(integration_3D_amr)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "amr_test.info";
  adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto result =
      run<3, dealii::MemorySpace::Host>(communicator, database, timers);

  double min_val = std::numeric_limits<double>::max();
  double max_val = std::numeric_limits<double>::min();
  for (unsigned int i = 0; i < result.locally_owned_size(); ++i)
  {
    if (result.local_element(i) < min_val)
      min_val = result.local_element(i);

    if (result.local_element(i) > max_val)
      max_val = result.local_element(i);
  }

  double global_max =
      dealii::Utilities::MPI::max(max_val, result.get_mpi_communicator());
  double global_min =
      dealii::Utilities::MPI::min(min_val, result.get_mpi_communicator());

  std::cout << min_val << " " << global_min << std::endl;
  std::cout << max_val << " " << global_max << std::endl;

  double expected_max = 527.0;
  double expected_min = 281.6;

  double const tolerance = 0.1;

  BOOST_CHECK_CLOSE(expected_max, global_max, tolerance);
  BOOST_CHECK_CLOSE(expected_min, global_min, tolerance);
}
