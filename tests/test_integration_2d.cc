/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_2D

#include "../application/adamantine.hh"

#include "main.cc"

BOOST_AUTO_TEST_CASE(intregation_2D)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "integration_2d.info";
  adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  auto result =
      run<2, dealii::MemorySpace::Host>(communicator, database, timers);

  double const tolerance = 0.1;
  for (unsigned int i = 0; i < result.local_size(); ++i)
    BOOST_CHECK_CLOSE(result.local_element(i), 1.357e+09, tolerance);
}
