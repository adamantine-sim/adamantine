/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ensemble_management.hh>

#include <numeric>

#define BOOST_TEST_MODULE EnsembleManagement

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "main.cc"

namespace utf = boost::unit_test;

// Fairly loose tolerance because this is a statistical check
BOOST_AUTO_TEST_CASE(fill_and_sync_random_vector, *utf::tolerance(10.))
{

  // Create the random vector
  double mean = 12.2;
  double stddev = 0.25;
  unsigned int ensemble_size = 5000;
  std::vector<double> vec = adamantine::get_normal_random_vector(
      ensemble_size, 0, mean, stddev, false);

  // Check vector size
  BOOST_TEST(vec.size() == ensemble_size);

  // Check vector mean
  double mean_check =
      std::accumulate(vec.cbegin(), vec.cend(), 0.) / ensemble_size;

  BOOST_TEST(mean == mean_check);

  // Check vector variance
  boost::accumulators::accumulator_set<
      double, boost::accumulators::features<boost::accumulators::tag::variance>>
      acc;
  for (unsigned int member = 0; member < ensemble_size; ++member)
  {
    acc(vec[member]);
  }
  BOOST_TEST(stddev * stddev == boost::accumulators::variance(acc));
}
