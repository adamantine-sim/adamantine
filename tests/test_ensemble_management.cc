/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ensemble_management.hh>

#include <deal.II/base/mpi.h>

#include <numeric>

#define BOOST_TEST_MODULE EnsembleManagement

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "main.cc"

BOOST_AUTO_TEST_CASE(fill_and_sync_random_vector)
{
  // Fairly loose tolerance because this is a statistical check
  double tolerance = 10.0;

  // Create the random vector
  double mean = -1.2;
  double stddev = 0.25;
  unsigned int ensemble_size = 5000;
  std::vector<double> vec =
      adamantine::fill_and_sync_random_vector(ensemble_size, mean, stddev);

  // Check vector size
  BOOST_CHECK(vec.size() == ensemble_size);

  // Check vector mean
  double mean_check = std::reduce(vec.cbegin(), vec.cend()) / ensemble_size;

  BOOST_CHECK_CLOSE(mean, mean_check, tolerance);

  // Check vector variance
  boost::accumulators::accumulator_set<
      double, boost::accumulators::features<boost::accumulators::tag::variance>>
      acc;
  for (unsigned int member = 0; member < ensemble_size; ++member)
  {
    acc(vec[member]);
  }
  BOOST_CHECK_CLOSE(stddev * stddev, boost::accumulators::variance(acc),
                    tolerance);
}
