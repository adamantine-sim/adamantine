/* SPDX-FileCopyrightText: Copyright (c) 2021-2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ensemble_management.hh>
#include <utils.hh>

namespace adamantine
{
std::vector<double> get_normal_random_vector(unsigned int length,
                                             unsigned int n_rejected_draws,
                                             double mean, double stddev)
{
  ASSERT(stddev >= 0., "Internal Error");

  std::mt19937 pseudorandom_number_generator;
  std::normal_distribution<> normal_dist_generator(mean, stddev);
  for (unsigned int i = 0; i < n_rejected_draws; ++i)
  {
    normal_dist_generator(pseudorandom_number_generator);
  }

  std::vector<double> output_vector(length);
  for (unsigned int i = 0; i < length; ++i)
  {
    output_vector[i] = normal_dist_generator(pseudorandom_number_generator);
  }

  return output_vector;
}

} // namespace adamantine
