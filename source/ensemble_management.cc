/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ensemble_management.hh>

namespace adamantine
{
std::vector<double> fill_and_sync_random_vector(unsigned int length,
                                                double mean, double stddev)
{
  std::vector<double> output_vector(length);

  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::random_device dev;
    std::mt19937 pseudorandom_number_generator(dev());
    std::normal_distribution<> normal_dist_generator(0.0, 1.0);

    for (unsigned int member = 0; member < length; ++member)
    {
      output_vector[member] =
          mean + stddev * normal_dist_generator(pseudorandom_number_generator);
    }
  }

  output_vector =
      dealii::Utilities::MPI::broadcast(MPI_COMM_WORLD, output_vector, 0);

  return output_vector;
}

} // namespace adamantine
