/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef ENSEMBLE_MANAGEMENT_HH
#define ENSEMBLE_MANAGEMENT_HH

#include <deal.II/base/mpi.h>

#include <random>
#include <vector>
namespace adamantine
{
std::vector<double> fill_and_sync_random_vector(unsigned int length,
                                                double mean, double stddev);
} // namespace adamantine

#endif
