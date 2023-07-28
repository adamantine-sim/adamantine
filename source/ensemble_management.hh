/* Copyright (c) 2021-2023, the adamantine authors.
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
/**
 * Return a vector of size @p length, with random values drawn following a
 * normal distribution of average @p mean and standard deviation @stddev. The
 * first @p n_rejected_draws are rejected.
 */
std::vector<double> get_normal_random_vector(unsigned int length,
                                             unsigned int n_rejected_draws,
                                             double mean, double stddev);
} // namespace adamantine

#endif
