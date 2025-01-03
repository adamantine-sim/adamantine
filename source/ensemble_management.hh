/* SPDX-FileCopyrightText: Copyright (c) 2021-2023, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
 * normal distribution of average @p mean and standard deviation @p stddev. The
 * first @p n_rejected_draws are rejected. @p n_rejected_draws is used to
 * ensure that wether we use one MPI rank or ten, we use the same random
 * numbers. If we don't reject the first few draws, all the processors will use
 * the same "random" numbers since they have the same seed. We also reject
 * negative values even because our physical quantities must be positive. If @p
 * verbose is true, we output a message when a negative value was rejected.
 */
std::vector<double> get_normal_random_vector(unsigned int length,
                                             unsigned int n_rejected_draws,
                                             double mean, double stddev,
                                             bool verbose);
} // namespace adamantine

#endif
