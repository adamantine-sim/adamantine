/* SPDX-FileCopyrightText: Copyright (c) 2021-2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef ENSEMBLE_MANAGEMENT_HH
#define ENSEMBLE_MANAGEMENT_HH

#include <HeatSource.hh>

#include <deal.II/base/mpi.h>

#include <boost/property_tree/ptree.hpp>

#include <memory>
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

/**
 * Return the sources encompassing all the sources of the different ensemble
 * members.
 */
template <int dim>
std::vector<std::shared_ptr<HeatSource<dim>>> get_bounding_heat_sources(
    std::vector<boost::property_tree::ptree> const &property_trees,
    MPI_Comm global_communicator);
} // namespace adamantine

#endif
