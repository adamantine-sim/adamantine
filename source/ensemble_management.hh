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
 * Return the sources encompassing all the sources of the different ensemble
 * members.
 */
template <int dim>
std::vector<std::shared_ptr<HeatSource<dim>>> get_bounding_heat_sources(
    std::vector<boost::property_tree::ptree> const &property_trees,
    MPI_Comm global_communicator);

/**
 * Given an input property tree @p database, return @p local_ensemble_size
 * databases each one modified to respect the standard deviation of each
 * quantity (if provided).
 */
std::vector<boost::property_tree::ptree> create_database_ensemble(
    boost::property_tree::ptree const &database, MPI_Comm local_communicator,
    unsigned int first_local_member, unsigned int local_ensemble_size,
    unsigned int global_ensemble_size);
} // namespace adamantine

#endif
