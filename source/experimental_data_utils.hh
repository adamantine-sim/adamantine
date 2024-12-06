/* SPDX-FileCopyrightText: Copyright (c) 2021-2023, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef EXPERIMENTAL_DATA_UTILS_HH
#define EXPERIMENTAL_DATA_UTILS_HH

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * Structure to encapsulate a point cloud and the values associated to each
 * point.
 */
template <int dim>
struct PointsValues
{
  std::vector<dealii::Point<dim>> points;
  std::vector<double> values;
};

/**
 * Read the experimental data (IR point cloud) and return a vector of
 * PointsValues. The size of the vector is equal to the number of frames.
 */
template <int dim>
std::vector<PointsValues<dim>> read_experimental_data_point_cloud(
    MPI_Comm const &communicator,
    boost::property_tree::ptree const &experiment_database);

/**
 * Get the pair of vectors that map the DOF indices to the
 * support points.
 */
template <int dim>
std::pair<std::vector<dealii::types::global_dof_index>,
          std::vector<dealii::Point<dim>>>
get_dof_to_support_mapping(dealii::DoFHandler<dim> const &dof_handler);

/**
 * Get the pair of vectors that map the experimental observation indices to the
 * dof indices.
 */
template <int dim>
std::pair<std::vector<int>, std::vector<int>>
get_expt_to_dof_mapping(PointsValues<dim> const &points_values,
                        dealii::DoFHandler<dim> const &dof_handler);

/**
 * Fill the @p temperature Vector given @p points_values.
 */
template <int dim>
void set_with_experimental_data(
    MPI_Comm const &communicator, PointsValues<dim> const &points_values,
    std::pair<std::vector<int>, std::vector<int>> &expt_to_dof_mapping,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature,
    bool verbose_output);

/**
 * Parse the log file that maps experimental frames to their times.
 * The log file has a headerless CSV format with the frame index (int) as the
 * first entry per line followed by the timestamp for each camera. For now we
 * assume that the first frame for each camera is synced so that we can
 * re-reference the times to a fixed offset from the first frame time. The
 * function returns a vector containing the frame timings for each camera, i.e.
 * the first index is the camera index and the second is the frame index. The
 * frame indices in the output are such that the 'first frame' listed in the
 * input file is index 0.
 */
std::vector<std::vector<double>>
read_frame_timestamps(boost::property_tree::ptree const &experiment_database);

} // namespace adamantine

#endif
