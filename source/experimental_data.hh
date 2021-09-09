/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef EXPERIMENTAL_DATA_HH
#define EXPERIMENTAL_DATA_HH

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
 * Fill the @p temperature Vector given @p points_values.
 */
template <int dim>
std::pair<std::vector<int>, std::vector<int>> set_with_experimental_data(
    PointsValues<dim> const &points_values,
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature);
} // namespace adamantine

#endif
