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

/**
 * Data structure representing a ray.
 */
template <int dim>
struct Ray
{
  /**
   * Origin of the ray.
   */
  dealii::Point<dim> origin;
  /**
   * Direction of propagation of the ray.
   */
  dealii::Tensor<1, dim> direction;
};

/**
 * This class performs ray tracing
 */
class RayTracing
{
public:
  /**
   * This class assumes the geometry is 3D.
   */
  static int constexpr dim = 3;

  /**
   * Constructor. Read all the rays and the values from all the frames.
   */
  RayTracing(boost::property_tree::ptree const &experiment_database);

  /**
   * Perform the ray tracing given a DoFHandler @p dof_handler. @p frame is the
   * relative frame the number, i.e., current frame id - first frame id
   */
  PointsValues<dim> get_intersection(dealii::DoFHandler<dim> const &dof_handler,
                                     unsigned int frame);

private:
  /**
   * Rays associated to each frame.
   */
  std::vector<std::vector<Ray<dim>>> _rays_all_frames;
  /**
   * Values associated to the rays of each frame.
   */
  std::vector<std::vector<double>> _values_all_frames;
};

} // namespace adamantine

#endif
