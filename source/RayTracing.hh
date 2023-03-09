/* Copyright (c) 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef RAY_TRACING_HH
#define RAY_TRACING_HH

#include <ExperimentalData.hh>

#include <deal.II/dofs/dof_handler.h>

namespace adamantine
{
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
class RayTracing final : public ExperimentalData<3>
{
public:
  static int constexpr dim = 3;

  /**
   * Constructor.
   */
  RayTracing(boost::property_tree::ptree const &experiment_database,
             dealii::DoFHandler<dim> const &dof_handler);

  unsigned int read_next_frame() override;

  PointsValues<dim> get_points_values() override;

private:
  /**
   * Next frame that should be read.
   */
  unsigned int _next_frame;
  /**
   * ID of the first camera.
   */
  unsigned int _first_camera_id;
  /**
   * ID of the last camera.
   */
  unsigned int _last_camera_id;
  /**
   * Generic file name of the frames.
   */
  std::string _data_filename;
  /**
   * DoFHandler of the mesh we want to perform the ray tracing on.
   */
  dealii::DoFHandler<dim> const &_dof_handler;
  /**
   * Rays associated to the current frame.
   */
  std::vector<Ray<dim>> _rays_current_frame;
  /**
   * Values associated to the rays of the current frame.
   */
  std::vector<double> _values_current_frame;
};

} // namespace adamantine

#endif
