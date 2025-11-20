/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <GoldakHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

namespace adamantine
{

template <int dim>
GoldakHeatSource<dim>::GoldakHeatSource(
    boost::property_tree::ptree const &beam_database,
    boost::optional<boost::property_tree::ptree const &> const
        &units_optional_database)
    : HeatSource<dim>(beam_database, units_optional_database)
{
}

template <int dim>
void GoldakHeatSource<dim>::update_time(double time)
{
  dealii::Point<3> const &path = this->_scan_path.value(time);
  // Copy the scalar value of path to the vectorized data type
  for (unsigned int d = 0; d < 3; ++d)
  {
    _beam_center(d) = path(d);
  }
  double segment_power_modifier = this->_scan_path.get_power_modifier(time);
  _alpha = 2.0 * this->_beam.absorption_efficiency * this->_beam.max_power *
           segment_power_modifier /
           (this->_beam.radius_squared * this->_beam.depth * _pi_over_3_to_1p5);
  _depth = this->_beam.depth;
  _radius_squared = this->_beam.radius_squared;
}

template <int dim>
double GoldakHeatSource<dim>::value(dealii::Point<dim> const &point) const
{
  double const z = point[axis<dim>::z] - _beam_center[axis<dim>::z][0];
  if ((z + this->_beam.depth) < 0.)
  {
    return 0.;
  }
  else
  {
    double xpy_squared =
        std::pow(point[axis<dim>::x] - _beam_center[axis<dim>::x][0], 2);
    if (dim == 3)
    {
      xpy_squared +=
          std::pow(point[axis<dim>::y] - _beam_center[axis<dim>::y][0], 2);
    }

    // Evaluating the exponential is very expensive. Return early if we know
    // that the heat source will be small.
    if (xpy_squared > 5. * this->_beam.radius_squared)
    {
      return 0.;
    }

    // Goldak heat source equation
    double heat_source =
        _alpha[0] * std::exp(-3.0 * xpy_squared / this->_beam.radius_squared +
                             -3.0 * std::pow(z / this->_beam.depth, 2));

    return heat_source;
  }
}

template <int dim>
dealii::VectorizedArray<double> GoldakHeatSource<dim>::value(
    dealii::Point<dim, dealii::VectorizedArray<double>> const &points) const
{
  auto const z = points[axis<dim>::z] - _beam_center[axis<dim>::z];
  auto const z_depth = z + _depth;
  dealii::VectorizedArray<double> depth_mask;
  for (unsigned int i = 0; i < depth_mask.size(); ++i)
  {
    depth_mask[i] = z_depth[i] < 0. ? 0. : 1.;
  }
  // If all the lanes of depth_mask are zero, we can return early
  if (depth_mask.sum() == 0)
  {
    return depth_mask;
  }

  auto xpy_squared = points[axis<dim>::x] - _beam_center[axis<dim>::x];
  xpy_squared *= xpy_squared;
  if constexpr (dim == 3)
  {
    auto y_squared = points[axis<dim>::y] - _beam_center[axis<dim>::y];
    y_squared *= y_squared;
    xpy_squared += y_squared;
  }

  // Evaluating the exponential is very expensive. Return early if we know
  // that the heat source will be small.
  auto const xpy_area = 5. * _radius_squared - xpy_squared;
  dealii::VectorizedArray<double> xpy_mask;
  for (unsigned int i = 0; i < xpy_mask.size(); ++i)
  {
    xpy_mask[i] = xpy_area[i] < 0. ? 0. : 1.;
  }
  // If all the lanes of xpy_mask are zero, we can return early
  if (xpy_mask.sum() == 0)
  {
    return xpy_mask;
  }

  // Goldak heat source equation:
  // alpha * exp(-3*(xpy_squared/radius_squared + (z/depth)^2))
  dealii::VectorizedArray<double> minus_three = -3.;
  xpy_squared /= _radius_squared;
  auto exponent = z / _depth;
  exponent *= exponent;
  exponent += xpy_squared;
  exponent *= minus_three;
  return depth_mask * _alpha * std::exp(exponent);
}

template <int dim>
dealii::BoundingBox<dim>
GoldakHeatSource<dim>::get_bounding_box(double const time,
                                        double const scaling_factor) const
{
  dealii::Point<3> const &beam_center = this->_scan_path.value(time, false);
  if constexpr (dim == 2)
  {
    return {{{beam_center[axis<dim>::x] - scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::z] - scaling_factor * this->_beam.depth},
             {beam_center[axis<dim>::x] + scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::z]}}};
  }
  else
  {
    return {{{beam_center[axis<dim>::x] - scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::y] - scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::z] - scaling_factor * this->_beam.depth},
             {beam_center[axis<dim>::x] + scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::y] + scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::z]}}};
  }
}

} // namespace adamantine

INSTANTIATE_DIM(GoldakHeatSource)
