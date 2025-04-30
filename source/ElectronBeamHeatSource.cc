/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ElectronBeamHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

namespace adamantine
{

template <int dim>
ElectronBeamHeatSource<dim>::ElectronBeamHeatSource(
    boost::property_tree::ptree const &beam_database,
    boost::optional<boost::property_tree::ptree const &> const
        &units_optional_database)
    : HeatSource<dim>(beam_database, units_optional_database)
{
}

template <int dim>
void ElectronBeamHeatSource<dim>::update_time(double time)
{
  dealii::Point<3> const &path = this->_scan_path.value(time);
  // Copy the scalar value of path to the vectorized data type
  for (unsigned int d = 0; d < 3; ++d)
  {
    _beam_center(d) = path(d);
  }
  double segment_power_modifier = this->_scan_path.get_power_modifier(time);
  _alpha =
      -this->_beam.absorption_efficiency * this->_beam.max_power *
      segment_power_modifier * _log_01 /
      (dealii::numbers::PI * this->_beam.radius_squared * this->_beam.depth);
  _depth = this->_beam.depth;
  _radius_squared = this->_beam.radius_squared;
}

template <int dim>
double ElectronBeamHeatSource<dim>::value(dealii::Point<dim> const &point,
                                          double const height) const
{
  double const z = point[axis<dim>::z] - height;
  if ((z + this->_beam.depth) < 0.)
  {
    return 0.;
  }
  else
  {
    double const distribution_z = -3. * std::pow(z / this->_beam.depth, 2) -
                                  2. * (z / this->_beam.depth) + 1.;

    double xpy_squared =
        std::pow(point[axis<dim>::x] - _beam_center[axis<dim>::x][0], 2);
    if constexpr (dim == 3)
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

    // Electron beam heat source equation
    double heat_source =
        _alpha[0] *
        std::exp(_log_01 * xpy_squared / this->_beam.radius_squared) *
        distribution_z;

    return heat_source;
  }
}

template <int dim>
dealii::VectorizedArray<double> ElectronBeamHeatSource<dim>::value(
    dealii::Point<dim, dealii::VectorizedArray<double>> const &points,
    dealii::VectorizedArray<double> const &height) const
{
  auto const z = points[axis<dim>::z] - height;
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

  dealii::VectorizedArray<double> minus_three = -3.;
  dealii::VectorizedArray<double> one = 1.;
  dealii::VectorizedArray<double> z_rel = z / _depth;
  dealii::VectorizedArray<double> distribution_z = z_rel * z_rel;
  distribution_z *= minus_three;
  distribution_z -= z_rel;
  distribution_z -= z_rel;
  distribution_z += one;

  dealii::VectorizedArray<double> exponent = _log_01 * xpy_squared;
  exponent /= _radius_squared;
  return depth_mask * _alpha * std::exp(exponent) * distribution_z;
}

template <int dim>
dealii::BoundingBox<dim>
ElectronBeamHeatSource<dim>::get_bounding_box(double const scaling_factor) const
{
  if constexpr (dim == 2)
  {
    return {
        {{_beam_center[axis<dim>::x][0] - scaling_factor * this->_beam.radius,
          _beam_center[axis<dim>::z][0] - scaling_factor * this->_beam.depth},
         {_beam_center[axis<dim>::x][0] + scaling_factor * this->_beam.radius,
          _beam_center[axis<dim>::z][0]}}};
  }
  else
  {
    return {
        {{_beam_center[axis<dim>::x][0] - scaling_factor * this->_beam.radius,
          _beam_center[axis<dim>::y][0] - scaling_factor * this->_beam.radius,
          _beam_center[axis<dim>::z][0] - scaling_factor * this->_beam.depth},
         {_beam_center[axis<dim>::x][0] + scaling_factor * this->_beam.radius,
          _beam_center[axis<dim>::y][0] + scaling_factor * this->_beam.radius,
          _beam_center[axis<dim>::z][0]}}};
  }
}
} // namespace adamantine

INSTANTIATE_DIM(ElectronBeamHeatSource)
