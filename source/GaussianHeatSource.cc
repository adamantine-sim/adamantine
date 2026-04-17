/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <GaussianHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

#include <deal.II/base/utilities.h>

namespace adamantine
{

template <int dim>
GaussianHeatSource<dim>::GaussianHeatSource(
    boost::property_tree::ptree const &beam_database,
    boost::optional<boost::property_tree::ptree const &> const
        &units_optional_database)
    : HeatSource<dim>(beam_database, units_optional_database),
      _five_axis(this->_scan_path.is_five_axis())
{
  _A = beam_database.get<double>("A");
  _B = beam_database.get<double>("B");
}

template <int dim>
void GaussianHeatSource<dim>::update_time(double time)
{
  double const segment_power_modifier =
      this->_scan_path.get_power_modifier(time);
  this->_source_on = (segment_power_modifier > 0.0);

  dealii::Point<3> const &path = this->_scan_path.value(time);
  _quaternion = this->_scan_path.get_current_quaternion();

  for (unsigned int d = 0; d < 3; ++d)
    _beam_center(d) = path(d);

  _depth = this->_beam.depth;
  _radius_squared = this->_beam.radius_squared;

  double const aspect_ratio = std::max(1.0, _depth[0] / this->_beam.radius);
  double const n =
      std::min(std::max(0.0, _A * std::log2(aspect_ratio) + _B), 9.0);
  _k = std::pow(2.0, n);

  double const V0 =
      0.5 * dealii::numbers::PI * _radius_squared[0] * _depth[0] *
      std::tgamma(1.0 / _k) / (_k * std::pow(3.0, 1.0 / _k));

  double const effective_power =
      this->_beam.absorption_efficiency * this->_beam.max_power *
      segment_power_modifier;

  _alpha = effective_power / V0;
}

template <int dim>
double GaussianHeatSource<dim>::value(dealii::Point<dim> const &point) const
{
  dealii::Point<dim> rotated_point;
  if constexpr (dim == 2)
  {
    rotated_point = point;
  }
  else
  {
    rotated_point = _five_axis ? _quaternion.rotate(point) : point;
  }

  double const z = rotated_point[axis<dim>::z] - _beam_center[axis<dim>::z][0];
  if ((z + _depth[0]) < 0.)
  {
    return 0.;
  }

  double xpy_squared = dealii::Utilities::fixed_power<2>(
      rotated_point[axis<dim>::x] - _beam_center[axis<dim>::x][0]);
  if constexpr (dim == 3)
  {
    xpy_squared += dealii::Utilities::fixed_power<2>(
        rotated_point[axis<dim>::y] - _beam_center[axis<dim>::y][0]);
  }

  if (xpy_squared > 5. * _radius_squared[0])
  {
    return 0.;
  }
  if (std::abs(z) > 5. * _depth[0])
  {
    return 0.;
  }

  double const radial_component =
      std::exp(-2.0 * xpy_squared / _radius_squared[0]);

  double const depth_component =
      std::exp(-3.0 * std::pow(std::abs(z) / _depth[0], _k));

  return _alpha[0] * radial_component * depth_component;
}

template <int dim>
dealii::VectorizedArray<double> GaussianHeatSource<dim>::value(
    dealii::Point<dim, dealii::VectorizedArray<double>> const &points) const
{
  dealii::Point<dim, dealii::VectorizedArray<double>> rotated_points;
  if constexpr (dim == 2)
  {
    rotated_points = points;
  }
  else
  {
    rotated_points = _five_axis ? _quaternion.rotate(points) : points;
  }

  auto const z = rotated_points[axis<dim>::z] - _beam_center[axis<dim>::z];
  auto const z_depth = z + _depth;

  dealii::VectorizedArray<double> depth_mask;
  for (unsigned int i = 0; i < depth_mask.size(); ++i)
  {
    depth_mask[i] = z_depth[i] < 0. ? 0. : 1.;
  }
  if (depth_mask.sum() == 0)
  {
    return depth_mask;
  }

  auto xpy_squared = dealii::Utilities::fixed_power<2>(
      rotated_points[axis<dim>::x] - _beam_center[axis<dim>::x]);
  if constexpr (dim == 3)
  {
    auto const y_squared = dealii::Utilities::fixed_power<2>(
        rotated_points[axis<dim>::y] - _beam_center[axis<dim>::y]);
    xpy_squared += y_squared;
  }

  auto const xpy_area = 5. * _radius_squared - xpy_squared;
  dealii::VectorizedArray<double> xpy_mask;
  for (unsigned int i = 0; i < xpy_mask.size(); ++i)
  {
    xpy_mask[i] = xpy_area[i] < 0. ? 0. : 1.;
  }
  if (xpy_mask.sum() == 0)
  {
    return xpy_mask;
  }

  dealii::VectorizedArray<double> shape = _k;

  auto const radial_component =
      std::exp(-2.0 * xpy_squared / _radius_squared);

  auto const depth_component =
      std::exp(-3.0 * std::pow(std::abs(z / _depth), shape));

  return depth_mask * _alpha * radial_component * depth_component;
}

template <int dim>
dealii::BoundingBox<dim>
GaussianHeatSource<dim>::get_bounding_box(double const time,
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
    dealii::Point<3> max_corner(
        beam_center[axis<dim>::x] + scaling_factor * this->_beam.radius,
        beam_center[axis<dim>::y] + scaling_factor * this->_beam.radius,
        beam_center[axis<dim>::z]);
    dealii::Point<3> min_corner(
        beam_center[axis<dim>::x] - scaling_factor * this->_beam.radius,
        beam_center[axis<dim>::y] - scaling_factor * this->_beam.radius,
        beam_center[axis<dim>::z] - scaling_factor * this->_beam.depth);

    if (this->_scan_path.is_five_axis())
    {
      dealii::Point<3> rotated_max_corner =
          this->_scan_path.rotate(time, max_corner);
      dealii::Point<3> rotated_min_corner =
          this->_scan_path.rotate(time, min_corner);

      dealii::Point<3> new_max_corner;
      dealii::Point<3> new_min_corner;
      for (int d = 0; d < 3; ++d)
      {
        new_max_corner[d] =
            std::max(rotated_max_corner[d], rotated_min_corner[d]);
        new_min_corner[d] =
            std::min(rotated_max_corner[d], rotated_min_corner[d]);
      }

      return {{new_min_corner, new_max_corner}};
    }
    else
    {
      return {{min_corner, max_corner}};
    }
  }
}

} // namespace adamantine

INSTANTIATE_DIM(GaussianHeatSource)
