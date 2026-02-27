/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <GaussianHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

namespace adamantine
{

template <int dim>
GaussianHeatSource<dim>::GaussianHeatSource(
    boost::property_tree::ptree const &beam_database,
    boost::optional<boost::property_tree::ptree const &> const
        &units_optional_database)
    : HeatSource<dim>(beam_database, units_optional_database)
{
  // Read the mandatory empirical coefficients for the Gaussian model.
  _A = beam_database.get<double>("A");
  _B = beam_database.get<double>("B");
}

template <int dim>
void GaussianHeatSource<dim>::update_time(double time)
{
  // Get the current beam center from the scan path
  dealii::Point<3> const &path = this->_scan_path.value(time);
  for (unsigned int d = 0; d < 3; ++d)
  {
    _beam_center(d) = path(d);
  }

  // Update the beam dimensions
  _depth = this->_beam.depth;
  _radius_squared = this->_beam.radius_squared;

  // Calculate the shape factor 'k' based on the beam aspect ratio.
  double const aspect_ratio = std::max(1.0, _depth[0] / this->_beam.radius); // TODO: check
  double const n = std::min(std::max(0.0, _A * std::log2(aspect_ratio) + _B), 9.0);
  _k = std::pow(2.0, n);
  
  // Calculate the effective volume (V0) of the heat source.
  // This is the integral of the weight function and is used for normalization.
  auto const V0 = 0.5 * dealii::numbers::PI * _radius_squared * _depth *
                  std::tgamma(1.0 / _k) /
                  (_k * std::pow(3.0, 1.0 / _k));

  double segment_power_modifier = this->_scan_path.get_power_modifier(time);
  double const effective_power = this->_beam.absorption_efficiency *
                                 this->_beam.max_power *
                                 segment_power_modifier;
                                 
  // Normalize alpha to ensure total power equals the integral of the heat source
  _alpha = effective_power / V0;
}

template <int dim>
double GaussianHeatSource<dim>::value(dealii::Point<dim> const &point,
                                    double const height) const
{
  // Get the reference depth of the beam
  double const z = point[axis<dim>::z] - height;
  if ((z + this->_beam.depth) < 0.)
  {
    return 0.;
  }
  
  double xpy_squared =
      std::pow(point[axis<dim>::x] - _beam_center[axis<dim>::x][0], 2);

  if (dim == 3)
  {
    xpy_squared +=
        std::pow(point[axis<dim>::y] - _beam_center[axis<dim>::y][0], 2);
  }

  // Optimization: return zero if the heat source contribution will be negligible.
  if (xpy_squared > 5. * this->_beam.radius_squared)
  {
    return 0.;
  }
  if (std::abs(z) > 5. * this->_beam.depth)
  {
    return 0.;
  }
  
  // Radial component
  double const radial_component =
      std::exp(-2.0 * xpy_squared / this->_radius_squared[0]); // TODO: check

  // Depth component
  double const depth_component =
      std::exp(-3.0 * std::pow(std::abs(z / this->_beam.depth), _k));
 
  // Heat source equation  
  return _alpha[0] * radial_component * depth_component;
}

template <int dim>
dealii::VectorizedArray<double> GaussianHeatSource<dim>::value(
    dealii::Point<dim, dealii::VectorizedArray<double>> const &points,
    dealii::VectorizedArray<double> const &height) const
{
  auto const z = points[axis<dim>::z] - height;
  auto const z_depth = z + _depth;
  
  // Optimization: create a mask for points where the heat source is negligible (depth)
  dealii::VectorizedArray<double> depth_mask;
  for (unsigned int i = 0; i < depth_mask.size(); ++i)
  {
    depth_mask[i] = z_depth[i] < 0. ? 0. : 1.;
  }
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

  // Optimization: create a mask for points where the heat source is negligible (radial)
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

  // Gaussian heat source equation:
  // alpha * exp(-3*(xpy_squared/radius_squared + (z/depth)^2))
  auto const radial_component =
    std::exp(-2.0 * xpy_squared / _radius_squared);

  dealii::VectorizedArray<double> shape = _k;

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
    return {{{beam_center[axis<dim>::x] - scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::y] - scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::z] - scaling_factor * this->_beam.depth},
             {beam_center[axis<dim>::x] + scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::y] + scaling_factor * this->_beam.radius,
              beam_center[axis<dim>::z]}}};
  }
}

} // namespace adamantine

INSTANTIATE_DIM(GaussianHeatSource)
