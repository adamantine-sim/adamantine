/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2024, the adamantine authors.
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
  _beam_center = this->_scan_path.value(time);
  double segment_power_modifier = this->_scan_path.get_power_modifier(time);
  _alpha =
      -this->_beam.absorption_efficiency * this->_beam.max_power *
      segment_power_modifier * _log_01 /
      (dealii::numbers::PI * this->_beam.radius_squared * this->_beam.depth);
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
        std::pow(point[axis<dim>::x] - _beam_center[axis<dim>::x], 2);
    if constexpr (dim == 3)
    {
      xpy_squared +=
          std::pow(point[axis<dim>::y] - _beam_center[axis<dim>::y], 2);
    }

    // Electron beam heat source equation
    double heat_source =
        _alpha * std::exp(_log_01 * xpy_squared / this->_beam.radius_squared) *
        distribution_z;

    return heat_source;
  }
}

template <int dim>
dealii::BoundingBox<dim>
ElectronBeamHeatSource<dim>::get_bounding_box(double const scaling_factor) const
{
  if constexpr (dim == 2)
  {
    return {{{_beam_center[axis<dim>::x] - scaling_factor * this->_beam.radius,
              _beam_center[axis<dim>::z] - scaling_factor * this->_beam.depth},
             {_beam_center[axis<dim>::x] + scaling_factor * this->_beam.radius,
              _beam_center[axis<dim>::z]}}};
  }
  else
  {
    return {{{_beam_center[axis<dim>::x] - scaling_factor * this->_beam.radius,
              _beam_center[axis<dim>::y] - scaling_factor * this->_beam.radius,
              _beam_center[axis<dim>::z] - scaling_factor * this->_beam.depth},
             {_beam_center[axis<dim>::x] + scaling_factor * this->_beam.radius,
              _beam_center[axis<dim>::y] + scaling_factor * this->_beam.radius,
              _beam_center[axis<dim>::z]}}};
  }
}
} // namespace adamantine

INSTANTIATE_DIM(ElectronBeamHeatSource)
