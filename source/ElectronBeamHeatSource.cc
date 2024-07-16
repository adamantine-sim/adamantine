/* Copyright (c) 2020 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ElectronBeamHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

#include <deal.II/base/memory_space.h>

namespace adamantine
{

template <int dim>
ElectronBeamHeatSource<dim>::ElectronBeamHeatSource(
    boost::property_tree::ptree const &database)
    : _beam(database),
      _scan_path(database.get<std::string>("scan_path_file"),
                 database.get<std::string>("scan_path_file_format"))
{
}

template <int dim>
void ElectronBeamHeatSource<dim>::update_time(double time)
{
  static const double log_01 = std::log(0.1);
  _beam_center = this->_scan_path.value(time);
  double segment_power_modifier = this->_scan_path.get_power_modifier(time);
  _alpha =
      -this->_beam.absorption_efficiency * this->_beam.max_power *
      segment_power_modifier * log_01 /
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
    if (dim == 3)
    {
      xpy_squared +=
          std::pow(point[axis<dim>::y] - _beam_center[axis<dim>::y], 2);
    }

    static const double log_01 = std::log(0.1);

    // Electron beam heat source equation
    double heat_source =
        _alpha * std::exp(log_01 * xpy_squared / this->_beam.radius_squared) *
        distribution_z;

    return heat_source;
  }
}
} // namespace adamantine

INSTANTIATE_DIM(ElectronBeamHeatSource)
