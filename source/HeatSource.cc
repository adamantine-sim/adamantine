/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCE_TEMPLATES_HH
#define HEAT_SOURCE_TEMPLATES_HH

#include <HeatSource.hh>
#include <instantiation.hh>
#include <utils.hh>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <cstdlib>

using std::pow;

namespace adamantine
{

template <int dim>
HeatSource<dim>::HeatSource(boost::property_tree::ptree const &database)
    : dealii::Function<dim>(), _max_height(0.), _scan_path("scan_path.txt")
{
  // Set the properties of the electron beam.
  _beam.depth = database.get<double>("depth");
  _beam.absorption_efficiency = database.get<double>("absorption_efficiency");
  _beam.radius_squared = pow(database.get("diameter", 2e-3) / 2.0, 2);
  boost::optional<double> max_power =
      database.get_optional<double>("max_power");
  if (max_power)
    _beam.max_power = max_power.get();
  else
  {
    std::string message =
        "When using HeatSource, the max power is not optional.";
    throw std::runtime_error(message);
  }
}

template <int dim>
void HeatSource<dim>::rewind_time()
{
}

template <int dim>
void HeatSource<dim>::save_time()
{
}

template <int dim>
double HeatSource<dim>::value(dealii::Point<dim> const &point,
                              unsigned int const /*component*/) const
{

  double const z = point[1] - _max_height;
  if ((z + _beam.depth) < 0.)
  {
    return 0.;
  }
  else
  {
    dealii::Point<1> time;
    time[0] = this->get_time();
    double const beam_center_x = _scan_path.value(time, 0);
    double xpy_squared = pow(point[0] - beam_center_x, 2);
    if (dim == 3)
    {
      double const beam_center_y = _scan_path.value(time, 2);
      xpy_squared += pow(point[2] - beam_center_y, 2);
    }
    double segment_power_modifier = _scan_path.get_power_modifier(time);
    double pi_over_3_to_1p5 = pow(dealii::numbers::PI / 3.0, 1.5);

    // Goldak heat source
    double heat_source =
        -2.0 * _beam.absorption_efficiency * _beam.max_power *
        segment_power_modifier /
        (_beam.radius_squared * _beam.depth * pi_over_3_to_1p5) *
        std::exp(-3.0 * xpy_squared / _beam.radius_squared +
                 -3.0 * pow(z / _beam.depth, 2));

    return heat_source;
  }
}
} // namespace adamantine

INSTANTIATE_DIM(HeatSource)

#endif
