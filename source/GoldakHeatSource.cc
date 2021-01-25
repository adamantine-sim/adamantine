/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <GoldakHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

namespace adamantine
{

template <int dim>
GoldakHeatSource<dim>::GoldakHeatSource(
    boost::property_tree::ptree const &database)
    : HeatSource<dim>(database)
{
}

template <int dim>
double GoldakHeatSource<dim>::value(dealii::Point<dim> const &point,
                                    double const time,
                                    double const height) const
{
  double const z = point[axis<dim>::z] - height;
  if ((z + this->_beam.depth) < 0.)
  {
    return 0.;
  }
  else
  {
    dealii::Point<3> const beam_center = this->_scan_path.value(time);
    double xpy_squared =
        std::pow(point[axis<dim>::x] - beam_center[axis<dim>::x], 2);
    if (dim == 3)
    {
      xpy_squared +=
          std::pow(point[axis<dim>::y] - beam_center[axis<dim>::y], 2);
    }
    double segment_power_modifier = this->_scan_path.get_power_modifier(time);
    double pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);

    // Goldak heat source equation
    double heat_source =
        2.0 * this->_beam.absorption_efficiency * this->_beam.max_power *
        segment_power_modifier /
        (this->_beam.radius_squared * this->_beam.depth * pi_over_3_to_1p5) *
        std::exp(-3.0 * xpy_squared / this->_beam.radius_squared +
                 -3.0 * std::pow(z / this->_beam.depth, 2));

    return heat_source;
  }
}
} // namespace adamantine

INSTANTIATE_DIM(GoldakHeatSource)
