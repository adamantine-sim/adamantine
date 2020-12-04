/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <GoldakHeatSource.hh>
#include <instantiation.hh>

namespace adamantine
{

template <int dim>
GoldakHeatSource<dim>::GoldakHeatSource(
    boost::property_tree::ptree const &database, double const max_height)
    : HeatSource<dim>(database, max_height)
{
}

template <int dim>
double GoldakHeatSource<dim>::value(dealii::Point<dim> const &point,
                                    double const time) const
{
  // NOTE: Due to the differing coordinate systems, "z" here is the second
  // component of the input point.
  double const z = point[1] - this->_max_height;
  if ((z + this->_beam.depth) < 0.)
  {
    return 0.;
  }
  else
  {
    dealii::Point<3> const beam_center = this->_scan_path.value(time);
    double xpy_squared = std::pow(point[0] - beam_center[0], 2);
    if (dim == 3)
    {
      // NOTE: Due to the differing coordinate systems, "y" here is the third
      // component of the input point.
      xpy_squared += std::pow(point[2] - beam_center[1], 2);
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
