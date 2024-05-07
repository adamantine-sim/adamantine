/* Copyright (c) 2020 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <GoldakHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

#include <deal.II/base/memory_space.h>

namespace adamantine
{

template <int dim, typename MemorySpaceType>
GoldakHeatSource<dim, MemorySpaceType>::GoldakHeatSource(
    BeamHeatSourceProperties const &beam,
    ScanPath<MemorySpaceType> const &scan_path)
    : HeatSource<dim, MemorySpaceType>(beam, scan_path)
{
}

template <int dim, typename MemorySpaceType>
void GoldakHeatSource<dim, MemorySpaceType>::update_time(double time)
{
  static const double _pi_over_3_to_1p5 =
      std::pow(dealii::numbers::PI / 3.0, 1.5);

  _beam_center = this->_scan_path.value(time);
  double segment_power_modifier = this->_scan_path.get_power_modifier(time);
  _alpha = 2.0 * this->_beam.absorption_efficiency * this->_beam.max_power *
           segment_power_modifier /
           (this->_beam.radius_squared * this->_beam.depth * _pi_over_3_to_1p5);
}

template <int dim, typename MemorySpaceType>
double
GoldakHeatSource<dim, MemorySpaceType>::value(dealii::Point<dim> const &point,
                                              double const height) const
{
  double const z = point[axis<dim>::z] - height;
  if ((z + this->_beam.depth) < 0.)
  {
    return 0.;
  }
  else
  {
    double xpy_squared =
        std::pow(point[axis<dim>::x] - _beam_center[axis<dim>::x], 2);
    if (dim == 3)
    {
      xpy_squared +=
          std::pow(point[axis<dim>::y] - _beam_center[axis<dim>::y], 2);
    }

    // Goldak heat source equation
    double heat_source =
        _alpha * std::exp(-3.0 * xpy_squared / this->_beam.radius_squared +
                          -3.0 * std::pow(z / this->_beam.depth, 2));

    return heat_source;
  }
}
} // namespace adamantine

INSTANTIATE_DIM_DEVICE(GoldakHeatSource)
INSTANTIATE_DIM_HOST(GoldakHeatSource)
