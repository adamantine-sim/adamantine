/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef CUBE_HEAT_SOURCE_HH
#define CUBE_HEAT_SOURCE_HH

#include <HeatSource.hh>

#include <deal.II/base/memory_space.h>

namespace adamantine
{
/**
 * Cube heat source. This source does not represent a physical source, it is
 * used for verification purpose.
 */
template <int dim>
class CubeHeatSource final
    : public HeatSource<dim, dealii::MemorySpace::Default>
{
public:
  /**
   * Constructor.
   *  \param[in] database requires the following entries:
   *   - <B>start_time</B>: double (when the source is turned on)
   *   - <B>end_time</B>: double (when the source is turned off)
   *   - <B>value</B>: double (value of the soruce)
   *   - <B>min_x</B>: double (minimum x coordinate of the cube)
   *   - <B>max_x</B>: double (maximum x coordinate of the cube)
   *   - <B>min_y</B>: double (minimum y coordinate of the cube)
   *   - <B>max_y</B>: double (maximum y coordinate of the cube)
   *   - <B>min_z</B>: double (3D only, minimum z coordinate of the cube)
   *   - <B>max_z</B>: double (3D only, maximum z coordinate of the cube)
   */
  CubeHeatSource(boost::property_tree::ptree const &database);

  /**
   * Set the time variable.
   */
  void update_time(double time) final;

  /**
   * Return the value of the source for a given point and time.
   */
  double value(dealii::Point<dim> const &point,
               double const /*height*/) const final;
  /**
   * Compute the current height of the where the heat source meets the material
   * (i.e. the current scan path height).
   */
  double get_current_height(double const time) const final;

private:
  bool _source_on = false;
  double _start_time;
  double _end_time;
  double _value;
  dealii::Point<dim> _min_point;
  dealii::Point<dim> _max_point;
};
} // namespace adamantine

#endif
