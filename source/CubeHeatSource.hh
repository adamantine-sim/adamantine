/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef CUBE_HEAT_SOURCE_HH
#define CUBE_HEAT_SOURCE_HH

#include <HeatSource.hh>

namespace adamantine
{
/**
 * Cube heat source. This source does not represent a physical source, it is
 * used for verification purpose.
 */
template <int dim>
class CubeHeatSource final : public HeatSource<dim>
{
public:
  /**
   * Constructor.
   *  \param[in] source_database requires the following entries:
   *   - <B>start_time</B>: double (when the source is turned on)
   *   - <B>end_time</B>: double (when the source is turned off)
   *   - <B>value</B>: double (value of the soruce)
   *   - <B>min_x</B>: double (minimum x coordinate of the cube)
   *   - <B>max_x</B>: double (maximum x coordinate of the cube)
   *   - <B>min_y</B>: double (minimum y coordinate of the cube)
   *   - <B>max_y</B>: double (maximum y coordinate of the cube)
   *   - <B>min_z</B>: double (3D only, minimum z coordinate of the cube)
   *   - <B>max_z</B>: double (3D only, maximum z coordinate of the cube)
   *  \param[in] units_optional_database may have the following entries:
   *   - <B>heat_source.dimension</B>
   *   - <B>heat_source.power</B>
   */
  CubeHeatSource(boost::property_tree::ptree const &source_database,
                 boost::optional<boost::property_tree::ptree const &> const
                     &units_optional_database);

  /**
   * Set the time variable.
   */
  void update_time(double time) final;

  /**
   * Return the value of the source for a given point and time.
   */
  double value(dealii::Point<dim> const &point) const final;

  /**
   * Same function as above but it uses vectorized data.
   */
  dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &points)
      const final;

  dealii::BoundingBox<dim>
  get_bounding_box(double const time, double const scaling_factor) const final;

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
