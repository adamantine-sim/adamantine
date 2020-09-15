/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCE_HH
#define HEAT_SOURCE_HH

#include <utils.hh>
#include <ScanPath.hh>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>

#include <iostream>
#include <istream>
#include <vector>

namespace adamantine
{
/**
 * This structure stores all the physical properties necessary to define an
 * heat source.
 */
struct HeatSourceProperties
{
public:
  /**
   * Absolute penetration of the electron beam into the material where 99% of
   * the beam energy is absorbed.
   */
  double depth;
  /**
   * Energy conversion efficiency on the surface.
   */
  double absorption_efficiency;

  /**
   * Square of the beam radius.
   */
  double radius_squared;
  /**
   * Maximum power of the beam.
   */
  double max_power;
};

/**
 * This class describes the evolution of a Goldak heat source.
 */
template <int dim>
class HeatSource : public dealii::Function<dim>
{
public:
  /**
   * Constructor.
   * \param[in] database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   *   - <B>input_file</B>: name of the file that contains the scan path
   *     segments
   */
  HeatSource(boost::property_tree::ptree const &database);

  /**
   * Set the maximum height of the domain. This is the height at which the
   * electron beam penetrate the material.
   */
  void set_max_height(double height);

  /**
   * Compute the heat source at a given point at the current time.
   */
  double value(dealii::Point<dim> const &point,
               unsigned int const component = 0) const override;

  /**
   * Reset the current time and the position to the last saved state.
   */
  void rewind_time();

  /**
   * Save the current time and the position in the list of successive positions
   * of the beam.
   */
  void save_time();

private:
  /**
   * Height of the domain.
   */
  double _max_height;
  /**
   * Structure of the physical properties of the heat source.
   */
  HeatSourceProperties _beam;

  /**
   * The list of segments in the scan path.
   */
  std::vector<ScanPathSegment> _segment_list;

  ScanPath _scan_path;
};

template <int dim>
inline void HeatSource<dim>::set_max_height(double height)
{
  _max_height = height;
}
} // namespace adamantine

#endif
