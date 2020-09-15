/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCE_HH
#define HEAT_SOURCE_HH

#include <utils.hh>

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
 * This enum distinguishes between the two types of scan path segments.
 */
enum class ScanPathSegmentType
{
  line,
  point
};

/**
 * This structure stores the relevant information for a single segment. The scan
 * path input file distingishes between spots and lines, but when we know the
 * end time and end location, spots become lines with a start point equal to its
 * end point. Everything one needs can be determined from these three quantities
 * (and the segment info from the preceding segment) but in the future it might
 * be worth adding in some redundant information like start time/point and
 * velocity.
 */
struct ScanPathSegment
{
  double end_time;            // Unit: seconds
  double power_modifier;      // Dimensionless
  dealii::Point<2> end_point; // Unit: m (NOTE: converted from mm in the file)
};

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

namespace internal
{
/**
 * This class calculates the position of the center of a heat source. It also
 * gives the power modifier for the current segment.
 */
class BeamCenter
{
public:
  BeamCenter();

  void load_segment_list(std::vector<ScanPathSegment> segment_list);

  double value(dealii::Point<1> const &time,
               unsigned int const component = 0) const;

  /**
   * Return the power coefficient for the current segment
   */
  double get_power_modifier(dealii::Point<1> const &time) const;

  void rewind_time();

  void save_time();

private:
  mutable unsigned int _current_segment;
  unsigned int _saved_segment;
  mutable dealii::Point<1> _current_time;
  dealii::Point<1> _saved_time;
  std::vector<ScanPathSegment> _segment_list;

  void update_current_segment_info(double time,
                                   dealii::Point<2> &segment_start_point,
                                   double &segment_start_time) const;
};
} // namespace internal

/**
 * Forward declaration of the tester friend class to HeatSource.
 */
class HeatSourceTester;

/**
 * This class describes the evolution of a Goldak heat source.
 */
template <int dim>
class HeatSource : public dealii::Function<dim>
{
  friend class HeatSourceTester;

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

  /**
   * This function reads the scan path file and populates the vector of
   * ScanPathSegments.
   */
  void parse_scan_path(std::string scan_path_file);

  internal::BeamCenter _beam_center;
};

template <int dim>
inline void HeatSource<dim>::set_max_height(double height)
{
  _max_height = height;
}
} // namespace adamantine

#endif
