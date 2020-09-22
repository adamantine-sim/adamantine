/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef SCAN_PATH_HH
#define SCAN_PATH_HH

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
  dealii::Point<3> end_point; // Unit: m (NOTE: converted from mm in the file)
};

/**
 * Forward declaration of the tester friend class to ScanPath.
 */
class ScanPathTester;

/**
 * This class calculates the position of the center of a heat source. It also
 * gives the power modifier for the current segment. It reads in the scan path
 * from a text file.
 */
class ScanPath
{
  friend class ScanPathTester;

public:
  /**
   * Construtor.
   * \param[in] scan_path_file is the name of the text file containing the scan
   * path
   */
  ScanPath(std::string scan_path_file);

  /**
   * Calculates the location of the scan path at a given time for a single
   * coordinate.
   */
  dealii::Point<3> value(double const &time);

  /**
   * Returns the power coefficient for the current segment
   */
  double get_power_modifier(double const &time);

private:
  /**
   * The list of information about each segment in the scan path.
   */
  std::vector<ScanPathSegment> _segment_list;

  /**
   * The index of the current segment in the scan path.
   */
  unsigned int _current_segment;

  /**
   * The current time.
   */
  double _current_time;

  /**
   * Method to determine the current segment, its start point, and start time.
   */
  void update_current_segment_info(double time,
                                   dealii::Point<3> &segment_start_point,
                                   double &segment_start_time);
};
} // namespace adamantine

#endif
