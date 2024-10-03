/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef SCAN_PATH_HH
#define SCAN_PATH_HH

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <boost/property_tree/ptree.hpp>

#include <filesystem>
#include <limits>
#include <string>
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
  double end_time =
      std::numeric_limits<double>::signaling_NaN(); // Unit: seconds
  double power_modifier =
      std::numeric_limits<double>::signaling_NaN(); // Dimensionless
  dealii::Point<3> end_point;                       // Unit: m
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
   * Default constructor. This creates an empty scan path with no segment.
   */
  ScanPath() = default;

  /**
   * Construtor.
   * \param[in] scan_path_file is the name of the text file containing the scan
   * path
   * \param[in] file_format is the format of the scan path file
   * \param[in] optional units property tree
   */
  ScanPath(std::string const &scan_path_file, std::string const &file_format,
           boost::optional<boost::property_tree::ptree const &> const
               &units_optional_database);

  /**
   * Calculate the location of the scan path at a given time for a single
   * coordinate.
   */
  dealii::Point<3> value(double const &time) const;

  /**
   * Return the power coefficient for the current segment
   */
  double get_power_modifier(double const &time) const;

  /**
   * Return the scan path's list of segments
   */
  std::vector<ScanPathSegment> get_segment_list() const;

  /**
   * Read the scan path file and update the list of segments.
   */
  void read_file();

  /**
   * Return true if we reach the end of the scan path.
   */
  bool is_finished() const;

private:
  /**
   * Method to load a "segment" scan path file
   */
  void load_segment_scan_path();

  /**
   * Method to load an "event series" scan path file
   */
  void load_event_series_scan_path();

  /**
   * Method to determine the current segment, its start point, and start time.
   */
  void update_current_segment_info(double time,
                                   dealii::Point<3> &segment_start_point,
                                   double &segment_start_time) const;

  /**
   * Flag is true if we have reached the end of _scan_path_file.
   */
  bool _scan_path_end = false;
  /**
   * Scaling factor for the distance.
   */
  double _distance_scaling = 1.;
  /**
   * Scaling factor for the velocity.
   */
  double _velocity_scaling = 1.;
  /**
   * File name of the scan path
   */
  std::string _scan_path_file;
  /**
   * Format of the scan path file, either segment of event_series.
   */
  std::string _file_format;
  /**
   * Time the last time _scan_path_file was updated.
   */
  std::filesystem::file_time_type _last_write_time;
  /**
   * The list of information about each segment in the scan path.
   */
  std::vector<ScanPathSegment> _segment_list;
  /**
   * The index of the current segment in the scan path.
   */
  mutable unsigned int _current_segment = 0;
};
} // namespace adamantine

#endif
