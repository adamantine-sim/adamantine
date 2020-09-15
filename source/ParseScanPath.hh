/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef SCANPATHPARSER_HH
#define SCANPATHPARSER_HH

#include <utils.hh>

#include <deal.II/base/point.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <istream>
#include <vector>

namespace adamantine
{

/**
 * This enum distinguishes between the two types of scan path segments.
 */
 /*
enum class ScanPathSegmentType
{
  line,
  point
};
*/
/**
 * This structure stores the relevant information for a single segment. The scan
 * path input file distingishes between spots and lines, but when we know the
 * end time and end location, spots become lines with a start point equal to its
 * end point. Everything one needs can be determined from these three quantities
 * (and the segment info from the preceding segment) but in the future it might
 * be worth adding in some redundant information like start time/point and
 * velocity.
 */
 /*
struct ScanPathSegment
{
  double end_time;            // Unit: seconds
  double power_modifier;      // Dimensionless
  dealii::Point<3> end_point; // Unit: m (NOTE: converted from mm in the file)
};
*/
/**
 * This function reads the scan path file and creates a vector of
 * ScanPathSegments.
 */
 /*
std::vector<ScanPathSegment> ParseScanPath(std::string scan_path_file)
{

  std::vector<ScanPathSegment> segments;

  // Open the file
  ASSERT_THROW(boost::filesystem::exists(scan_path_file),
               "The file " + scan_path_file + " does not exist.");
  std::ifstream file;
  file.open(scan_path_file);
  std::string line;
  int line_index = 0;
  while (getline(file, line))
  {
    std::cout << line << std::endl;

    // Skip the header
    if (line_index > 2)
    {
      std::vector<std::string> split_line;
      boost::split(split_line, line, boost::is_any_of(" "),
                   boost::token_compress_on);
      ScanPathSegment segment;

      // Set the segment type
      ScanPathSegmentType segment_type;
      if (split_line[0] == "0")
      {
        if (segments.size() == 0)
        {
          std::string message =
              "Error: Scan paths must begin with a 'point' segment.";
          throw std::runtime_error(message);
        }
        segment_type = ScanPathSegmentType::line;
      }
      else if (split_line[0] == "1")
      {
        segment_type = ScanPathSegmentType::point;
      }
      else
      {
        std::string message = "Error: Mode type in scan path file line " +
                              std::to_string(line_index) + "not recognized.";
        throw std::runtime_error(message);
      }

      // Set the segment end position
      segment.end_point(0) = std::stod(split_line[1]);
      segment.end_point(1) = std::stod(split_line[2]);
      segment.end_point(2) = std::stod(split_line[3]);

      // Set the power modifier
      segment.power_modifier = std::stod(split_line[4]);

      // Set the velocity and end time
      if (segment_type == ScanPathSegmentType::point)
      {
        if (segments.size() > 0)
        {
          segment.end_time =
              segments.back().end_time + std::stod(split_line[5]);
        }
        else
        {
          segment.end_time = std::stod(split_line[5]);
        }
      }
      else
      {
        double velocity = std::stod(split_line[5]);
        double line_length =
            segment.end_point.distance(segments.back().end_point);
        segment.end_time =
            segments.back().end_time + std::abs(line_length / velocity);
      }
      segments.push_back(segment);
    }
    line_index++;
  }
  file.close();

  return segments;
};
*/
} // namespace adamantine

#endif
