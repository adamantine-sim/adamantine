/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef SCAN_PATH_TEMPLATES_HH
#define SCAN_PATH_TEMPLATES_HH

#include <ScanPath.hh>
#include <instantiation.hh>
#include <utils.hh>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <cstdlib>

using std::pow;

namespace adamantine
{

ScanPath::ScanPath(std::string scan_path_file)
    : dealii::Function<1>(), _current_segment(0), _saved_segment(-1)
{
  // General initializations
  _current_time[0] = -1.;

  // Parse the scan path
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
        if (_segment_list.size() == 0)
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

      // Set the power modifier
      segment.power_modifier = std::stod(split_line[4]);

      // Set the velocity and end time
      if (segment_type == ScanPathSegmentType::point)
      {
        if (_segment_list.size() > 0)
        {
          segment.end_time =
              _segment_list.back().end_time + std::stod(split_line[5]);
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
            segment.end_point.distance(_segment_list.back().end_point);
        segment.end_time =
            _segment_list.back().end_time + std::abs(line_length / velocity);
      }
      _segment_list.push_back(segment);
    }
    line_index++;
  }
  file.close();
}

void ScanPath::update_current_segment_info(
    double time, dealii::Point<2> &segment_start_point,
    double &segment_start_time) const
{
  // Get to the correct segment (assumes that the current time is never before
  // the current segment starts)
  if (time > _segment_list[_current_segment].end_time)
  {
    while (time > _segment_list[_current_segment].end_time)
    {
      ++_current_segment;
    }
  }
  // Update the start position and time for the current segment
  if (_current_segment > 0)
  {
    segment_start_time = _segment_list[_current_segment - 1].end_time;
    segment_start_point = _segment_list[_current_segment - 1].end_point;
  }
  else
  {
    segment_start_time = 0.0;
    segment_start_point = _segment_list[_current_segment].end_point;
  }
}

double ScanPath::value(dealii::Point<1> const &time,
                       unsigned int const component) const
{
  // The global coordinate system is (x,z,y), while the scan path coordinate
  // system is (x,y,z). I need to convert the global coordinate number to the
  // scan path coordinates.
  int beam_component = 0;
  if (component == 2)
  {
    beam_component = 1;
  }
  ASSERT_THROW(component != 1, "Invalid BeamCenter component.");

  // Get to the correct segment (assumes that the current time is never
  // before the current segment starts)
  dealii::Point<2> segment_start_point;
  double segment_start_time = 0.0;
  _current_time[0] = time[0];
  update_current_segment_info(time[0], segment_start_point, segment_start_time);

  // Calculate the position in the direction given by "component"
  double position =
      segment_start_point[beam_component] +
      (_segment_list[_current_segment].end_point[beam_component] -
       segment_start_point[beam_component]) /
          (_segment_list[_current_segment].end_time - segment_start_time) *
          (time[0] - segment_start_time);

  return position;
}

double ScanPath::get_power_modifier(dealii::Point<1> const &time) const
{
  // Get to the correct segment (assumes that the current time is never
  // before the current segment starts)
  dealii::Point<2> segment_start_point;
  double segment_start_time = 0.0;
  _current_time[0] = time[0];
  update_current_segment_info(time[0], segment_start_point, segment_start_time);

  return _segment_list[_current_segment].power_modifier;
}

void ScanPath::rewind_time()
{
  _current_segment = _saved_segment;
  _current_time = _saved_time;
}

void ScanPath::save_time()
{
  _saved_segment = _current_segment;
  _saved_time = _current_time;
}
} // namespace adamantine

#endif
