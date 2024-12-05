/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ScanPath.hh>
#include <types.hh>
#include <utils.hh>

#include <boost/algorithm/string.hpp>

#include <fstream>

namespace adamantine
{
ScanPath::ScanPath(std::string const &scan_path_file,
                   std::string const &file_format,
                   boost::optional<boost::property_tree::ptree const &> const
                       &units_optional_database)
    : _scan_path_file(scan_path_file), _file_format(file_format)
{
  // Get the scaling factor of the different units, if provided. If the property
  // tree is not given, we assume that all the scaling factors are equal to one.
  if (units_optional_database)
  {
    auto database = units_optional_database.get();
    // PropertyTreeInput units.heat_source.scan_path
    std::string unit = database.get("heat_source.scan_path", "meter");
    _distance_scaling = g_unit_scaling_factor[unit];
    // PropertyTreeInput units.heat_source.velocity
    unit = database.get("heat_source.velocity", "meter/second");
    _velocity_scaling = g_unit_scaling_factor[unit];
  }

  ASSERT_THROW((_file_format == "segment") || (_file_format == "event_series"),
               "Error: Format of scan path file not recognized.");

  wait_for_file(_scan_path_file,
                "Waiting for scan path file: " + _scan_path_file);

  read_file();
}

void ScanPath::read_file()
{
  wait_for_file_to_update(_scan_path_file, "Waiting for " + _scan_path_file,
                          _last_write_time);

  if (_file_format == "segment")
  {
    load_segment_scan_path();
  }
  else
  {
    load_event_series_scan_path();
  }
}

void ScanPath::load_segment_scan_path()
{
  _segment_list.clear();
  std::ifstream file;
  file.open(_scan_path_file);
  std::string line;
  unsigned int data_index = 0;
  // Skip first line
  getline(file, line);
  // Read the number of path segments
  getline(file, line);
  unsigned int n_segments = std::stoi(line);
  // Skip third line
  getline(file, line);
  // Read file as long as there are lines to read or we reached the number of
  // segments to read, whichever comes first
  while ((data_index < n_segments) && (getline(file, line)))
  {
    // If we reach the end of the scan path, we stop reading the file.
    if (line.find("SCAN_PATH_END") != std::string::npos)
    {
      _scan_path_end = true;
      break;
    }

    std::vector<std::string> split_line;
    boost::split(split_line, line, boost::is_any_of(" "),
                 boost::token_compress_on);
    ScanPathSegment segment;

    // Set the segment type
    ScanPathSegmentType segment_type = ScanPathSegmentType::line;
    if (split_line[0] == "0")
    {
      // Check to make sure the segment isn't the first, if it is, throw an
      // exception (the first segment must be a point in the spec).
      ASSERT_THROW(_segment_list.size() > 0,
                   "Error: Scan paths must begin with a 'point' segment.");
    }
    else if (split_line[0] == "1")
    {
      segment_type = ScanPathSegmentType::point;
    }
    else
    {
      ASSERT_THROW(false, "Error: Mode type in scan path file line " +
                              std::to_string(data_index + 4) +
                              " not recognized.");
    }

    // Set the segment end position
    segment.end_point(0) = std::stod(split_line[1]) * _distance_scaling;
    segment.end_point(1) = std::stod(split_line[2]) * _distance_scaling;
    segment.end_point(2) = std::stod(split_line[3]) * _distance_scaling;

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
      double velocity = std::stod(split_line[5]) * _velocity_scaling;
      double line_length =
          segment.end_point.distance(_segment_list.back().end_point);
      segment.end_time =
          _segment_list.back().end_time + std::abs(line_length / velocity);
    }
    _segment_list.push_back(segment);
    data_index++;
  }
  file.close();
}

void ScanPath::load_event_series_scan_path()
{
  _segment_list.clear();
  std::ifstream file;
  file.open(_scan_path_file);
  std::string line;

  double last_power = 0.0;
  while (getline(file, line))
  {
    if (line == "")
      continue;

    // If we reach the end of the scan path, we stop reading the file.
    if (line.find("SCAN_PATH_END") != std::string::npos)
    {
      _scan_path_end = true;
      break;
    }

    // For an event series the first segment is a ScanPathSegment point, then
    // the rest are ScanPathSegment lines
    ScanPathSegment segment;

    std::vector<std::string> split_line;
    boost::split(split_line, line, boost::is_any_of(" ,,"),
                 boost::token_compress_on);

    // Set the segment end time
    segment.end_time = std::stod(split_line[0]);

    // Set the segment end position
    segment.end_point(0) = std::stod(split_line[1]) * _distance_scaling;
    segment.end_point(1) = std::stod(split_line[2]) * _distance_scaling;
    segment.end_point(2) = std::stod(split_line[3]) * _distance_scaling;

    // Set the power modifier
    segment.power_modifier = last_power;
    last_power = std::stod(split_line[4]);

    _segment_list.push_back(segment);
  }
}

void ScanPath::update_current_segment_info(
    double time, dealii::Point<3> &segment_start_point,
    double &segment_start_time) const
{
  // Get to the correct segment
  _current_segment = 0;
  while (time > _segment_list[_current_segment].end_time)
  {
    ++_current_segment;
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

dealii::Point<3> ScanPath::value(double const &time) const
{
  // If the current time is after the scan path data is over, return a point
  // that is (presumably) out of the domain.
  if (time > _segment_list.back().end_time)
  {
    dealii::Point<3> out_of_domain_point(std::numeric_limits<double>::lowest(),
                                         std::numeric_limits<double>::lowest(),
                                         std::numeric_limits<double>::lowest());
    return out_of_domain_point;
  }

  // Get to the correct segment
  dealii::Point<3> segment_start_point;
  double segment_start_time = 0.0;
  update_current_segment_info(time, segment_start_point, segment_start_time);

  // Calculate the position in the direction given by "component"
  dealii::Point<3> position =
      segment_start_point +
      (_segment_list[_current_segment].end_point - segment_start_point) /
          (_segment_list[_current_segment].end_time - segment_start_time) *
          (time - segment_start_time);

  return position;
}

double ScanPath::get_power_modifier(double const &time) const
{
  // If the current time is after the scan path data is over, set the power to
  // zero.
  if (time > _segment_list.back().end_time)
    return 0.0;

  // Get to the correct segment
  dealii::Point<3> segment_start_point;
  double segment_start_time = 0.0;
  update_current_segment_info(time, segment_start_point, segment_start_time);

  return _segment_list[_current_segment].power_modifier;
}

std::vector<ScanPathSegment> ScanPath::get_segment_list() const
{
  return _segment_list;
}

bool ScanPath::is_finished() const { return _scan_path_end; }

} // namespace adamantine
