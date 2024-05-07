/* Copyright (c) 2016 - 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ScanPath.hh>
#include <utils.hh>

#include <deal.II/base/memory_space.h>

#include <fstream>

namespace adamantine
{

template <typename MemorySpace>
std::vector<ScanPathSegment>
ScanPath<MemorySpace>::extract_scan_paths(std::string scan_path_file,
                                          std::string file_format)
{
  // Parse the scan path
  wait_for_file(scan_path_file,
                "Waiting for scan path file: " + scan_path_file);

  if (file_format == "segment")
  {
    return load_segment_scan_path(scan_path_file);
  }
  else if (file_format == "event_series")
  {
    return load_event_series_scan_path(scan_path_file);
  }
  else
  {
    ASSERT_THROW(false, "Error: Format of scan path file not recognized.");
  }

  return {};
}

template <typename MemorySpace>
std::vector<ScanPathSegment>
ScanPath<MemorySpace>::load_segment_scan_path(std::string scan_path_file)
{
  std::vector<ScanPathSegment> segment_list;

  std::ifstream file;
  file.open(scan_path_file);
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
      ASSERT_THROW(segment_list.size() > 0,
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
    segment.end_point(0) = std::stod(split_line[1]);
    segment.end_point(1) = std::stod(split_line[2]);
    segment.end_point(2) = std::stod(split_line[3]);

    // Set the power modifier
    segment.power_modifier = std::stod(split_line[4]);

    // Set the velocity and end time
    if (segment_type == ScanPathSegmentType::point)
    {
      if (segment_list.size() > 0)
      {
        segment.end_time =
            segment_list.back().end_time + std::stod(split_line[5]);
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
          segment.end_point.distance(segment_list.back().end_point);
      segment.end_time =
          segment_list.back().end_time + std::abs(line_length / velocity);
    }
    segment_list.push_back(segment);
    data_index++;
  }
  file.close();
  return segment_list;
}

template <typename MemorySpace>
std::vector<ScanPathSegment>
ScanPath<MemorySpace>::load_event_series_scan_path(std::string scan_path_file)
{
  std::vector<ScanPathSegment> segment_list;

  std::ifstream file;
  file.open(scan_path_file);
  std::string line;

  double last_power = 0.0;
  while (getline(file, line))
  {
    // For an event series the first segment is a ScanPathSegment point, then
    // the rest are ScanPathSegment lines
    ScanPathSegment segment;

    std::vector<std::string> split_line;
    boost::split(split_line, line, boost::is_any_of(" ,,"),
                 boost::token_compress_on);

    // Set the segment end time
    segment.end_time = std::stod(split_line[0]);

    // Set the segment end position
    segment.end_point(0) = std::stod(split_line[1]);
    segment.end_point(1) = std::stod(split_line[2]);
    segment.end_point(2) = std::stod(split_line[3]);

    // Set the power modifier
    segment.power_modifier = last_power;
    last_power = std::stod(split_line[4]);

    segment_list.push_back(segment);
  }
  return segment_list;
}

template <typename MemorySpace>
void ScanPath<MemorySpace>::update_current_segment_info(
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

template <typename MemorySpace>
dealii::Point<3> ScanPath<MemorySpace>::value(double const &time) const
{
  // If the current time is after the scan path data is over, return a point
  // that is (presumably) out of the domain.
  if (time > _segment_list[_segment_list.size() - 1].end_time)
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

template <typename MemorySpace>
double ScanPath<MemorySpace>::get_power_modifier(double const &time) const
{
  // If the current time is after the scan path data is over, set the power to
  // zero.
  if (time > _segment_list[_segment_list.size() - 1].end_time)
    return 0.0;

  // Get to the correct segment
  dealii::Point<3> segment_start_point;
  double segment_start_time = 0.0;
  update_current_segment_info(time, segment_start_point, segment_start_time);

  return _segment_list[_current_segment].power_modifier;
}

template <typename MemorySpace>
std::vector<ScanPathSegment> ScanPath<MemorySpace>::get_segment_list() const
{
  return {&_segment_list[0], &_segment_list[0] + _segment_list.size()};
}

} // namespace adamantine

template class adamantine::ScanPath<dealii::MemorySpace::Default>;
template class adamantine::ScanPath<dealii::MemorySpace::Host>;
