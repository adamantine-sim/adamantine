/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCE_TEMPLATES_HH
#define HEAT_SOURCE_TEMPLATES_HH

#include <HeatSource.hh>
#include <instantiation.hh>
#include <utils.hh>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <cstdlib>

using std::pow;

namespace adamantine
{

namespace internal
{
BeamCenter::BeamCenter() : _current_segment(0), _saved_segment(-1)
{
  _current_time[0] = -1.;
}

void BeamCenter::load_segment_list(std::vector<ScanPathSegment> segment_list)
{
  _segment_list = segment_list;
}

void BeamCenter::update_current_segment_info(
    double time, dealii::Point<2> &segment_start_point,
    double &segment_start_time) const
{
  // Get to the correct segment (assumes that the current time is never before
  // the current segment starts)
  // dealii::Point<2> segment_start_point;
  // double segment_start_time = 0.0;
  //_current_time[0] = time[0];
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

double BeamCenter::value(dealii::Point<1> const &time,
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

double BeamCenter::get_power_modifier(dealii::Point<1> const &time) const
{
  // Get to the correct segment (assumes that the current time is never
  // before the current segment starts)
  dealii::Point<2> segment_start_point;
  double segment_start_time = 0.0;
  _current_time[0] = time[0];
  update_current_segment_info(time[0], segment_start_point, segment_start_time);

  return _segment_list[_current_segment].power_modifier;
}

void BeamCenter::rewind_time()
{
  _current_segment = _saved_segment;
  _current_time = _saved_time;
}

void BeamCenter::save_time()
{
  _saved_segment = _current_segment;
  _saved_time = _current_time;
} // namespace internal
} // namespace internal

template <int dim>
HeatSource<dim>::HeatSource(boost::property_tree::ptree const &database)
    : dealii::Function<dim>(), _max_height(0.)
{
  // Set the properties of the electron beam.
  _beam.depth = database.get<double>("depth");
  _beam.absorption_efficiency = database.get<double>("absorption_efficiency");
  _beam.radius_squared = pow(database.get("diameter", 2e-3) / 2.0, 2);
  boost::optional<double> max_power =
      database.get_optional<double>("max_power");
  if (max_power)
    _beam.max_power = max_power.get();
  else
  {
    std::string message =
        "When using HeatSource, the max power is not optional.";
    throw std::runtime_error(message);
  }

  // Parse the scan path
  std::string scan_path_file = "scan_path.txt";
  parse_scan_path(scan_path_file);

  // Initialize the beam center object
  _beam_center.load_segment_list(_segment_list);
}

template <int dim>
void HeatSource<dim>::parse_scan_path(std::string scan_path_file)
{

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

template <int dim>
void HeatSource<dim>::rewind_time()
{
}

template <int dim>
void HeatSource<dim>::save_time()
{
}

template <int dim>
double HeatSource<dim>::value(dealii::Point<dim> const &point,
                              unsigned int const /*component*/) const
{

  double const z = point[1] - _max_height;
  if ((z + _beam.depth) < 0.)
  {
    return 0.;
  }
  else
  {
    dealii::Point<1> time;
    time[0] = this->get_time();
    double const beam_center_x = _beam_center.value(time, 0);
    double xpy_squared = pow(point[0] - beam_center_x, 2);
    if (dim == 3)
    {
      double const beam_center_y = _beam_center.value(time, 2);
      xpy_squared += pow(point[2] - beam_center_y, 2);
    }
    double segment_power_modifier = _beam_center.get_power_modifier(time);
    double pi_over_3_to_1p5 = pow(dealii::numbers::PI / 3.0, 1.5);

    // Goldak heat source
    double heat_source =
        -2.0 * _beam.absorption_efficiency * _beam.max_power *
        segment_power_modifier /
        (_beam.radius_squared * _beam.depth * pi_over_3_to_1p5) *
        std::exp(-3.0 * xpy_squared / _beam.radius_squared +
                 -3.0 * pow(z / _beam.depth, 2));

    return heat_source;
  }
}
} // namespace adamantine

INSTANTIATE_DIM(HeatSource)

#endif
