/* Copyright (c) 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <PointCloud.hh>
#include <instantiation.hh>

#include <boost/filesystem.hpp>

#include <fstream>
#include <regex>

namespace adamantine
{
template <int dim>
PointCloud<dim>::PointCloud(
    boost::property_tree::ptree const &experiment_database)
{
  // Format of the file names: the format is pretty arbitrary, #frame and
  // #camera are replaced by the frame and the camera number.
  // PropertyTreeInput experiment.file
  _data_filename = experiment_database.get<std::string>("file");
  // PropertyTreeInput experiment.first_frame
  _next_frame = experiment_database.get("first_frame", 0);
  // PropertyTreeInput experiment.first_camera_id
  _first_camera_id = experiment_database.get<unsigned int>("first_camera_id");
  // PropertyTreeInput experiment.last_camera_id
  _last_camera_id = experiment_database.get<int>("last_camera_id");
}

template <int dim>
unsigned int PointCloud<dim>::read_next_frame()
{
  _points_values_current_frame.points.clear();
  _points_values_current_frame.values.clear();
  for (unsigned int camera_id = _first_camera_id;
       camera_id < _last_camera_id + 1; ++camera_id)
  {
    // Use regex to get the next file to read
    std::regex camera_regex("#camera");
    std::regex frame_regex("#frame");
    auto filename =
        std::regex_replace((std::regex_replace(_data_filename, camera_regex,
                                               std::to_string(camera_id))),
                           frame_regex, std::to_string(_next_frame));
    unsigned int counter = 1;
    while (!boost::filesystem::exists(filename))
    {
      // Spin loop waiting for the file to appear (message printed if counter
      // overflows)
      if (counter == 0)
        std::cout << "Waiting for the next frame" << std::endl;
      ++counter;
    }

    // Read and parse the file
    std::ifstream file;
    file.open(filename);
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
      std::size_t pos = 0;
      std::size_t last_pos = 0;
      std::size_t line_length = line.length();
      unsigned int i = 0;
      dealii::Point<dim> point;
      double value = 0.;
      while (last_pos < line_length + 1)
      {
        pos = line.find_first_of(",", last_pos);
        // If no comma was found that we read until the end of the file
        if (pos == std::string::npos)
        {
          pos = line_length;
        }

        if (pos != last_pos)
        {
          char *end = line.data() + pos;
          if (i < dim)
          {
            point[i] = std::strtod(line.data() + last_pos, &end);
          }
          else
          {
            value = std::strtod(line.data() + last_pos, &end);
          }

          ++i;
        }

        last_pos = pos + 1;
      }

      _points_values_current_frame.points.push_back(point);
      _points_values_current_frame.values.push_back(value);
    }
  }

  return _next_frame++;
}

template <int dim>
PointsValues<dim> PointCloud<dim>::get_points_values()
{
  return _points_values_current_frame;
}

} // namespace adamantine

INSTANTIATE_DIM(PointCloud);
