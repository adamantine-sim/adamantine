/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <experimental_data.hh>
#include <utils.hh>

#include <deal.II/arborx/bvh.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>

#include <boost/filesystem.hpp>

#include <cstdlib>
#include <fstream>
#include <regex>

namespace adamantine
{
template <int dim>
std::vector<PointsValues<dim>> read_experimental_data_point_cloud(
    MPI_Comm const &communicator,
    boost::property_tree::ptree const &experiment_database)
{
  // Format of the file names: the format is pretty arbitrary, #frame and
  // #camera are replaced by the frame and the camera number.
  // PropertyTreeInput experiment.file
  std::string data_filename = experiment_database.get<std::string>("file");
  // PropertyTreeInput experiment.first_frame
  unsigned int first_frame = experiment_database.get("first_frame", 0);
  // PropertyTreeInput experiment.last_frame
  unsigned int last_frame = experiment_database.get<unsigned int>("last_frame");
  // PropertyTreeInput experiment.first_camera_id
  unsigned int first_camera_id =
      experiment_database.get<unsigned int>("first_camera_id");
  // PropertyTreeInput experiment.last_camera_id
  unsigned int last_camera_id = experiment_database.get<int>("last_camera_id");
  // PropertyTreeInput experiment.data_columns
  std::string data_columns =
      experiment_database.get<std::string>("data_columns");

  std::vector<PointsValues<dim>> points_values_all_frames(last_frame + 1 -
                                                          first_frame);
  for (unsigned int frame = first_frame; frame < last_frame + 1; ++frame)
  {
    PointsValues<dim> points_values;
    for (unsigned int camera_id = first_camera_id;
         camera_id < last_camera_id + 1; ++camera_id)
    {
      // Use regex to get the next file to read
      std::regex camera_regex("#camera");
      std::regex frame_regex("#frame");
      auto regex_filename =
          std::regex_replace((std::regex_replace(data_filename, camera_regex,
                                                 std::to_string(camera_id))),
                             frame_regex, std::to_string(frame));
      ASSERT(boost::filesystem::exists(regex_filename),
             "The file " + regex_filename + " does not exist.");
      std::string filename("data_" + std::to_string(frame) + "_" +
                           std::to_string(camera_id) + ".csv");

      // Use bash to create a new file that only contains the columns that we
      // care about. For large files this divides by four the time to parse the
      // files. It also simplifies reading the files. Only rank zero renames
      // the file.
      if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        std::string cut_command("cut -d, -f" + data_columns + " " +
                                regex_filename + " > " + filename);
        std::system(cut_command.c_str());
      }

      // Wait for rank zero before reading the file.
      MPI_Barrier(communicator);

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

        points_values.points.push_back(point);
        points_values.values.push_back(value);
      }

      // Wait for every rank to be done reading the temporary stripped file and
      // then remove it.
      MPI_Barrier(communicator);
      if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        std::string rm_command("rm " + filename);
        std::system(rm_command.c_str());
      }
    }
    points_values_all_frames[frame - first_frame] = points_values;
  }

  return points_values_all_frames;
}

template <int dim>
std::pair<std::vector<int>, std::vector<int>> set_with_experimental_data(
    PointsValues<dim> const &points_values,
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature)
{
  // First we need to get all the supports points and the associated dof
  // indices
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> indices_points;
  dealii::DoFTools::map_dofs_to_support_points(
      dealii::StaticMappingQ1<dim>::mapping, dof_handler, indices_points);
  // Change the format to something that can be used by ArborX
  std::vector<dealii::types::global_dof_index> dof_indices(
      indices_points.size());
  std::vector<dealii::Point<dim>> support_points(indices_points.size());
  unsigned int pos = 0;
  for (auto map_it = indices_points.begin(); map_it != indices_points.end();
       ++map_it, ++pos)
  {
    dof_indices[pos] = map_it->first;
    support_points[pos] = map_it->second;
  }

  // Perform the search
  dealii::ArborXWrappers::BVH bvh(support_points);
  dealii::ArborXWrappers::PointNearestPredicate pt_nearest(points_values.points,
                                                           1);
  auto [indices, offset] = bvh.query(pt_nearest);

  // Fill in the temperature
  unsigned int const n_queries = points_values.points.size();
  for (unsigned int i = 0; i < n_queries; ++i)
  {
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      temperature[dof_indices[indices[j]]] = points_values.values[i];
    }
  }

  temperature.compress(dealii::VectorOperation::insert);

  return {indices, offset};
}
} // namespace adamantine

//-------------------- Explicit Instantiations --------------------//
namespace adamantine
{
template std::vector<PointsValues<2>> read_experimental_data_point_cloud(
    MPI_Comm const &communicator,
    boost::property_tree::ptree const &experiment_database);
template std::vector<PointsValues<3>> read_experimental_data_point_cloud(
    MPI_Comm const &communicator,
    boost::property_tree::ptree const &experiment_database);

template std::pair<std::vector<int>, std::vector<int>>
set_with_experimental_data(
    PointsValues<2> const &points_values,
    dealii::DoFHandler<2> const &dof_handler,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature);
template std::pair<std::vector<int>, std::vector<int>>
set_with_experimental_data(
    PointsValues<3> const &points_values,
    dealii::DoFHandler<3> const &dof_handler,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature);
} // namespace adamantine
