/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <experimental_data_utils.hh>
#include <utils.hh>

#include <deal.II/arborx/distributed_tree.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/hp/fe_values.h>

#include <boost/algorithm/string.hpp>

#include <fstream>
#include <unordered_set>

namespace adamantine
{
template <int dim>
std::pair<std::vector<dealii::types::global_dof_index>,
          std::vector<dealii::Point<dim>>>
get_dof_to_support_mapping(dealii::DoFHandler<dim> const &dof_handler)
{
  std::vector<dealii::types::global_dof_index> dof_indices;
  std::vector<dealii::Point<dim>> support_points;
  std::unordered_set<dealii::types::global_dof_index> visited_dof_indices;

  // Manually do what dealii::DoFTools::map_dofs_to_support_points does, since
  // that doesn't currently work with FE_Nothing. The issue has been fixed in
  // deal.II 9.7
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe(0);

  dealii::FEValues<dim, dim> fe_values(fe, fe.get_unit_support_points(),
                                       dealii::update_quadrature_points);

  std::vector<dealii::types::global_dof_index> local_dof_indices(
      fe.n_dofs_per_cell());
  auto locally_owned_dofs = dof_handler.locally_owned_dofs();

  for (auto const &cell :
       dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::ActiveFEIndexEqualTo(
                                    0, /* locally owned */ true)))
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    const std::vector<dealii::Point<dim>> &points =
        fe_values.get_quadrature_points();
    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    {
      // Skip duplicate points like vertices and indices that correspond to
      // ghosted elements
      if ((visited_dof_indices.count(local_dof_indices[i]) == 0) &&
          (locally_owned_dofs.is_element(local_dof_indices[i])))
      {
        dof_indices.push_back(local_dof_indices[i]);
        support_points.push_back(points[i]);
        visited_dof_indices.insert(local_dof_indices[i]);
      }
    }
  }

  return {dof_indices, support_points};
}

template <int dim>
std::pair<std::vector<int>, std::vector<int>>
get_expt_to_dof_mapping(PointsValues<dim> const &points_values,
                        dealii::DoFHandler<dim> const &dof_handler)
{
  auto [dof_indices, support_points] = get_dof_to_support_mapping(dof_handler);

  auto communicator = dof_handler.get_communicator();
  int my_rank = dealii::Utilities::MPI::this_mpi_process(communicator);

  // Perform the search
  dealii::ArborXWrappers::DistributedTree distributed_tree(communicator,
                                                           support_points);
  dealii::ArborXWrappers::PointNearestPredicate pt_nearest(points_values.points,
                                                           1);
  auto [indices_ranks, offset] = distributed_tree.query(pt_nearest);

  // Convert the indices and offsets to a pair that maps experimental indices to
  // dof indices
  std::vector<int> obs_indices;
  std::vector<int> global_indices;
  obs_indices.resize(indices_ranks.size());
  global_indices.resize(indices_ranks.size());
  unsigned int const n_queries = points_values.points.size();
  for (unsigned int i = 0; i < n_queries; ++i)
  {
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      obs_indices[j] = i;
      global_indices[j] = indices_ranks[j].second == my_rank
                              ? dof_indices[indices_ranks[j].first]
                              : -1;
    }
  }

  dealii::Utilities::MPI::max(global_indices, communicator, global_indices);

  return std::make_pair(obs_indices, global_indices);
}

template <int dim>
void set_with_experimental_data(
    MPI_Comm const &communicator, PointsValues<dim> const &points_values,
    std::pair<std::vector<int>, std::vector<int>> &expt_to_dof_mapping,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature,
    bool verbose_output)
{
  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  if ((rank == 0) && verbose_output)
  {
    std::cout << "Number of unique experimental points is "
              << std::set<double>(expt_to_dof_mapping.second.begin(),
                                  expt_to_dof_mapping.second.end())
                     .size()
              << std::endl;
  }

  auto locally_owned_dofs = temperature.locally_owned_elements();
  for (unsigned int i = 0; i < points_values.values.size(); ++i)
  {
    // Only add locally owned elements
    if (locally_owned_dofs.is_element(expt_to_dof_mapping.second[i]))
      temperature[expt_to_dof_mapping.second[i]] = points_values.values[i];
  }

  temperature.compress(dealii::VectorOperation::insert);
}

std::vector<std::vector<double>>
read_frame_timestamps(boost::property_tree::ptree const &experiment_database)
{
  // PropertyTreeInput experiment.log_filename
  std::string log_filename =
      experiment_database.get<std::string>("log_filename");

  wait_for_file(log_filename, "Waiting for frame time stamps: " + log_filename);

  // PropertyTreeInput experiment.first_frame_temporal_offset
  double first_frame_offset =
      experiment_database.get("first_frame_temporal_offset", 0.0);

  // PropertyTreeInput experiment.first_frame
  unsigned int first_frame =
      experiment_database.get<unsigned int>("first_frame", 0);
  // PropertyTreeInput experiment.last_frame
  unsigned int last_frame = experiment_database.get<unsigned int>("last_frame");

  // PropertyTreeInput experiment.first_camera_id
  unsigned int first_camera_id =
      experiment_database.get<unsigned int>("first_camera_id");
  // PropertyTreeInput experiment.last_camera_id
  unsigned int last_camera_id =
      experiment_database.get<unsigned int>("last_camera_id");

  unsigned int num_cameras = last_camera_id - first_camera_id + 1;
  std::vector<std::vector<double>> time_stamps(num_cameras);

  std::vector<double> first_frame_value(num_cameras);

  // Read and parse the file
  std::ifstream file;
  file.open(log_filename);
  std::string line;
  while (std::getline(file, line))
  {
    unsigned int entry_index = 0;
    std::stringstream s_stream(line);
    bool frame_of_interest = false;
    unsigned int frame = std::numeric_limits<unsigned int>::max();
    while (s_stream.good())
    {
      std::string substring;
      std::getline(s_stream, substring, ',');
      boost::trim(substring);

      if (entry_index == 0)
      {
        std::string error_message = "The file " + log_filename +
                                    " does not have consecutive frame indices.";
        ASSERT_THROW(std::stoi(substring) - frame == 1 ||
                         frame == std::numeric_limits<unsigned int>::max(),
                     error_message.c_str());
        frame = std::stoi(substring);
        if (frame >= first_frame && frame <= last_frame)
          frame_of_interest = true;
      }
      else
      {
        if (frame == first_frame && substring.size() > 0)
          first_frame_value[entry_index - 1] = std::stod(substring);

        if (frame_of_interest && substring.size() > 0)
          time_stamps[entry_index - 1].push_back(
              std::stod(substring) - first_frame_value[entry_index - 1] +
              first_frame_offset);
      }
      entry_index++;
    }
  }
  return time_stamps;
}

} // namespace adamantine

//-------------------- Explicit Instantiations --------------------//
namespace adamantine
{
template void set_with_experimental_data(
    MPI_Comm const &communicator, PointsValues<2> const &points_values,
    std::pair<std::vector<int>, std::vector<int>> &expt_to_dof_mapping,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature,
    bool verbose_output);
template void set_with_experimental_data(
    MPI_Comm const &communicator, PointsValues<3> const &points_values,
    std::pair<std::vector<int>, std::vector<int>> &expt_to_dof_mapping,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature,
    bool verbose_output);
template std::pair<std::vector<int>, std::vector<int>>
get_expt_to_dof_mapping(PointsValues<2> const &points_values,
                        dealii::DoFHandler<2> const &dof_handler);
template std::pair<std::vector<int>, std::vector<int>>
get_expt_to_dof_mapping(PointsValues<3> const &points_values,
                        dealii::DoFHandler<3> const &dof_handler);
template std::pair<std::vector<dealii::types::global_dof_index>,
                   std::vector<dealii::Point<2>>>
get_dof_to_support_mapping(dealii::DoFHandler<2> const &dof_handler);
template std::pair<std::vector<dealii::types::global_dof_index>,
                   std::vector<dealii::Point<3>>>
get_dof_to_support_mapping(dealii::DoFHandler<3> const &dof_handler);
} // namespace adamantine
