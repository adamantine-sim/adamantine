/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <material_deposition.hh>
#include <utils.hh>

#include <deal.II/arborx/bvh.h>
#include <deal.II/grid/filtered_iterator.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <iostream>

namespace adamantine
{
template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    std::vector<std::unique_ptr<adamantine::HeatSource<dim>>> &heat_sources)
{
  // PropertyTreeInput geometry.material_deposition
  bool material_deposition =
      geometry_database.get("material_deposition", false);
  if (!material_deposition)
  {
    return {{}, {}};
  }
  else
  {
    std::string method =
        geometry_database.get<std::string>("material_deposition_method");

    ASSERT_THROW(method.compare("file") == 0 ||
                     method.compare("scan_paths") == 0,
                 "Error: Method type for material deposition, '" + method +
                     "', is not recognized. Valid options are: 'file' "
                     "and 'scan_paths'");
    if (method.compare("file") == 0)
    {
      return read_material_deposition<dim>(geometry_database);
    }
    else
    {
      std::vector<
          std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>>
          box_and_time_list;
      for (auto const &source : heat_sources)
      {
        std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
            temp_boxes_and_times = deposition_along_scan_path<dim>(
                geometry_database, source->get_scan_path());
        box_and_time_list.push_back(temp_boxes_and_times);
      }

      return merge_bounding_box_lists(box_and_time_list);
    }
  }
}

template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database)
{
  // PropertyTreeInput geometry.material_deposition_file
  std::string material_deposition_filename =
      geometry_database.get<std::string>("material_deposition_file");

  std::vector<dealii::BoundingBox<dim>> material_deposition_boxes;
  std::vector<double> material_deposition_times;

  // Read file
  ASSERT_THROW(boost::filesystem::exists(material_deposition_filename),
               "The file " + material_deposition_filename + " does not exist.");
  std::ifstream file;
  file.open(material_deposition_filename);
  std::string line;
  getline(file, line);
  int dim_ = std::stoi(line);
  ASSERT_THROW(dim_ == dim, "Dimension in " + material_deposition_filename +
                                " does not match " + std::to_string(dim) +
                                " .");
  while (getline(file, line))
  {
    std::vector<std::string> split_line;
    boost::split(split_line, line, boost::is_any_of(" "),
                 boost::token_compress_on);
    // First, read the center of the box
    std::vector<double> center(dim);
    for (int d = 0; d < dim; ++d)
    {
      center[d] = std::stod(split_line[d]);
    }
    // Next, read the size of the box
    std::vector<double> box_size(dim);
    for (int d = 0; d < dim; ++d)
    {
      box_size[d] = std::stod(split_line[dim + d]);
    }
    // Finally read the time of deposition
    material_deposition_times.push_back(std::stod(split_line[2 * dim]));
    // Check that the time is increasing
    unsigned int times_size = material_deposition_times.size();
    if (times_size > 1)
      ASSERT_THROW(material_deposition_times[times_size - 2] <=
                       material_deposition_times[times_size - 1],
                   "Time stamp not increasing.");

    // Build the dealii::BoundingBox
    if constexpr (dim == 2)
      material_deposition_boxes.emplace_back(
          std::make_pair(dealii::Point<dim>(center[0] - 0.5 * box_size[0],
                                            center[1] - 0.5 * box_size[1]),
                         dealii::Point<dim>(center[0] + 0.5 * box_size[0],
                                            center[1] + 0.5 * box_size[1])));
    else
    {
      material_deposition_boxes.emplace_back(
          std::make_pair(dealii::Point<dim>(center[0] - 0.5 * box_size[0],
                                            center[1] - 0.5 * box_size[1],
                                            center[2] - 0.5 * box_size[2]),
                         dealii::Point<dim>(center[0] + 0.5 * box_size[0],
                                            center[1] + 0.5 * box_size[1],
                                            center[2] + 0.5 * box_size[2])));
    }
  }
  file.close();

  return std::make_pair(material_deposition_boxes, material_deposition_times);
}

template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
deposition_along_scan_path(boost::property_tree::ptree const &geometry_database,
                           ScanPath const &scan_path)
{
  std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
      boxes_and_times;

  // TODO

  return boxes_and_times;
}

template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
merge_bounding_box_lists(
    std::vector<
        std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>>
        bounding_box_lists)
{
  std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
      merged_list;

  // TODO

  return merged_list;
}

template <int dim>
std::vector<std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>
get_elements_to_activate(
    dealii::DoFHandler<dim> const &dof_handler,
    std::vector<dealii::BoundingBox<dim>> const &material_deposition_boxes)
{
  // Exit early if we can
  if (material_deposition_boxes.size() == 0)
    return std::vector<
        std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>();

  // We activate the cells that intersect a box. To do that we use ArborX.
  // First, we create the bounding boxes of all the non-activated cells.
  std::vector<dealii::BoundingBox<dim>> bounding_boxes;
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      cell_iterators;
  for (auto const &cell : dealii::filter_iterators(
           dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(1)))
  {
    bounding_boxes.push_back(cell->bounding_box());
    cell_iterators.push_back(cell);
  }

  // Perform the search
  dealii::ArborXWrappers::BVH bvh(bounding_boxes);
  dealii::ArborXWrappers::BoundingBoxIntersectPredicate bb_intersect(
      material_deposition_boxes);
  auto [indices, offset] = bvh.query(bb_intersect);

  unsigned int const n_queries = material_deposition_boxes.size();
  std::vector<
      std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>
      elements_to_activate(n_queries);
  for (unsigned int i = 0; i < n_queries; ++i)
  {
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      elements_to_activate[i].push_back(cell_iterators[indices[j]]);
    }
  }

  return elements_to_activate;
}
} // namespace adamantine

//-------------------- Explicit Instantiations --------------------//
namespace adamantine
{
template std::pair<std::vector<dealii::BoundingBox<2>>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    std::vector<std::unique_ptr<adamantine::HeatSource<2>>> &heat_sources);
template std::pair<std::vector<dealii::BoundingBox<3>>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    std::vector<std::unique_ptr<adamantine::HeatSource<3>>> &heat_sources);

template std::pair<std::vector<dealii::BoundingBox<2>>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database);
template std::pair<std::vector<dealii::BoundingBox<3>>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database);

template std::vector<
    std::vector<typename dealii::DoFHandler<2>::active_cell_iterator>>
get_elements_to_activate(
    dealii::DoFHandler<2> const &dof_handler,
    std::vector<dealii::BoundingBox<2>> const &material_deposition_boxes);
template std::vector<
    std::vector<typename dealii::DoFHandler<3>::active_cell_iterator>>
get_elements_to_activate(
    dealii::DoFHandler<3> const &dof_handler,
    std::vector<dealii::BoundingBox<3>> const &material_deposition_boxes);
} // namespace adamantine
