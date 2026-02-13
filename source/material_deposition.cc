/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <material_deposition.hh>
#include <utils.hh>

#include <deal.II/arborx/bvh.h>
#include <deal.II/grid/filtered_iterator.h>

#include <boost/algorithm/string.hpp>

#include <algorithm>
#include <fstream>
#include <tuple>

namespace adamantine
{
template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    std::vector<std::shared_ptr<HeatSource<dim>>> &heat_sources)
{
  // PropertyTreeInput geometry.material_deposition
  bool material_deposition =
      geometry_database.get("material_deposition", false);

  if (!material_deposition)
    return {{}, {}, {}, {}};

  std::string method =
      geometry_database.get<std::string>("material_deposition_method");

  if (method == "file")
  {
    return read_material_deposition<dim>(geometry_database);
  }
  else
  {
    std::vector<
        std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
                   std::vector<double>, std::vector<double>>>
        deposition_paths;
    for (auto const &source : heat_sources)
    {
      deposition_paths.emplace_back(deposition_along_scan_path<dim>(
          geometry_database, source->get_scan_path()));
    }

    return merge_deposition_paths<dim>(deposition_paths);
  }
}

template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database)
{
  // PropertyTreeInput geometry.material_deposition_file
  std::string material_deposition_filename =
      geometry_database.get<std::string>("material_deposition_file");

  std::vector<dealii::BoundingBox<dim>> material_deposition_boxes;
  std::vector<double> material_deposition_times;
  std::vector<double> material_deposition_cos;
  std::vector<double> material_deposition_sin;

  // Read file
  wait_for_file(material_deposition_filename,
                "Waiting for material deposition file: " +
                    material_deposition_filename);
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
    // Read the time of deposition
    material_deposition_times.push_back(std::stod(split_line[2 * dim]));
    // Check that the time is increasing
    unsigned int times_size = material_deposition_times.size();
    if (times_size > 1)
      ASSERT_THROW(material_deposition_times[times_size - 2] <=
                       material_deposition_times[times_size - 1],
                   "Time stamp not increasing.");
    // Read the angle of material deposition
    double deposition_angle = std::stod(split_line[2 * dim + 1]);
    material_deposition_cos.push_back(std::cos(deposition_angle));
    material_deposition_sin.push_back(std::sin(deposition_angle));

    // Build the dealii::BoundingBox
    dealii::Point<dim> bounding_pt_a;
    dealii::Point<dim> bounding_pt_b;
    for (int d = 0; d < dim; ++d)
    {
      bounding_pt_a[d] = center[d] - 0.5 * box_size[d];
      bounding_pt_b[d] = center[d] + 0.5 * box_size[d];
    }
    material_deposition_boxes.emplace_back(
        std::make_pair(bounding_pt_a, bounding_pt_b));
  }
  file.close();

  return std::make_tuple(material_deposition_boxes, material_deposition_times,
                         material_deposition_cos, material_deposition_sin);
}

template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
deposition_along_scan_path(boost::property_tree::ptree const &geometry_database,
                           ScanPath const &scan_path)
{
  std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
             std::vector<double>, std::vector<double>>
      deposition_path;
  int constexpr tuple_box = 0;
  int constexpr tuple_time = 1;
  int constexpr tuple_cos = 2;
  int constexpr tuple_sin = 3;

  // Load the box size information and lead time

  // PropertyTreeInput geometry.deposition_length
  double deposition_length = geometry_database.get<double>("deposition_length");
  // PropertyTreeInput geometry.deposition_height
  double deposition_height = geometry_database.get<double>("deposition_height");
  // PropertyTreeInput geometry.deposition_width
  double deposition_width =
      geometry_database.get<double>("deposition_width", 0.0);

  // PropertyTreeInput geometry.deposition_lead_time
  double lead_time = geometry_database.get<double>("deposition_lead_time");

  // Loop through the scan path segments, adding boxes inside each one
  std::vector<ScanPathSegment> segment_list = scan_path.get_segment_list();
  double segment_start_time = 0.0;
  Quaternion segment_start_rotation = segment_list.at(0).end_rotation;
  dealii::Point<3> segment_start_point = segment_list.at(0).end_point;
  dealii::Point<3> build_ref_segment_start_point =
      scan_path.is_five_axis()
          ? segment_start_rotation.inv_rotate(segment_start_point)
          : segment_start_point;
  for (ScanPathSegment segment : segment_list)
  {
    // Only add material if the power is on
    double const eps = 1.0e-10;
    double const eps_time = 1.0e-10;
    if (segment.power_modifier > eps)
    {
      Quaternion segment_end_rotation = segment.end_rotation;
      dealii::Point<3> segment_end_point = segment.end_point;
      dealii::Point<3> build_ref_segment_end_point =
          scan_path.is_five_axis()
              ? segment_end_rotation.inv_rotate(segment_end_point)
              : segment_end_point;
      double const segment_length =
          segment_end_point.distance(segment_start_point);
      bool in_segment = true;
      dealii::Point<3> center = build_ref_segment_start_point;
      double const segment_velocity =
          segment_length / (segment.end_time - segment_start_time);

      // Set the segment orientation. In spot mode, set the cos to 1 and the sin
      // to 0.
      double const cos =
          segment_length != 0.
              ? (segment_end_point[0] - segment_start_point[0]) / segment_length
              : 1.0;
      double const sin =
          segment_length != 0.
              ? (segment_end_point[1] - segment_start_point[1]) / segment_length
              : 0.0;
      double next_box_length = deposition_length;

      while (in_segment)
      {
        double distance_to_box_center =
            center.distance(build_ref_segment_start_point);
        double time_to_box_center =
            segment_velocity != 0. ? distance_to_box_center / segment_velocity
                                   : 0.;

        dealii::Point<dim> bounding_pt_a;
        dealii::Point<dim> bounding_pt_b;

        if (segment_end_rotation.is_valid())
        {
          if constexpr (dim == 3)
          {
            double x_length = std::abs(cos) * next_box_length +
                              std::abs(sin) * deposition_width;
            double y_length = std::abs(cos) * deposition_width +
                              std::abs(sin) * next_box_length;

            dealii::Point<3> max_corner(x_length / 2., y_length / 2., 0.);
            dealii::Point<3> min_corner(-x_length / 2., -y_length / 2.,
                                        -deposition_height);

            // We need to rotate the box to match the scan path
            dealii::Point<3> rotated_max_corner =
                segment_end_rotation.rotate(max_corner);
            dealii::Point<3> rotated_min_corner =
                segment_end_rotation.rotate(min_corner);

            // We need to recompute the min and max corners after rotation.
            dealii::Point<3> new_max_corner;
            dealii::Point<3> new_min_corner;
            for (int d = 0; d < 3; ++d)
            {
              new_max_corner[d] =
                  std::max(rotated_max_corner[d], rotated_min_corner[d]);
              new_min_corner[d] =
                  std::min(rotated_max_corner[d], rotated_min_corner[d]);
            }

            bounding_pt_a = center + new_min_corner;
            bounding_pt_b = center + new_max_corner;
          }
          else
          {
            ASSERT_THROW_NOT_IMPLEMENTED();
          }
        }
        else
        {
          std::vector<double> box_size(dim);
          box_size.at(axis<dim>::z) = deposition_height;

          if constexpr (dim == 2)
          {
            box_size.at(axis<dim>::x) = next_box_length;
          }
          else
          {
            box_size.at(axis<dim>::x) = std::abs(cos) * next_box_length +
                                        std::abs(sin) * deposition_width;
            box_size.at(axis<dim>::y) = std::abs(cos) * deposition_width +
                                        std::abs(sin) * next_box_length;
          }

          for (int d = 0; d < dim - 1; ++d)
          {
            bounding_pt_a[d] = center[d] - 0.5 * box_size[d];
            bounding_pt_b[d] = center[d] + 0.5 * box_size[d];
          }
          bounding_pt_a[dim - 1] = center[dim - 1] - box_size[dim - 1];
          bounding_pt_b[dim - 1] = center[dim - 1];
        }

        std::get<tuple_box>(deposition_path)
            .push_back(std::make_pair(bounding_pt_a, bounding_pt_b));
        std::get<tuple_time>(deposition_path)
            .push_back(std::max(
                segment_start_time + time_to_box_center - lead_time, eps_time));
        std::get<tuple_cos>(deposition_path).push_back(cos);
        std::get<tuple_sin>(deposition_path).push_back(sin);

        // Get the next box center
        if (distance_to_box_center + eps > segment_length)
        {
          in_segment = false;
        }
        else
        {
          // Check to see if the next box is at the end of the segment and
          // needs to have a modified length
          double center_increment = deposition_length;
          if (distance_to_box_center + deposition_length > segment_length)
          {
            center_increment = (segment_length - distance_to_box_center);
            next_box_length = segment_length - distance_to_box_center;
          }

          if (segment_end_rotation.is_valid())
          {
            if constexpr (dim == 3)
            {
              dealii::Point<3> incr_direction = build_ref_segment_end_point;
              incr_direction -= center;
              incr_direction /= incr_direction.norm();
              dealii::Point<3> incr = center_increment * incr_direction;
              center += incr;
            }
          }
          else
          {
            center[0] += cos * center_increment;
            center[1] += sin * center_increment;
          }
        }
      }
    }
    segment_start_point = segment.end_point;
    segment_start_time = segment.end_time;
    segment_start_rotation = segment.end_rotation;
    build_ref_segment_start_point =
        segment_start_rotation.is_valid()
            ? segment_start_rotation.inv_rotate(segment_start_point)
            : segment_start_point;
  }
  return deposition_path;
}

template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
merge_deposition_paths(
    std::vector<std::tuple<std::vector<dealii::BoundingBox<dim>>,
                           std::vector<double>, std::vector<double>,
                           std::vector<double>>> const &deposition_paths)
{
  // Split the vector of tuples in four vectors
  std::vector<dealii::BoundingBox<dim>> bounding_boxes;
  std::vector<double> time;
  std::vector<double> cos;
  std::vector<double> sin;
  for (auto const &path : deposition_paths)
  {
    bounding_boxes.insert(bounding_boxes.end(), std::get<0>(path).begin(),
                          std::get<0>(path).end());
    time.insert(time.end(), std::get<1>(path).begin(), std::get<1>(path).end());
    cos.insert(cos.end(), std::get<2>(path).begin(), std::get<2>(path).end());
    sin.insert(sin.end(), std::get<3>(path).begin(), std::get<3>(path).end());
  }

  // Create the permutation that sort the deposition times chronologically
  unsigned int const n_boxes = bounding_boxes.size();
  std::vector<int> permutation(n_boxes);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&](int const &i, int const &j) { return time[i] < time[j]; });

  // Apply the permutation to all the vectors. This is not the most memory
  // efficient way to do it but I don't think it matters. We store a lot more
  // dofs
  std::vector<dealii::BoundingBox<dim>> permutated_bounding_boxes(n_boxes);
  std::vector<double> permutated_time(n_boxes);
  std::vector<double> permutated_cos(n_boxes);
  std::vector<double> permutated_sin(n_boxes);
  for (unsigned int i = 0; i < n_boxes; ++i)
  {
    permutated_bounding_boxes[i] = bounding_boxes[permutation[i]];
    permutated_time[i] = time[permutation[i]];
    permutated_cos[i] = cos[permutation[i]];
    permutated_sin[i] = sin[permutation[i]];
  }

  return std::make_tuple(permutated_bounding_boxes, permutated_time,
                         permutated_cos, permutated_sin);
}

template <int dim>
std::vector<std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>
get_elements_to_activate(
    Geometry<dim> const &geometry, dealii::DoFHandler<dim> const &dof_handler,
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
#if ARBORX_VERSION_MAJOR >= 2
  if (geometry.use_stl())
  {
    for (auto const &cell : dealii::filter_iterators(
             dof_handler.active_cell_iterators(),
             dealii::IteratorFilters::LocallyOwnedCell(),
             dealii::IteratorFilters::ActiveFEIndexEqualTo(1)))
    {
      if (geometry.is_within_stl(cell))
      {
        bounding_boxes.push_back(cell->bounding_box());
        cell_iterators.push_back(cell);
      }
    }
  }
  else
#endif
  {
    for (auto const &cell : dealii::filter_iterators(
             dof_handler.active_cell_iterators(),
             dealii::IteratorFilters::LocallyOwnedCell(),
             dealii::IteratorFilters::ActiveFEIndexEqualTo(1)))
    {
      bounding_boxes.push_back(cell->bounding_box());
      cell_iterators.push_back(cell);
    }
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
template std::tuple<std::vector<dealii::BoundingBox<2>>, std::vector<double>,
                    std::vector<double>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    std::vector<std::shared_ptr<HeatSource<2>>> &heat_sources);
template std::tuple<std::vector<dealii::BoundingBox<3>>, std::vector<double>,
                    std::vector<double>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    std::vector<std::shared_ptr<HeatSource<3>>> &heat_sources);

template std::tuple<std::vector<dealii::BoundingBox<2>>, std::vector<double>,
                    std::vector<double>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database);
template std::tuple<std::vector<dealii::BoundingBox<3>>, std::vector<double>,
                    std::vector<double>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database);

template std::vector<
    std::vector<typename dealii::DoFHandler<2>::active_cell_iterator>>
get_elements_to_activate(
    Geometry<2> const &geometry, dealii::DoFHandler<2> const &dof_handler,
    std::vector<dealii::BoundingBox<2>> const &material_deposition_boxes);
template std::vector<
    std::vector<typename dealii::DoFHandler<3>::active_cell_iterator>>
get_elements_to_activate(
    Geometry<3> const &geometry, dealii::DoFHandler<3> const &dof_handler,
    std::vector<dealii::BoundingBox<3>> const &material_deposition_boxes);

template std::tuple<std::vector<dealii::BoundingBox<2, double>>,
                    std::vector<double>, std::vector<double>,
                    std::vector<double>>
merge_deposition_paths(
    std::vector<std::tuple<std::vector<dealii::BoundingBox<2, double>>,
                           std::vector<double>, std::vector<double>,
                           std::vector<double>>> const &deposition_paths);
template std::tuple<std::vector<dealii::BoundingBox<3, double>>,
                    std::vector<double>, std::vector<double>,
                    std::vector<double>>
merge_deposition_paths(
    std::vector<std::tuple<std::vector<dealii::BoundingBox<3, double>>,
                           std::vector<double>, std::vector<double>,
                           std::vector<double>>> const &deposition_paths);

template std::tuple<std::vector<dealii::BoundingBox<2>>, std::vector<double>,
                    std::vector<double>, std::vector<double>>
deposition_along_scan_path(boost::property_tree::ptree const &geometry_database,
                           ScanPath const &scan_path);
template std::tuple<std::vector<dealii::BoundingBox<3>>, std::vector<double>,
                    std::vector<double>, std::vector<double>>
deposition_along_scan_path(boost::property_tree::ptree const &geometry_database,
                           ScanPath const &scan_path);
} // namespace adamantine
