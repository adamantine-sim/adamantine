/* Copyright (c) 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <RayTracing.hh>
#include <utils.hh>

#include <deal.II/arborx/distributed_tree.h>
#include <deal.II/grid/filtered_iterator.h>

#include <boost/filesystem.hpp>

#include <Kokkos_HostSpace.hpp>

#include <fstream>
#include <regex>

#include <ArborX_Ray.hpp>

namespace adamantine
{
/**
 * This class implements a predicate for the intersection of Ray and the first
 * BoundingBox. In ArborX, this is a nearest search between a Ray and the
 * BoundingBoxes.
 */
class RayNearestPredicate
{
public:
  /**
   * Constructor. @p points is a list of points which we are interested in
   * knowing if they intersect ArborXWrappers::BVH bounding boxes.
   */
  RayNearestPredicate(std::vector<Ray<3>> const &rays) : _rays(rays) {}

  /**
   * Number of rays stored in the structure.
   */
  std::size_t size() const { return _rays.size(); }

  /**
   * Return the `i`th Ray stored in the object.
   */
  Ray<3> const &get(unsigned int i) const { return _rays[i]; }

private:
  std::vector<Ray<3>> _rays;
};
} // namespace adamantine

namespace ArborX
{
template <>
struct AccessTraits<adamantine::RayNearestPredicate, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;

  static std::size_t size(adamantine::RayNearestPredicate const &ray_nearest)
  {
    return ray_nearest.size();
  }

  static auto get(adamantine::RayNearestPredicate const &ray_nearest,
                  std::size_t i)
  {
    auto const &ray = ray_nearest.get(i);
    auto const &origin = ray.origin;
    auto const &direction = ray.direction;
    ArborX::Experimental::Ray arborx_ray = {
        ArborX::Point{(float)origin[0], (float)origin[1], (float)origin[2]},
        ArborX::Experimental::Vector{(float)direction[0], (float)direction[1],
                                     (float)direction[2]}};
    return nearest(arborx_ray, 1);
  }
};
} // namespace ArborX

namespace adamantine
{
RayTracing::RayTracing(boost::property_tree::ptree const &experiment_database,
                       dealii::DoFHandler<3> const &dof_handler)
    : _dof_handler(dof_handler)
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

unsigned int RayTracing::read_next_frame()
{
  _rays_current_frame.clear();
  _values_current_frame.clear();
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
    std::getline(file, line); // skip the header
    while (std::getline(file, line))
    {
      std::size_t pos = 0;
      std::size_t last_pos = 0;
      std::size_t line_length = line.length();
      unsigned int i = 0;
      dealii::Point<dim> point;
      dealii::Tensor<1, dim> direction;
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
          else if (i < 2 * dim)
          {
            // Calculate the direction from the first and second points in the
            // file
            direction[i - dim] =
                std::strtod(line.data() + last_pos, &end) - point[i - dim];
          }
          else
          {
            value = std::strtod(line.data() + last_pos, &end);
          }

          ++i;
        }

        last_pos = pos + 1;
      }

      Ray<dim> ray{point, direction};
      _rays_current_frame.push_back(ray);
      _values_current_frame.push_back(value);
    }
  }

  return _next_frame++;
}

PointsValues<3> RayTracing::get_points_values()
{
  PointsValues<dim> points_values;

  // Perform the ray tracing to get the cells that are intersected by rays

  // Create the bounding boxes associated to the locally owned cells with FE
  // index = 0
  std::vector<dealii::BoundingBox<dim>> bounding_boxes;
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      cell_iterators;
  for (auto const &cell : dealii::filter_iterators(
           _dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    bounding_boxes.push_back(cell->bounding_box());
    cell_iterators.push_back(cell);
  }

  // Use ArborX to find where the rays intersect the activated cells. All the
  // processors have access to all the rays but we still need to use
  // DistributedTree because some rays can be stopped by activated cells on a
  // different processors. Since the rays are on all the processors, we don't
  // need to communicate the results to other processors.
  auto communicator = _dof_handler.get_communicator();
  dealii::ArborXWrappers::DistributedTree distributed_tree(communicator,
                                                           bounding_boxes);
  RayNearestPredicate ray_nearest(_rays_current_frame);
  auto [indices_ranks, offset] = distributed_tree.query(ray_nearest);

  // Find the exact intersections points
  // See https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
  // NOTE that we assume that the faces are flat. If the faces are curved,
  // this is wrong.
  int const my_rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  unsigned int const n_rays = _rays_current_frame.size();
  unsigned int n_intersections = 0;
  for (unsigned int i = 0; i < n_rays; ++i)
  {
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      if (indices_ranks[j].second == my_rank)
      {
        ++n_intersections;
      }
    }
  }

  // If there are no intersections with the mesh leave early
  if (n_intersections == 0)
    return points_values;

  points_values.points.reserve(n_intersections);
  points_values.values.reserve(n_intersections);
  auto constexpr reference_cell = dealii::ReferenceCells::get_hypercube<dim>();
  double constexpr tol = 1e-10; // 1e-6
  for (unsigned int i = 0; i < n_rays; ++i)
  {
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      if (indices_ranks[j].second == my_rank)
      {
        double distance = std::numeric_limits<double>::max();
        dealii::Point<dim> intersection;
        auto const &cell = cell_iterators[indices_ranks[j].first];
        // We know that the ray intersects the bounding box but we don't know
        // where it intersects the cells. We need to check the intersection of
        // the ray with each face of the cell.
        for (unsigned int f = 0; f < reference_cell.n_faces(); ++f)
        {
          // First we check if the ray is parallel to the face. If this is the
          // case, either the ray misses the face or the ray hits the edge of
          // the face. In that last case, the ray is also orthogonal to
          // another face and it is safe to discard all rays parallel to a
          // face.
          auto const point_0 = cell->face(f)->vertex(0);
          auto const point_1 = cell->face(f)->vertex(1);
          auto const point_2 = cell->face(f)->vertex(2);
          dealii::Tensor<1, dim> edge_01({point_1[0] - point_0[0],
                                          point_1[1] - point_0[1],
                                          point_1[2] - point_0[2]});
          dealii::Tensor<1, dim> edge_02({point_2[0] - point_0[0],
                                          point_2[1] - point_0[1],
                                          point_2[2] - point_0[2]});
          auto const &ray_direction = _rays_current_frame[i].direction;
          dealii::Tensor<2, dim> matrix(
              {{-ray_direction[0], -ray_direction[1], -ray_direction[2]},
               {edge_01[0], edge_01[1], edge_01[2]},
               {edge_02[0], edge_02[1], edge_02[2]}});
          double const det = dealii::determinant(matrix);
          double face_area = 0.;
          for (int e = 0; e < dim; ++e)
            for (int k = 0; k < dim; ++k)
              face_area += std::abs(edge_01[e] * edge_02[k]);
          // If determinant is close to zero, the ray is parallel to the face
          // and we go to the next face.
          if (std::abs(det) < tol * face_area)
            continue;
          // Compute the distance along the ray direction between the origin
          // of the ray and the intersection point.
          auto const cross_product = dealii::cross_product_3d(edge_01, edge_02);
          auto const &ray_origin = _rays_current_frame[i].origin;
          dealii::Tensor<1, dim> p0_ray({ray_origin[0] - point_0[0],
                                         ray_origin[1] - point_0[1],
                                         ray_origin[2] - point_0[2]});
          double d = cross_product * p0_ray / det;
          // If d is less than 0, this means that the ray intersects the plane
          // of the face but not the face itself. We can exit early.
          if (d < 0)
            continue;
          // We can finally compute the intersection point. It is possible
          // that a ray intersects multiple faces. For instance if the mesh is
          // a cube the ray will get into the cube from one face and it will
          // get out of the cube by the opposite face. The correct
          // intersection point is the one with the smallest distance.
          if (d < distance)
          {
            // The point intersects the plane of the face but maybe not the
            // face itself. Check that the point is on the face. NOTE: We
            // assume that the face is an axis-aligned rectangle.
            dealii::Point<dim> face_intersection =
                ray_origin + d * ray_direction;
            std::vector<double> min(dim, std::numeric_limits<double>::max());
            std::vector<double> max(dim, std::numeric_limits<double>::lowest());
            for (unsigned int coord = 0; coord < dim; ++coord)
            {
              if (point_0[coord] < min[coord])
                min[coord] = point_0[coord];
              if (point_0[coord] > max[coord])
                max[coord] = point_0[coord];

              if (point_1[coord] < min[coord])
                min[coord] = point_1[coord];
              if (point_1[coord] > max[coord])
                max[coord] = point_1[coord];

              if (point_2[coord] < min[coord])
                min[coord] = point_2[coord];
              if (point_2[coord] > max[coord])
                max[coord] = point_2[coord];
            }

            bool on_the_face = true;
            double const effective_edge = std::sqrt(face_area);
            for (int coord = 0; coord < dim; ++coord)
            {
              if ((face_intersection[coord] <
                   (min[coord] - tol * effective_edge)) ||
                  (face_intersection[coord] >
                   (max[coord] + tol * effective_edge)))
              {
                on_the_face = false;
                break;
              }
            }

            if (on_the_face)
            {
              distance = d;
              intersection = face_intersection;
            }
          }
        }
        if (distance < std::numeric_limits<double>::max())
        {
          points_values.points.push_back(intersection);
          points_values.values.push_back(_values_current_frame[i]);
        }
      }
    }
  }
  // We know that there are at most n_intersections between the rays and the
  // mesh
  ASSERT(points_values.points.size() <= n_intersections,
         "Error when computing the intersection of rays with the mesh.");

  return points_values;
}

} // namespace adamantine
