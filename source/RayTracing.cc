/* Copyright (c) 2023 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <RayTracing.hh>
#include <utils.hh>

#include <deal.II/arborx/distributed_tree.h>
#include <deal.II/grid/filtered_iterator.h>

#include <Kokkos_Core.hpp>

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
        {(float)origin[0], (float)origin[1], (float)origin[2]},
        {(float)direction[0], (float)direction[1], (float)direction[2]}};
    // When the mesh is unstructured, bounding boxes do not tightly bound the
    // cells and so many of them overlap. A ray may hit a bounding box but may
    // missed the cell. We need to ask for more bounding boxes to increase the
    // chance that we don't miss a ray-cell interaction. Asking for eight
    // bounding boxes seems to work pretty well.
    return nearest(arborx_ray, 8);
  }
};
} // namespace ArborX

namespace
{
template <int dim>
bool point_in_triangle(dealii::Point<dim> const &a, dealii::Point<dim> const &b,
                       dealii::Point<dim> const &c, dealii::Point<dim> const &p)
{
  // To decide if a point is inside the triangle, we compute the barycentric
  // coordinate of the points. If they are positive, the point is in the
  // triangle. See:
  // https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle/544947
  // and
  // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
  dealii::Tensor<1, dim> ab({b[0] - a[0], b[1] - a[1], b[2] - a[2]});
  dealii::Tensor<1, dim> ac({c[0] - a[0], c[1] - a[1], c[2] - a[2]});
  dealii::Tensor<1, dim> ap({p[0] - a[0], p[1] - a[1], p[2] - a[2]});
  double const ab_ab = ab * ab;
  double const ab_ac = ab * ac;
  double const ac_ac = ac * ac;
  double const ap_ab = ap * ab;
  double const ap_ac = ap * ac;
  double const denom = ab_ab * ac_ac - ab_ac * ab_ac;
  double const v = (ac_ac * ap_ab - ab_ac * ap_ac) / denom;
  double const w = (ab_ab * ap_ac - ab_ac * ap_ab) / denom;
  double const u = 1. - v - w;
  if ((u >= 0.) && (v >= 0.) && (w >= 0.))
  {
    return true;
  }

  return false;
}
} // namespace

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
    wait_for_file(filename, "Waiting for the next frame: " + filename);

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
        pos = line.find_first_of(',', last_pos);
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
  // need to communicate the results to other processors. Note that we may lose
  // some rays because the bounding boxes are larger than the cells and so a ray
  // can be stopped by a bounding box and then miss the associated cell. We
  // could get around this by asking ArborX for all the bounding boxes that the
  // rays intersect but then we would need to keep track of the order of the
  // intersections ourselves.
  // TODO The current code can be simplified a lot once we can use ArborX 2.0
  // This version of ArborX supports distributed ray tracing on triangle.
  // Currently we create a bounding box for each cell, use ArborX to perform a
  // coarse search, and finally perform a fine search. With ArborX 2.0, we will
  // need to create triangles for each face of each cell and then, ArborX will
  // perform the ray tracing on these triangles. This will simplify the current
  // code and will solve the problem of missing ray-cell intersections.
#if DEAL_II_VERSION_GTE(9, 7, 0)
  auto communicator = _dof_handler.get_mpi_communicator();
#else
  auto communicator = _dof_handler.get_communicator();
#endif
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

  std::vector<dealii::Point<dim>> points;
  std::vector<double> values;
  if (n_intersections != 0)
  {
    points.reserve(n_intersections);
    values.reserve(n_intersections);
    auto constexpr reference_cell =
        dealii::ReferenceCells::get_hypercube<dim>();
    double constexpr tol = 1e-10;
    for (unsigned int i = 0; i < n_rays; ++i)
    {
      dealii::Point<dim> intersection;
      double distance = std::numeric_limits<double>::max();
      for (int j = offset[i]; j < offset[i + 1]; ++j)
      {
        // The bounding boxes found can be on different processors. Instead of
        // performing the fine search and then comparing results between
        // processors, we just filter out the bounding boxes found on a
        // processor different than indices_ranks[offet[i]].second. Some rays
        // will be lost but the loss should be minimal.
        if ((indices_ranks[j].second == indices_ranks[offset[i]].second) &&
            (indices_ranks[j].second == my_rank))
        {
          auto const &cell = cell_iterators[indices_ranks[j].first];

          // We know that the ray intersects the bounding box but we don't know
          // where it intersects the cells. We need to check the intersection of
          // the ray with each face of the cell. To do that, we split each face
          // in two triangles and check if the point belongs either triangles.
          // We then use the barycentric coordinate method which involves
          // calculating the "weights" of the point relative to each vertex of
          // the triangle; if all weights are non-negative, then the point lies
          // inside the triangle.
          for (unsigned int f = 0; f < reference_cell.n_faces(); ++f)
          {
            auto const point_0 = cell->face(f)->vertex(0);
            auto const point_1 = cell->face(f)->vertex(1);
            auto const point_2 = cell->face(f)->vertex(2);
            auto const point_3 = cell->face(f)->vertex(3);
            // First we check if the ray is parallel to the face. If this is the
            // case, either the ray misses the face or the ray hits the edge of
            // the face. In that last case, the ray is also orthogonal to
            // another face and it is safe to discard all rays parallel to a
            // face.
            dealii::Tensor<1, dim> const edge_01({point_1[0] - point_0[0],
                                                  point_1[1] - point_0[1],
                                                  point_1[2] - point_0[2]});
            dealii::Tensor<1, dim> const edge_02({point_2[0] - point_0[0],
                                                  point_2[1] - point_0[1],
                                                  point_2[2] - point_0[2]});
            dealii::Tensor<1, dim> const ray_direction(
                {static_cast<double>(_rays_current_frame[i].direction[0]),
                 static_cast<double>(_rays_current_frame[i].direction[1]),
                 static_cast<double>(_rays_current_frame[i].direction[2])});
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
            auto const cross_product =
                dealii::cross_product_3d(edge_01, edge_02);
            auto const &ray_origin = _rays_current_frame[i].origin;
            dealii::Tensor<1, dim> p0_ray({ray_origin[0] - point_0[0],
                                           ray_origin[1] - point_0[1],
                                           ray_origin[2] - point_0[2]});
            double const d = cross_product * p0_ray / det;
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
              dealii::Point<dim> face_intersection =
                  ray_origin + d * ray_direction;
              // Check if the point intersect the first triangle
              if (point_in_triangle(point_0, point_1, point_2,
                                    face_intersection))
              {
                distance = d;
                intersection = face_intersection;
              }
              else if (point_in_triangle(point_1, point_2, point_3,
                                         face_intersection))
              {
                distance = d;
                intersection = face_intersection;
              }
            }
          }
        }
      }
      if (distance < std::numeric_limits<double>::max())
      {
        points.push_back(intersection);
        values.push_back(_values_current_frame[i]);
      }
    }
    // We know that there are at most n_intersections between the rays and the
    // mesh
    ASSERT(points.size() <= n_intersections,
           "Error when computing the intersection of rays with the mesh.");
  }

  // We need all the processors to know the intersection points of the rays with
  // the mesh and the values at these points.
  PointsValues<dim> points_values;
  auto all_points = dealii::Utilities::MPI::all_gather(communicator, points);
  auto all_values = dealii::Utilities::MPI::all_gather(communicator, values);
  for (unsigned int i = 0; i < all_points.size(); ++i)
  {
    points_values.points.insert(points_values.points.end(),
                                all_points[i].begin(), all_points[i].end());
    points_values.values.insert(points_values.values.end(),
                                all_values[i].begin(), all_values[i].end());
  }

  return points_values;
}

} // namespace adamantine
