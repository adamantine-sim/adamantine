/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef GEOMETRY_TEMPLATES_HH
#define GEOMETRY_TEMPLATES_HH

#include <Geometry.hh>
#include <MaterialStates.hh>
#include <instantiation.hh>
#include <types.hh>
#include <utils.hh>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <fstream>

namespace adamantine
{
namespace
{
#if ARBORX_VERSION_MAJOR >= 2
using Point = ArborX::Point<3, double>;
using Triangle = ArborX::Triangle<3, double>;
#else
using Point = ArborX::ExperimentalHyperGeometry::Point<3, double>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3, double>;
#endif

Point read_point(char const *facet, double const unit_scaling)
{
  Point vertex;
  float value = 0;

  std::memcpy(&value, facet, sizeof value);
  vertex[0] = static_cast<double>(value) * unit_scaling;
  std::memcpy(&value, facet + 4, sizeof value);
  vertex[1] = static_cast<double>(value) * unit_scaling;
  std::memcpy(&value, facet + 8, sizeof value);
  vertex[2] = static_cast<double>(value) * unit_scaling;

  return vertex;
}

#if ARBORX_VERSION_MAJOR >= 2
struct InsideCallBack
{
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(const Predicate &predicate,
                                  const Value &value,
                                  const OutputFunctor &out) const
  {
    auto p = ArborX::getGeometry(predicate);
#if ARBORX_VERSION_MINOR > 0
    auto a = ArborX::Experimental::closestPoint(p, value.a, value.b, value.c);
#else
    auto a = ArborX::Details::Dispatch::distance<
        ArborX::GeometryTraits::PointTag, ArborX::GeometryTraits::TriangleTag,
        decltype(p), Value>::closest_point(p, value.a, value.b, value.c);
#endif

    auto pa = a - p;
    auto e1 = value.b - value.a;
    auto e2 = value.c - value.a;
    double x = e1[1] * e2[2] - e2[1] * e1[2];
    double y = -(e1[0] * e2[2] - e2[0] * e1[2]);
    double z = e1[0] * e2[1] - e2[0] * e1[1];

    bool v = pa[0] * x + pa[1] * y + pa[2] * z > 0.;
    out(v);
  }
};
#endif
} // namespace

template <int dim>
Geometry<dim>::Geometry(
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    boost::optional<boost::property_tree::ptree const &> const
        &units_optional_database)
    : _triangulation(communicator)
{
  // PropertyTreeInput geometry.import_mesh
  bool import_mesh = database.get<bool>("import_mesh");

  if (import_mesh == true)
  {
    // PropertyTreeInput geometry.mesh_file
    std::string mesh_file = database.get<std::string>("mesh_file");
    // PropertyTreeInput geometry.mesh_format
    std::string mesh_format = database.get<std::string>("mesh_format");
    dealii::Triangulation<dim> serial_triangulation;
    dealii::GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_triangulation);
    typename dealii::GridIn<dim>::Format grid_in_format;
    if (mesh_format == "abaqus")
    {
      grid_in_format = dealii::GridIn<dim>::Format::abaqus;
    }
    else if (mesh_format == "assimp")
    {
      grid_in_format = dealii::GridIn<dim>::Format::assimp;
    }
    else if (mesh_format == "unv")
    {
      grid_in_format = dealii::GridIn<dim>::Format::unv;
    }
    else if (mesh_format == "ucd")
    {
      grid_in_format = dealii::GridIn<dim>::Format::ucd;
    }
    else if (mesh_format == "dbmesh")
    {
      grid_in_format = dealii::GridIn<dim>::Format::dbmesh;
    }
    else if (mesh_format == "gmsh")
    {
      grid_in_format = dealii::GridIn<dim>::Format::msh;
    }
    else if (mesh_format == "tecplot")
    {
      grid_in_format = dealii::GridIn<dim>::Format::tecplot;
    }
    else if (mesh_format == "xda")
    {
      grid_in_format = dealii::GridIn<dim>::Format::xda;
    }
    else if (mesh_format == "vtk")
    {
      grid_in_format = dealii::GridIn<dim>::Format::vtk;
    }
    else if (mesh_format == "vtu")
    {
      grid_in_format = dealii::GridIn<dim>::Format::vtu;
    }
    else if (mesh_format == "exodusii")
    {
      grid_in_format = dealii::GridIn<dim>::Format::exodusii;
    }
    else
    {
      grid_in_format = dealii::GridIn<dim>::Format::Default;
    }

    grid_in.read(mesh_file, grid_in_format);
    _triangulation.copy_triangulation(serial_triangulation);

    // Apply user-specified scaling to the mesh
    // PropertyTreeInput units.mesh
    std::string const mesh_unit =
        units_optional_database
            ? units_optional_database.get().get("mesh", "meter")
            : "meter";
    double mesh_scaling = g_unit_scaling_factor[mesh_unit];
    dealii::GridTools::scale(mesh_scaling, _triangulation);

    // Set the mesh material id to 0 if specified in the input
    // PropertyTreeInput geometry.reset_material_id
    auto reset_material_id = database.get("reset_material_id", false);
    if (reset_material_id)
    {
      for (auto cell : _triangulation.active_cell_iterators())
      {
        cell->set_material_id(0);
      }
    }
  }
  else
  {
    std::vector<unsigned int> repetitions(dim);
    // PropertyTreeInput geometry.length_divisions
    repetitions[axis<dim>::x] = database.get("length_divisions", 10);
    // PropertyTreeInput geometry.height_divisions
    repetitions[axis<dim>::z] = database.get("height_divisions", 10);
    // PropertyTreeInput geometry.width_divisions
    if (dim == 3)
      repetitions[axis<dim>::y] = database.get("width_divisions", 10);

    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    // PropertyTreeInput geometry.length
    p2[axis<dim>::x] = database.get<double>("length");
    // PropertyTreeInput geometry.length_origin
    p1[axis<dim>::x] = database.get("length_origin", 0.0);
    // PropertyTreeInput geometry.height
    p2[axis<dim>::z] = database.get<double>("height");
    // PropertyTreeInput geometry.height_origin
    p1[axis<dim>::z] = database.get("height_origin", 0.0);
    if (dim == 3)
    {
      // PropertyTreeInput geometry.width
      p2[axis<dim>::y] = database.get<double>("width");
      // PropertyTreeInput geometry.width_origin
      p1[axis<dim>::y] = database.get("width_origin", 0.0);
    }

    p2 = p2 + p1;

    // For now we assume that the geometry is very simple.
    dealii::GridGenerator::subdivided_hyper_rectangle(
        _triangulation, repetitions, p1, p2, true);

    // Assign the MaterialID.
    for (auto cell : _triangulation.active_cell_iterators())
    {
      cell->set_material_id(0);
    }
  }

  assign_material_state(database);

  auto stl_filename = database.get_optional<std::string>("stl_filename");
  if (stl_filename)
  {
    _use_stl = true;
    std::string const stl_unit =
        units_optional_database
            ? units_optional_database.get().get("mesh", "meter")
            : "meter";
    read_stl(*stl_filename, g_unit_scaling_factor[stl_unit]);
#if ARBORX_VERSION_MAJOR >= 2
    _bvh = std::make_unique<ArborX::BVH<Kokkos::HostSpace, Triangle>>(
        Kokkos::DefaultHostExecutionSpace{}, _stl_triangles);
#endif
  }
}

#if ARBORX_VERSION_MAJOR >= 2
template <int dim>
bool Geometry<dim>::is_within_stl(
    typename dealii::DoFHandler<dim>::active_cell_iterator const &cell) const
{
  if constexpr (dim == 2)
  {
    return false;
  }
  else
  {
    ASSERT(_bvh, "BVH not initialized.");

    // First we need to filter out the cells that intersects the triangles. The
    // center of the cells may be within the STL shape which will create a false
    // positive.
    Kokkos::DefaultHostExecutionSpace space;
    Kokkos::View<int *, Kokkos::HostSpace> offset("offset", 0);
    Kokkos::View<Triangle *, Kokkos::HostSpace> intersected_triangles(
        "intersected_triangles", 0);
    Kokkos::View<ArborX::Box<3, double>[1], Kokkos::HostSpace> bounding_box(
        "bounding_box");
    auto const boundary_points = cell->bounding_box().get_boundary_points();
    auto const &point_a = boundary_points.first;
    auto const &point_b = boundary_points.second;
    bounding_box[0] =
        ArborX::Box<3, double>(Point({{point_a[0], point_a[1], point_a[2]}}),
                               Point({{point_b[0], point_b[1], point_b[2]}}));

    _bvh->query(space, ArborX::Experimental::make_intersects(bounding_box),
                intersected_triangles, offset);
    if (intersected_triangles.extent(0) > 0)
    {
      return false;
    }

    // Now that the cells that intersect the STL have been filtered out, we
    // check the cells that the cells are within the STL. A cell is within the
    // STL, if the dot product between the normal of the closest triangle and
    // the vector from the cell center to the closest triangle is positive.
    Kokkos::View<Point[1], Kokkos::HostSpace> cell_center("cell_center");
    cell_center(0) = {
        {cell->center()[0], cell->center()[1], cell->center()[2]}};
    Kokkos::View<bool[1], Kokkos::HostSpace> inside("inside");
    _bvh->query(space, ArborX::Experimental::make_nearest(cell_center, 1),
                InsideCallBack{}, inside, offset);

    return inside(0);
  }
}
#endif

template <int dim>
void Geometry<dim>::assign_material_state(
    boost::property_tree::ptree const &database)
{
  // PropertyTreeInput geometry.material_height
  double const material_height = database.get("material_height", 1e9);
  // PropertyTreeInput geometry.use_powder
  bool const use_powder = database.get("use_powder", false);

  if (use_powder)
  {
    // PropertyTreeInput geometry.powder_layer
    double const powder_layer = database.get<double>("powder_layer");
    double const solid_height = material_height - powder_layer;

    for (auto cell : _triangulation.active_cell_iterators())
    {
      if (cell->center()[axis<dim>::z] < solid_height)
      {
        cell->set_user_index(static_cast<int>(SolidLiquidPowder::State::solid));
      }
      else
      {
        cell->set_user_index(
            static_cast<int>(SolidLiquidPowder::State::powder));
      }
    }
  }
  else
  {
    // Everything is made of solid material
    for (auto cell : _triangulation.active_cell_iterators())
    {
      cell->set_user_index(static_cast<int>(SolidLiquidPowder::State::solid));
    }
  }
}

template <int dim>
void Geometry<dim>::read_stl(std::string const &filename,
                             double const stl_scaling)
{
  std::ifstream file(filename.c_str(), std::ios::binary);
  ASSERT_THROW(file.good(), "Unable to open STL file: " + filename);

  // Read 80 byte header
  char header_info[80] = "";
  file.read(header_info, 80);

  // Read the number of triangles
  unsigned int n_triangles = 0;
  {
    char n_tri[4];
    file.read(n_tri, 4);

    std::memcpy(&n_triangles, n_tri, sizeof n_triangles);
  }

  _stl_triangles = Kokkos::View<Triangle *, Kokkos::HostSpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "stl_triangles"),
      n_triangles);

  for (unsigned int i = 0; i < n_triangles; ++i)
  {
    char facet[50];

    // Read one 50-byte triangle
    file.read(facet, 50);
    // Check that we really read 50-byte
    ASSERT_THROW(file.gcount() == 50, "Error reading triangle " +
                                          std::to_string(i) +
                                          " from STL file: " + filename);

    // Populate each point of the triangle
    // facet + 12 skips the triangle's unit normal
    auto p1 = read_point(facet + 12, stl_scaling);
    auto p2 = read_point(facet + 24, stl_scaling);
    auto p3 = read_point(facet + 36, stl_scaling);

    // Add a new triangle to the View
    _stl_triangles(i) = {p1, p2, p3};
  }
  file.close();
}
} // namespace adamantine

INSTANTIATE_DIM(Geometry)

#endif
