/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE Geometry

#include <Geometry.hh>
#include <MaterialStates.hh>
#include <types.hh>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>

#include "main.cc"

namespace utf = boost::unit_test;

template <int dim>
void check_material_id(
    dealii::parallel::distributed::Triangulation<dim> const &tria,
    dealii::types::boundary_id top_boundary)
{
  for (auto cell :
       dealii::filter_iterators(tria.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    if (cell->at_boundary())
    {
      bool powder = false;
      for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::faces_per_cell;
           ++i)
      {
        if ((cell->face(i)->at_boundary()) &&
            (cell->face(i)->boundary_id() == top_boundary))
        {
          BOOST_TEST(cell->user_index() ==
                     static_cast<unsigned int>(
                         adamantine::SolidLiquidPowder::State::powder));
          powder = true;
          break;
        }
      }
      if (powder == false)
        BOOST_TEST(cell->user_index() ==
                   static_cast<unsigned int>(
                       adamantine::SolidLiquidPowder::State::solid));
    }
    else
      BOOST_TEST(cell->user_index() ==
                 static_cast<unsigned int>(
                     adamantine::SolidLiquidPowder::State::solid));
  }
}

BOOST_AUTO_TEST_CASE(geometry_2D)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("import_mesh", false);
  database.put("length", 12);
  database.put("length_divisions", 4);
  database.put("height", 6);
  database.put("height_divisions", 5);
  database.put("material_height", 6.);
  database.put("use_powder", true);
  database.put("powder_layer", 1.2);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  adamantine::Geometry<2> geometry(communicator, database,
                                   units_optional_database);
  dealii::parallel::distributed::Triangulation<2> const &tria =
      geometry.get_triangulation();

  BOOST_TEST(tria.n_active_cells() == 20);

  dealii::types::boundary_id const top_boundary = 3;
  check_material_id(tria, top_boundary);
}

BOOST_AUTO_TEST_CASE(geometry_3D)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("import_mesh", false);
  database.put("length", 12);
  database.put("length_divisions", 4);
  database.put("height", 4);
  database.put("height_divisions", 2);
  database.put("width", 6);
  database.put("width_divisions", 5);
  database.put("material_height", 4);
  database.put("use_powder", true);
  database.put("powder_layer", 2);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  adamantine::Geometry<3> geometry(communicator, database,
                                   units_optional_database);
  dealii::parallel::distributed::Triangulation<3> const &tria =
      geometry.get_triangulation();

  BOOST_TEST(tria.n_active_cells() == 40);

  dealii::types::boundary_id const top_boundary = 5;
  check_material_id(tria, top_boundary);
}

BOOST_AUTO_TEST_CASE(geometry_shifted_origin, *utf::tolerance(1e-12))
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("import_mesh", false);
  database.put("length", 12);
  database.put("length_divisions", 4);
  database.put("height", 4);
  database.put("height_divisions", 2);
  database.put("width", 6);
  database.put("width_divisions", 5);
  database.put("material_height", 4);
  database.put("use_powder", true);
  database.put("powder_layer", 2);

  database.put("length_origin", 2.);
  database.put("height_origin", -1.);
  database.put("width_origin", 3.);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  adamantine::Geometry<3> geometry(communicator, database,
                                   units_optional_database);
  dealii::parallel::distributed::Triangulation<3> const &tria =
      geometry.get_triangulation();

  auto bounding_box = dealii::GridTools::compute_bounding_box(tria);

  BOOST_TEST(bounding_box.get_boundary_points().first(0) == 2.);
  BOOST_TEST(bounding_box.get_boundary_points().first(1) == 3.);
  BOOST_TEST(bounding_box.get_boundary_points().first(2) == -1.);
  BOOST_TEST(bounding_box.get_boundary_points().second(0) == 14.);
  BOOST_TEST(bounding_box.get_boundary_points().second(1) == 9.);
  BOOST_TEST(bounding_box.get_boundary_points().second(2) == 3.);
}

BOOST_AUTO_TEST_CASE(gmsh)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("import_mesh", true);
  database.put("mesh_file", "extruded_cube.msh");
  database.put("mesh_format", "gmsh");
  database.put("material_height", 1.);
  database.put("use_powder", true);
  database.put("powder_layer", 0.05);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  adamantine::Geometry<3> geometry(communicator, database,
                                   units_optional_database);
  dealii::parallel::distributed::Triangulation<3> const &tria =
      geometry.get_triangulation();

  BOOST_TEST(tria.n_active_cells() == 320);

  dealii::types::boundary_id const top_boundary = 1;
  check_material_id(tria, top_boundary);
}

BOOST_AUTO_TEST_CASE(read_stl)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("import_mesh", false);
  database.put("length", 12);
  database.put("length_divisions", 4);
  database.put("height", 4);
  database.put("height_divisions", 2);
  database.put("width", 6);
  database.put("width_divisions", 5);
  database.put("material_height", 4);
  database.put("use_powder", true);
  database.put("powder_layer", 2);
  database.put("stl_filename", "Simple_3D_ring.stl");
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  adamantine::Geometry<3> geometry(communicator, database,
                                   units_optional_database);

  // The STL file contains 512 triangles. The bounding box of the ring is
  // 21x21x4.
  auto stl_triangles = geometry.get_stl_triangles();
  BOOST_TEST(stl_triangles.extent(0) == 512);
  double x_min = 1000.0;
  double x_max = -1000.0;
  double y_min = 1000.0;
  double y_max = -1000.0;
  double z_min = 1000.0;
  double z_max = -1000.0;
  for (unsigned int i = 0; i < stl_triangles.extent(0); ++i)
  {
    auto t = stl_triangles(i);

    x_min = std::min(t.a[0], x_min);
    x_max = std::max(t.a[0], x_max);
    y_min = std::min(t.a[1], y_min);
    y_max = std::max(t.a[1], y_max);
    z_min = std::min(t.a[2], z_min);
    z_max = std::max(t.a[2], z_max);

    x_min = std::min(t.b[0], x_min);
    x_max = std::max(t.b[0], x_max);
    y_min = std::min(t.b[1], y_min);
    y_max = std::max(t.b[1], y_max);
    z_min = std::min(t.b[2], z_min);
    z_max = std::max(t.b[2], z_max);

    x_min = std::min(t.c[0], x_min);
    x_max = std::max(t.c[0], x_max);
    y_min = std::min(t.c[1], y_min);
    y_max = std::max(t.c[1], y_max);
    z_min = std::min(t.c[2], z_min);
    z_max = std::max(t.c[2], z_max);
  }

  BOOST_TEST(x_max - x_min == 21.0);
  BOOST_TEST(y_max - y_min == 21.0);
  BOOST_TEST(z_max - z_min == 4.0);
}

#if ARBORX_VERSION_MAJOR >= 2
BOOST_AUTO_TEST_CASE(within_stl)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("import_mesh", false);
  database.put("length", 40);
  database.put("length_divisions", 80);
  database.put("height", 4);
  database.put("height_divisions", 4);
  database.put("width", 10);
  database.put("width_divisions", 40);
  database.put("material_height", 4);
  database.put("use_powder", true);
  database.put("powder_layer", 2);
  database.put("stl_filename", "Simple_3D_ring.stl");
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  adamantine::Geometry<3> geometry(communicator, database,
                                   units_optional_database);

  dealii::FE_Q<3> fe(1);
  dealii::DoFHandler<3> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe);

  std::vector<double> cells_within_vec;
  for (auto const &cell : dof_handler.active_cell_iterators())
  {
    bool within = geometry.is_within_stl(cell);
    cells_within_vec.push_back(within);
  }

  // Uncommenting the code below allows to visualize the cells that are within
  // the STL and to write the gold solution.
  /*
  dealii::Vector<double> cells_within(cells_within_vec.size());
  for (unsigned int i = 0; i < cells_within_vec.size(); ++i)
    cells_within[i] = cells_within_vec[i];

  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(cells_within, "within");
  data_out.build_patches();
  std::ofstream output("cells_within.vtu");
  data_out.write_vtu(output);

  std::ofstream gold_file_writer("within_stl_gold.txt");
  for (auto const value : cells_within_vec)
    gold_file_writer << value << " ";
  gold_file_writer.close();
  */

  std::ifstream gold_file("within_stl_gold.txt");
  for (unsigned int i = 0; i < cells_within_vec.size(); ++i)
  {
    double gold_value = -1.;
    gold_file >> gold_value;
    BOOST_TEST(cells_within_vec[i] == gold_value);
  }
}
#endif
