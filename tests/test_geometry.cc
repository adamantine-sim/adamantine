/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE Geometry

#include <Geometry.hh>
#include <MaterialStates.hh>
#include <types.hh>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <boost/property_tree/ptree.hpp>

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
