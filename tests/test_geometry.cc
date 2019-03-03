/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Geometry

#include "main.cc"

#include "Geometry.hh"
#include "types.hh"
#include <deal.II/grid/filtered_iterator.h>
#include <boost/property_tree/ptree.hpp>

template <int dim>
void check_material_id(
    dealii::parallel::distributed::Triangulation<dim> const &tria)
{
  dealii::types::boundary_id const top_boundary = 3;
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
          BOOST_CHECK(cell->user_index() ==
                      static_cast<int>(adamantine::MaterialState::powder));
          powder = true;
          break;
        }
      }
      if (powder == false)
        BOOST_CHECK(cell->user_index() ==
                    static_cast<int>(adamantine::MaterialState::solid));
    }
    else
      BOOST_CHECK(cell->user_index() ==
                  static_cast<int>(adamantine::MaterialState::solid));
  }
}

BOOST_AUTO_TEST_CASE(geometry_2D)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("length", 12);
  database.put("length_divisions", 4);
  database.put("height", 6);
  database.put("height_divisions", 5);

  adamantine::Geometry<2> geometry(communicator, database);
  dealii::parallel::distributed::Triangulation<2> const &tria =
      geometry.get_triangulation();

  BOOST_CHECK(tria.n_active_cells() == 20);
  check_material_id(tria);
}

BOOST_AUTO_TEST_CASE(geometry_3D)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree database;
  database.put("length", 12);
  database.put("length_divisions", 4);
  database.put("height", 4);
  database.put("height_divisions", 2);
  database.put("width", 6);
  database.put("width_divisions", 5);

  adamantine::Geometry<3> geometry(communicator, database);
  dealii::parallel::distributed::Triangulation<3> const &tria =
      geometry.get_triangulation();

  BOOST_CHECK(tria.n_active_cells() == 40);
  check_material_id(tria);
}
