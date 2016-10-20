/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE MaterialProperty

#include "main.cc"

#include "Geometry.hh"
#include "MaterialProperty.hh"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <boost/mpi/communicator.hpp>
#include <boost/property_tree/ptree.hpp>

BOOST_AUTO_TEST_CASE(material_property)
{
  boost::mpi::communicator communicator;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);

  // Create the MaterialProperty
  boost::property_tree::ptree database;
  database.put("n_materials", 1);
  database.put("material_0.solid.density", 1.);
  database.put("material_0.solid.thermal_conductivity", 10.);
  database.put("material_0.powder.conductivity", 10.);
  database.put("material_0.liquid", "");
  database.put("material_0.liquidus", "100");
  adamantine::MaterialProperty<2> mat_prop(
      communicator, geometry.get_triangulation(), database);

  dealii::Triangulation<2> tria;
  dealii::GridGenerator::hyper_cube(tria);
  for (auto cell : tria.active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
  }
  dealii::LA::distributed::Vector<double> dummy;

  for (auto cell : tria.active_cell_iterators())
  {
    double const density =
        mat_prop.get(cell, adamantine::Property::density, dummy);
    BOOST_CHECK(density == 1.);
    double const th_conduc =
        mat_prop.get(cell, adamantine::Property::thermal_conductivity, dummy);
    BOOST_CHECK(th_conduc == 10.);
    double const liquidus =
        mat_prop.get(cell, adamantine::Property::liquidus, dummy);
    BOOST_CHECK(liquidus == 100.);
  }
}
