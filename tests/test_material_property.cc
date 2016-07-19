/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE MaterialProperty

#include "main.cc"

#include "MaterialProperty.hh"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <boost/property_tree/ptree.hpp>

BOOST_AUTO_TEST_CASE(material_property)
{
  boost::property_tree::ptree database;
  database.put("n_materials", 1);
  database.put("material_0.solid.thermal_conductivity", 10.);
  database.put("material_0.powder.conductivity", 10.);
  database.put("material_0.liquid", "");
  adamantine::MaterialProperty mat_prop(database);

  dealii::Triangulation<2> tria;
  dealii::GridGenerator::hyper_cube(tria);
  for (auto cell : tria.active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(adamantine::MaterialState::solid);
  }
  dealii::LA::distributed::Vector<double> dummy;

  for (auto cell : tria.active_cell_iterators())
  {
    double const value = mat_prop.get<2, double>(
        cell, adamantine::Property::thermal_conductivity, dummy);
    BOOST_CHECK(value == 10.);
  }
}
