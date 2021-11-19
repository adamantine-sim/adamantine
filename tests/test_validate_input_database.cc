/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE validate_input_database

#include <validate_input_database.hh>

#include "main.cc"

namespace adamantine
{

BOOST_AUTO_TEST_CASE(expected_passes)
{
  boost::property_tree::ptree database;
  database.put("boundary.type", "adiabatic");
  database.put("discretization.fe_degree", "1");
  database.put("discretization.quadrature", "gauss");
  database.put("geometry.dim", "3");
  validate_input_database(database);
}

BOOST_AUTO_TEST_CASE(expected_failures)
{
  boost::property_tree::ptree database;

  // Check 1: Invalid BC combination
  database.put("boundary.type", "adiabatic,convective");
  database.put("discretization.fe_degree", "1");
  database.put("discretization.quadrature", "gauss");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);

  // Check 2: Invalid quadrature type
  database.put("boundary.type", "adiabatic");
  database.put("discretization.quadrature", "gass");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
}

} // namespace adamantine
