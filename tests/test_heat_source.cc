/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE HeatSource

#include <HeatSource.hh>

#include "main.cc"

namespace adamantine
{

BOOST_AUTO_TEST_CASE(heat_source_value_2d)
{
  double const tolerance = 1e-10;

  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  HeatSource<2> heat_source(database);

  double value = 0.0;

  dealii::Point<2> point1(0.0,0.15);
  heat_source.set_time(1.0e-7);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point1);
  std::cout << "Checking point 1..." << std::endl;
  BOOST_CHECK_CLOSE(value, 0.0, tolerance);

  dealii::Point<2> point2(10.0,0.0);
  heat_source.set_time(5.0e-7);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point2);
  std::cout << "Checking point 2..." << std::endl;
  BOOST_CHECK_CLOSE(value, 0.0, tolerance);

  // Check the beam center 0.001 s into the second segment
  dealii::Point<2> point3(8.0e-4,0.2);
  heat_source.set_time(0.001001);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point3);
  double pi_over_3_to_1p5 = pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value = -2.0*0.1*10.0/(0.5*0.5*0.1*pi_over_3_to_1p5);
  std::cout << "Checking point 3..." << std::endl;
  BOOST_CHECK_CLOSE(value, expected_value, tolerance);

  // Check slightly off beam center 0.001 s into the second segment
  dealii::Point<2> point4(7.0e-4,0.19);
  heat_source.set_time(0.001001);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point4);
  expected_value = -2.0*0.1*10.0/(0.5*0.5*0.1*pi_over_3_to_1p5);
  expected_value *= std::exp(-3.0*1.0e-4*1.0e-4/0.25 - 3.0*0.01*0.01/0.1/0.1);
  std::cout << "Checking point 4..." << std::endl;
  BOOST_CHECK_CLOSE(value, expected_value, tolerance);

}

BOOST_AUTO_TEST_CASE(heat_source_value_3d)
{
  double const tolerance = 1e-12;

  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  HeatSource<3> heat_source(database);

  double value = 0.0;

  dealii::Point<3> point1(0.0,0.15,0.0);
  heat_source.set_time(1.0e-7);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point1);
  std::cout << "Checking point 1..." << std::endl;
  BOOST_CHECK_CLOSE(value, 0.0, tolerance);

  dealii::Point<3> point2(10.0,0.0,0.0);
  heat_source.set_time(5.0e-7);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point2);
  std::cout << "Checking point 2..." << std::endl;
  BOOST_CHECK_CLOSE(value, 0.0, tolerance);

  // Check the beam center 0.001 s into the second segment
  dealii::Point<3> point3(8e-4,0.2,0.0);
  heat_source.set_time(0.001001);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point3);
  double pi_over_3_to_1p5 = pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value = -2.0*0.1*10.0/0.5/0.5/0.1/pi_over_3_to_1p5;
  std::cout << "Checking point 3..." << std::endl;
  BOOST_CHECK_CLOSE(value, expected_value, tolerance);

  // Check slightly off beam center 0.001 s into the second segment
  dealii::Point<3> point4(7.0e-4,0.19,0.0);
  heat_source.set_time(0.001001);
  heat_source.set_max_height(0.2);
  value = heat_source.value(point4);
  expected_value = -2.0*0.1*10.0/(0.5*0.5*0.1*pi_over_3_to_1p5);
  expected_value *= std::exp(-3.0*1.0e-4*1.0e-4/0.25 - 3.0*0.01*0.01/0.1/0.1);
  std::cout << "Checking point 4..." << std::endl;
  BOOST_CHECK_CLOSE(value, expected_value, tolerance);
}

} // namespace adamantine
