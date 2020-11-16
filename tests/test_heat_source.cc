/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE HeatSource

#include <ElectronBeamHeatSource.hh>
#include <GoldakHeatSource.hh>
#include <HeatSource.hh>
#include <ScanPath.hh>

#include "main.cc"

namespace adamantine
{

BOOST_AUTO_TEST_CASE(heat_source_value_2d)
{
  double const tolerance = 1e-12;

  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("scan_path_file", "scan_path.txt");
  database.put("scan_path_file_format", "segment");
  GoldakHeatSource<2> goldak_heat_source(database);
  ElectronBeamHeatSource<2> eb_heat_source(database);

  double g_value = 0.0;
  double eb_value = 0.0;

  std::cout << "Checking point 1..." << std::endl;
  dealii::Point<2> point1(0.0, 0.15);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point1, 1.0e-7);
  BOOST_CHECK_CLOSE(g_value, 0.0, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point1, 1.0e-7);
  BOOST_CHECK_CLOSE(eb_value, 0.0, tolerance);

  std::cout << "Checking point 2..." << std::endl;
  dealii::Point<2> point2(10.0, 0.0);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point2, 5.0e-7);
  BOOST_CHECK_CLOSE(g_value, 0.0, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point2, 5.0e-7);
  BOOST_CHECK_CLOSE(eb_value, 0.0, tolerance);

  // Check the beam center 0.001 s into the second segment
  std::cout << "Checking point 3..." << std::endl;
  dealii::Point<2> point3(8.0e-4, 0.2);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point3, 0.001001);
  double pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value =
      2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  BOOST_CHECK_CLOSE(g_value, expected_value, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point3, 0.001001);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) * 1. * 1.;
  BOOST_CHECK_CLOSE(eb_value, expected_value, tolerance);

  // Check slightly off beam center 0.001 s into the second segment
  std::cout << "Checking point 4..." << std::endl;
  dealii::Point<2> point4(7.0e-4, 0.19);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point4, 0.001001);
  expected_value = 2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  expected_value *=
      std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
  BOOST_CHECK_CLOSE(g_value, expected_value, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point4, 0.001001);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
                   std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
                   (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
  BOOST_CHECK_CLOSE(eb_value, expected_value, tolerance);
}

BOOST_AUTO_TEST_CASE(heat_source_value_3d)
{
  double const tolerance = 1e-12;

  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("scan_path_file", "scan_path.txt");
  database.put("scan_path_file_format", "segment");

  GoldakHeatSource<3> goldak_heat_source(database);
  ElectronBeamHeatSource<3> eb_heat_source(database);

  double g_value = 0.0;
  double eb_value = 0.0;

  std::cout << "Checking point 1..." << std::endl;
  dealii::Point<3> point1(0.0, 0.15, 0.0);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point1, 1.0e-7);
  BOOST_CHECK_CLOSE(g_value, 0.0, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point1, 1.0e-7);
  BOOST_CHECK_CLOSE(eb_value, 0.0, tolerance);

  std::cout << "Checking point 2..." << std::endl;
  dealii::Point<3> point2(10.0, 0.0, 0.0);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point2, 5.0e-7);
  BOOST_CHECK_CLOSE(g_value, 0.0, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point2, 5.0e-7);
  BOOST_CHECK_CLOSE(eb_value, 0.0, tolerance);

  // Check the beam center 0.001 s into the second segment
  std::cout << "Checking point 3..." << std::endl;
  dealii::Point<3> point3(8e-4, 0.2, 0.0);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point3, 0.001001);
  double pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value = 2.0 * 0.1 * 10.0 / 0.5 / 0.5 / 0.1 / pi_over_3_to_1p5;
  BOOST_CHECK_CLOSE(g_value, expected_value, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point3, 0.001001);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) * 1. * 1.;
  BOOST_CHECK_CLOSE(eb_value, expected_value, tolerance);

  // Check slightly off beam center 0.001 s into the second segment
  std::cout << "Checking point 4..." << std::endl;
  dealii::Point<3> point4(7.0e-4, 0.19, 0.0);
  goldak_heat_source.set_max_height(0.2);
  g_value = goldak_heat_source.value(point4, 0.001001);
  expected_value = 2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  expected_value *=
      std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
  BOOST_CHECK_CLOSE(g_value, expected_value, tolerance);

  eb_heat_source.set_max_height(0.2);
  eb_value = eb_heat_source.value(point4, 0.001001);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
                   std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
                   (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
  BOOST_CHECK_CLOSE(eb_value, expected_value, tolerance);
}

} // namespace adamantine
