/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE HeatSource

#include <ElectronBeamHeatSource.hh>
#include <GoldakHeatSource.hh>
#include <HeatSources.hh>
#include <ScanPath.hh>

#include "main.cc"

namespace utf = boost::unit_test;

namespace adamantine
{

template <int dim>
std::tuple<Kokkos::View<ScanPathSegment *, Kokkos::HostSpace>,
           ElectronBeamHeatSource<dim>, GoldakHeatSource<dim>>
create_heat_sources(std::string scan_path_file)
{
  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("scan_path_file", scan_path_file);
  database.put("scan_path_file_format", "segment");
  std::vector<ScanPathSegment> scan_path_segments =
      ScanPath::read_file(database.get<std::string>("scan_path_file"),
                          database.get<std::string>("scan_path_file_format"));
  Kokkos::View<ScanPathSegment *, Kokkos::HostSpace> scan_path_segments_view(
      "scan_path_segments", scan_path_segments.size());
  Kokkos::deep_copy(scan_path_segments_view,
                    Kokkos::View<ScanPathSegment *, Kokkos::HostSpace>{
                        scan_path_segments.data(), scan_path_segments.size()});
  BeamHeatSourceProperties beam(database);
  return std::tuple(
      scan_path_segments_view,
      ElectronBeamHeatSource<dim>{database, ScanPath(scan_path_segments_view)},
      GoldakHeatSource<dim>{beam, ScanPath(scan_path_segments_view)});
}

BOOST_AUTO_TEST_CASE(heat_source_value_2d, *utf::tolerance(1e-12))
{
  auto [scan_paths_segments, eb_heat_source, goldak_heat_source] =
      create_heat_sources<2>("scan_path.txt");

  double g_value = 0.0;
  double eb_value = 0.0;

  std::cout << "Checking point 1..." << std::endl;
  dealii::Point<2> point1(0.0, 0.15);
  goldak_heat_source.update_time(1.0e-7);
  g_value = goldak_heat_source.value(point1, 0.2);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(1.0e-7);
  eb_value = eb_heat_source.value(point1, 0.2);
  BOOST_TEST(eb_value == 0.0);

  std::cout << "Checking point 2..." << std::endl;
  dealii::Point<2> point2(10.0, 0.0);
  goldak_heat_source.update_time(5.0e-7);
  g_value = goldak_heat_source.value(point2, 0.2);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(5.0e-7);
  eb_value = eb_heat_source.value(point2, 0.2);
  BOOST_TEST(eb_value == 0.0);

  // Check the beam center 0.001 s into the second segment
  std::cout << "Checking point 3..." << std::endl;
  dealii::Point<2> point3(8.0e-4, 0.2);
  goldak_heat_source.update_time(0.001001);
  g_value = goldak_heat_source.value(point3, 0.2);
  double pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value =
      2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  BOOST_TEST(g_value == expected_value);

  eb_heat_source.update_time(0.001001);
  eb_value = eb_heat_source.value(point3, 0.2);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) * 1. * 1.;
  BOOST_TEST(eb_value == expected_value);

  // Check slightly off beam center 0.001 s into the second segment
  std::cout << "Checking point 4..." << std::endl;
  dealii::Point<2> point4(7.0e-4, 0.19);
  g_value = goldak_heat_source.value(point4, 0.2);
  expected_value = 2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  expected_value *=
      std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
  BOOST_TEST(g_value == expected_value);

  eb_value = eb_heat_source.value(point4, 0.2);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
                   std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
                   (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
  BOOST_TEST(eb_value == expected_value);

  // Checking beyond the defined time, where the expected value is zero
  std::cout << "Checking point 5..." << std::endl;
  goldak_heat_source.update_time(100.0);
  g_value = goldak_heat_source.value(point4, 0.2);
  expected_value = 0.0;
  BOOST_TEST(g_value == expected_value);

  eb_heat_source.update_time(100.0);
  eb_value = eb_heat_source.value(point4, 0.2);
  BOOST_TEST(eb_value == expected_value);
}

BOOST_AUTO_TEST_CASE(heat_source_value_3d, *utf::tolerance(1e-12))
{
  auto [scan_paths_segments, eb_heat_source, goldak_heat_source] =
      create_heat_sources<3>("scan_path.txt");

  double g_value = 0.0;
  double eb_value = 0.0;

  std::cout << "Checking point 1..." << std::endl;
  dealii::Point<3> point1(0.0, 0.0, 0.15);
  goldak_heat_source.update_time(1.0e-7);
  g_value = goldak_heat_source.value(point1, 0.2);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(1.0e-7);
  eb_value = eb_heat_source.value(point1, 0.2);
  BOOST_TEST(eb_value == 0.0);

  std::cout << "Checking point 2..." << std::endl;
  dealii::Point<3> point2(10.0, 0.0, 0.0);
  goldak_heat_source.update_time(5.0e-7);
  g_value = goldak_heat_source.value(point2, 0.2);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(5.0e-7);
  eb_value = eb_heat_source.value(point2, 0.2);
  BOOST_TEST(eb_value == 0.0);

  // Check the beam center 0.001 s into the second segment
  std::cout << "Checking point 3..." << std::endl;
  dealii::Point<3> point3(8e-4, 0.1, 0.2);
  goldak_heat_source.update_time(0.001001);
  g_value = goldak_heat_source.value(point3, 0.2);
  double pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value = 2.0 * 0.1 * 10.0 / 0.5 / 0.5 / 0.1 / pi_over_3_to_1p5;
  BOOST_TEST(g_value == expected_value);

  eb_heat_source.update_time(0.001001);
  eb_value = eb_heat_source.value(point3, 0.2);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) * 1. * 1.;
  BOOST_TEST(eb_value == expected_value);

  // Check slightly off beam center 0.001 s into the second segment
  std::cout << "Checking point 4..." << std::endl;
  dealii::Point<3> point4(7.0e-4, 0.1, 0.19);
  g_value = goldak_heat_source.value(point4, 0.2);
  expected_value = 2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  expected_value *=
      std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
  BOOST_TEST(g_value == expected_value);

  eb_value = eb_heat_source.value(point4, 0.2);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
                   std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
                   (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
  BOOST_TEST(eb_value == expected_value);
}

BOOST_AUTO_TEST_CASE(heat_source_height, *utf::tolerance(1e-12))
{
  auto [scan_paths_segments, eb_heat_source, goldak_heat_source] =
      create_heat_sources<2>("scan_path_layers.txt");

  double g_height = 0.0;
  double eb_height = 0.0;

  // Check the height for the first segment
  g_height = goldak_heat_source.get_current_height(1.0e-7);
  BOOST_TEST(g_height == 0.);

  eb_height = eb_heat_source.get_current_height(1.0e-7);
  BOOST_TEST(eb_height == 0.);

  // Check the height for the second segment
  g_height = goldak_heat_source.get_current_height(0.001001);
  BOOST_TEST(g_height == 0.);

  eb_height = eb_heat_source.get_current_height(0.001001);
  BOOST_TEST(eb_height == 0.);

  // Check the height for the third segment
  g_height = goldak_heat_source.get_current_height(0.003);
  BOOST_TEST(g_height == 0.001);

  eb_height = eb_heat_source.get_current_height(0.003);
  BOOST_TEST(eb_height == 0.001);
}

} // namespace adamantine
