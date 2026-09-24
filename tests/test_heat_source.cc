/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE HeatSource

#include <ElectronBeamHeatSource.hh>
#include <GaussianHeatSource.hh>
#include <GoldakHeatSource.hh>
#include <HeatSource.hh>
#include <ScanPath.hh>

#include "main.cc"

namespace utf = boost::unit_test;

namespace adamantine
{

template <int dim>
void check_gaussian_heat_source()
{
  boost::property_tree::ptree database;
  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 0.1);
  database.put("max_power", 10.);
  database.put("scan_path_file", "scan_path.txt");
  database.put("scan_path_file_format", "segment");
  database.put("A", 1.);
  database.put("B", 1.);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  GaussianHeatSource<dim> heat_source(database, units_optional_database);
  heat_source.update_time(0.001001);
  BOOST_TEST(heat_source.is_source_on());

  dealii::Point<dim> center;
  center[axis<dim>::x] = 8.0e-4;
  center[axis<dim>::z] = 0.1;
  if constexpr (dim == 3)
  {
    center[axis<dim>::y] = 0.1;
  }

  // depth / radius = 2, so A = B = 1 gives k = 2^(1 + 1) = 4.
  double constexpr radius_squared = 0.05 * 0.05;
  double constexpr depth = 0.1;
  double constexpr k = 4.;
  double const normalization = 0.5 * dealii::numbers::PI * radius_squared *
                               depth * std::tgamma(1. / k) /
                               (k * std::pow(3., 1. / k));
  double const expected_center = 1. / normalization;
  BOOST_TEST(heat_source.value(center) == expected_center);

  dealii::Point<dim> off_center = center;
  off_center[axis<dim>::x] += 0.01;
  double radial_distance_squared = 0.01 * 0.01;
  if constexpr (dim == 3)
  {
    off_center[axis<dim>::y] += 0.02;
    radial_distance_squared += 0.02 * 0.02;
  }
  off_center[axis<dim>::z] -= 0.025;
  double const expected_off_center =
      expected_center *
      std::exp(-2. * radial_distance_squared / radius_squared) *
      std::exp(-3. * std::pow(0.025 / depth, k));
  BOOST_TEST(heat_source.value(off_center) == expected_off_center);

  dealii::Point<dim> outside = center;
  outside[axis<dim>::z] += 1.e-6;
  BOOST_TEST(heat_source.value(outside) == 0.);
  outside = center;
  outside[axis<dim>::z] -= depth + 1.e-6;
  BOOST_TEST(heat_source.value(outside) == 0.);
  outside = center;
  outside[axis<dim>::x] += 0.12;
  BOOST_TEST(heat_source.value(outside) == 0.);

  unsigned int constexpr n_lanes = dealii::VectorizedArray<double>::size();
  dealii::Point<dim, dealii::VectorizedArray<double>> vectorized_points;
  for (unsigned int lane = 0; lane < n_lanes; ++lane)
  {
    dealii::Point<dim> const &point = lane == 0 ? off_center : outside;
    for (unsigned int d = 0; d < dim; ++d)
    {
      vectorized_points[d][lane] = point[d];
    }
  }
  dealii::VectorizedArray<double> const values =
      heat_source.value(vectorized_points);
  BOOST_TEST(values[0] == expected_off_center);
  for (unsigned int lane = 1; lane < n_lanes; ++lane)
  {
    BOOST_TEST(values[lane] == 0.);
  }

  heat_source.update_time(100.);
  BOOST_TEST(!heat_source.is_source_on());
  BOOST_TEST(heat_source.value(center) == 0.);
}

BOOST_AUTO_TEST_CASE(gaussian_heat_source_value_2d, *utf::tolerance(1e-12))
{
  check_gaussian_heat_source<2>();
}

BOOST_AUTO_TEST_CASE(gaussian_heat_source_value_3d, *utf::tolerance(1e-12))
{
  check_gaussian_heat_source<3>();
}

BOOST_AUTO_TEST_CASE(heat_source_value_2d, *utf::tolerance(1e-12))
{
  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("scan_path_file", "scan_path.txt");
  database.put("scan_path_file_format", "segment");
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  GoldakHeatSource<2> goldak_heat_source(database, units_optional_database);
  ElectronBeamHeatSource<2> eb_heat_source(database, units_optional_database);

  double g_value = 0.0;
  double eb_value = 0.0;

  dealii::Point<2> point1(0.0, 0.05);
  goldak_heat_source.update_time(1.0e-7);
  g_value = goldak_heat_source.value(point1);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(1.0e-7);
  eb_value = eb_heat_source.value(point1);
  BOOST_TEST(eb_value == 0.0);

  dealii::Point<2> point2(10.0, 0.0);
  goldak_heat_source.update_time(5.0e-7);
  g_value = goldak_heat_source.value(point2);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(5.0e-7);
  eb_value = eb_heat_source.value(point2);
  BOOST_TEST(eb_value == 0.0);

  // Check the beam center 0.001 s into the second segment
  dealii::Point<2> point3(8.0e-4, 0.1);
  goldak_heat_source.update_time(0.001001);
  g_value = goldak_heat_source.value(point3);
  double pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value =
      2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  BOOST_TEST(g_value == expected_value);

  eb_heat_source.update_time(0.001001);
  eb_value = eb_heat_source.value(point3);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) * 1. * 1.;
  BOOST_TEST(eb_value == expected_value);

  // Check slightly off beam center 0.001 s into the second segment
  dealii::Point<2> point4(7.0e-4, 0.09);
  g_value = goldak_heat_source.value(point4);
  expected_value = 2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  expected_value *=
      std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
  BOOST_TEST(g_value == expected_value);

  eb_value = eb_heat_source.value(point4);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
                   std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
                   (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
  BOOST_TEST(eb_value == expected_value);

  // Checking beyond the defined time, where the expected value is zero
  goldak_heat_source.update_time(100.0);
  g_value = goldak_heat_source.value(point4);
  expected_value = 0.0;
  BOOST_TEST(g_value == expected_value);

  eb_heat_source.update_time(100.0);
  eb_value = eb_heat_source.value(point4);
  BOOST_TEST(eb_value == expected_value);
}

BOOST_AUTO_TEST_CASE(heat_source_vectorized_value_2d, *utf::tolerance(1e-12))
{
  unsigned int constexpr n_lanes = dealii::VectorizedArray<double>::size();

  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("scan_path_file", "scan_path.txt");
  database.put("scan_path_file_format", "segment");
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  // Check Goldak heat source
  GoldakHeatSource<2> goldak_heat_source(database, units_optional_database);
  goldak_heat_source.update_time(0.001001);
  dealii::Point<2, dealii::VectorizedArray<double>> points;
  if (n_lanes > 1)
  {
    // Put the first point in the first lane
    points[0][0] = 8.0e-4;
    points[1][0] = 0.1;
    // Put the second point in the second lane
    points[0][1] = 7.0e-4;
    points[1][1] = 0.09;
    // Fill the other lanes with points outside of the domain
    for (unsigned int i = 2; i < n_lanes; ++i)
    {
      points[0][i] = -1e10;
      points[1][i] = -1e10;
    }
    auto const values = goldak_heat_source.value(points);

    double const pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);

    double const expected_value_0 =
        2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
    BOOST_TEST(values[0] == expected_value_0);

    double expected_value_1 =
        2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
    expected_value_1 *=
        std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
    BOOST_TEST(values[1] == expected_value_1);

    for (unsigned int i = 2; i < n_lanes; ++i)
    {
      BOOST_TEST(values[i] == 0.);
    }
  }
  else
  {
    // Check the first point
    // Only one lane is available
    points[0][0] = 8.0e-4;
    points[1][0] = 0.1;
    auto values = goldak_heat_source.value(points);

    double const pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);

    double const expected_value_0 =
        2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
    BOOST_TEST(values[0] == expected_value_0);

    // Check the second point
    points[0][0] = 7.0e-4;
    points[1][0] = 0.09;
    values = goldak_heat_source.value(points);
    double expected_value_1 =
        2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
    expected_value_1 *=
        std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
    BOOST_TEST(values[0] == expected_value_1);
  }

  // Check the electron beam heat source
  ElectronBeamHeatSource<2> eb_heat_source(database, units_optional_database);
  eb_heat_source.update_time(0.001001);
  if (n_lanes > 1)
  {
    // points is correctly initialized. We can just check the values
    auto const values = eb_heat_source.value(points);

    double const expected_value_0 = -0.1 * 10. * 1.0 * std::log(0.1) /
                                    (dealii::numbers::PI * 0.5 * 0.5 * 0.1);
    BOOST_TEST(values[0] == expected_value_0);

    double const expected_value_1 =
        -0.1 * 10. * 1.0 * std::log(0.1) /
        (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
        std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
        (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
    BOOST_TEST(values[1] == expected_value_1);

    for (unsigned int i = 2; i < n_lanes; ++i)
    {
      BOOST_TEST(values[i] == 0.);
    }
  }
  else
  {
    // Check the first point
    // Only one lane is available
    points[0][0] = 8.0e-4;
    points[1][0] = 0.1;
    auto values = eb_heat_source.value(points);

    double const expected_value_0 = -0.1 * 10. * 1.0 * std::log(0.1) /
                                    (dealii::numbers::PI * 0.5 * 0.5 * 0.1);
    BOOST_TEST(values[0] == expected_value_0);

    // Check the second point
    points[0][0] = 7.0e-4;
    points[1][0] = 0.09;
    values = eb_heat_source.value(points);
    double const expected_value_1 =
        -0.1 * 10. * 1.0 * std::log(0.1) /
        (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
        std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
        (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
    BOOST_TEST(values[0] == expected_value_1);
  }
}

BOOST_AUTO_TEST_CASE(heat_source_value_3d, *utf::tolerance(1e-12))
{
  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("scan_path_file", "scan_path.txt");
  database.put("scan_path_file_format", "segment");

  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  GoldakHeatSource<3> goldak_heat_source(database, units_optional_database);
  ElectronBeamHeatSource<3> eb_heat_source(database, units_optional_database);

  double g_value = 0.0;
  double eb_value = 0.0;

  dealii::Point<3> point1(0.0, 0.0, 0.05);
  goldak_heat_source.update_time(1.0e-7);
  g_value = goldak_heat_source.value(point1);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(1.0e-7);
  eb_value = eb_heat_source.value(point1);
  BOOST_TEST(eb_value == 0.0);

  dealii::Point<3> point2(10.0, 0.0, 0.0);
  goldak_heat_source.update_time(5.0e-7);
  g_value = goldak_heat_source.value(point2);
  BOOST_TEST(g_value == 0.0);

  eb_heat_source.update_time(5.0e-7);
  eb_value = eb_heat_source.value(point2);
  BOOST_TEST(eb_value == 0.0);

  // Check the beam center 0.001 s into the second segment
  dealii::Point<3> point3(8e-4, 0.1, 0.1);
  goldak_heat_source.update_time(0.001001);
  g_value = goldak_heat_source.value(point3);
  double pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
  double expected_value = 2.0 * 0.1 * 10.0 / 0.5 / 0.5 / 0.1 / pi_over_3_to_1p5;
  BOOST_TEST(g_value == expected_value);

  eb_heat_source.update_time(0.001001);
  eb_value = eb_heat_source.value(point3);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) * 1. * 1.;
  BOOST_TEST(eb_value == expected_value);

  // Check slightly off beam center 0.001 s into the second segment
  dealii::Point<3> point4(7.0e-4, 0.1, 0.09);
  g_value = goldak_heat_source.value(point4);
  expected_value = 2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
  expected_value *=
      std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
  BOOST_TEST(g_value == expected_value);

  eb_value = eb_heat_source.value(point4);
  expected_value = -0.1 * 10. * 1.0 * std::log(0.1) /
                   (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
                   std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
                   (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
  BOOST_TEST(eb_value == expected_value);
}

BOOST_AUTO_TEST_CASE(heat_source_vectorized_value_3d, *utf::tolerance(1e-12))
{
  unsigned int constexpr n_lanes = dealii::VectorizedArray<double>::size();

  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("absorption_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("scan_path_file", "scan_path.txt");
  database.put("scan_path_file_format", "segment");
  boost::optional<boost::property_tree::ptree const &> units_optional_database;

  // Check Goldak heat source
  GoldakHeatSource<3> goldak_heat_source(database, units_optional_database);
  goldak_heat_source.update_time(0.001001);
  dealii::Point<3, dealii::VectorizedArray<double>> points;
  if (n_lanes > 1)
  {
    // Put the first point in the first lane
    points[0][0] = 8.0e-4;
    points[1][0] = 0.1;
    points[2][0] = 0.1;
    // Put the second point in the second lane
    points[0][1] = 7.0e-4;
    points[1][1] = 0.1;
    points[2][1] = 0.09;
    // Fill the other lanes with points outside of the domain
    for (unsigned int i = 2; i < n_lanes; ++i)
    {
      points[0][i] = -1e10;
      points[1][i] = -1e10;
      points[2][i] = -1e10;
    }
    auto const values = goldak_heat_source.value(points);

    double const pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);

    double const expected_value_0 =
        2.0 * 0.1 * 10.0 / 0.5 / 0.5 / 0.1 / pi_over_3_to_1p5;
    BOOST_TEST(values[0] == expected_value_0);

    double expected_value_1 =
        2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
    expected_value_1 *=
        std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
    BOOST_TEST(values[1] == expected_value_1);

    for (unsigned int i = 2; i < n_lanes; ++i)
    {
      BOOST_TEST(values[i] == 0.);
    }
  }
  else
  {
    // Check the first point
    // Only one lane is available
    points[0][0] = 8.0e-4;
    points[1][0] = 0.1;
    points[2][0] = 0.1;
    auto values = goldak_heat_source.value(points);
    double const pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);

    double const expected_value_0 =
        2.0 * 0.1 * 10.0 / 0.5 / 0.5 / 0.1 / pi_over_3_to_1p5;
    BOOST_TEST(values[0] == expected_value_0);

    // Check the second point
    points[0][0] = 7.0e-4;
    points[1][0] = 0.1;
    points[2][0] = 0.09;
    values = goldak_heat_source.value(points);
    double expected_value_1 =
        2.0 * 0.1 * 10.0 / (0.5 * 0.5 * 0.1 * pi_over_3_to_1p5);
    expected_value_1 *=
        std::exp(-3.0 * 1.0e-4 * 1.0e-4 / 0.25 - 3.0 * 0.01 * 0.01 / 0.1 / 0.1);
    BOOST_TEST(values[0] == expected_value_1);
  }

  // Check the electron beam heat source
  ElectronBeamHeatSource<3> eb_heat_source(database, units_optional_database);
  eb_heat_source.update_time(0.001001);
  if (n_lanes > 1)
  {
    // points is correctly initialized. We can just check the values
    auto const values = eb_heat_source.value(points);

    double const expected_value_0 = -0.1 * 10. * 1.0 * std::log(0.1) /
                                    (dealii::numbers::PI * 0.5 * 0.5 * 0.1);
    BOOST_TEST(values[0] == expected_value_0);

    double const expected_value_1 =
        -0.1 * 10. * 1.0 * std::log(0.1) /
        (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
        std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
        (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
    BOOST_TEST(values[1] == expected_value_1);

    for (unsigned int i = 2; i < n_lanes; ++i)
    {
      BOOST_TEST(values[i] == 0.);
    }
  }
  else
  {
    // Check the first point
    // Only one lane is available
    points[0][0] = 8.0e-4;
    points[1][0] = 0.1;
    points[2][0] = 0.1;
    auto values = eb_heat_source.value(points);

    double const expected_value_0 = -0.1 * 10. * 1.0 * std::log(0.1) /
                                    (dealii::numbers::PI * 0.5 * 0.5 * 0.1);
    BOOST_TEST(values[0] == expected_value_0);

    // Check the second point
    points[0][0] = 7.0e-4;
    points[1][0] = 0.1;
    points[2][0] = 0.09;
    values = eb_heat_source.value(points);
    double const expected_value_1 =
        -0.1 * 10. * 1.0 * std::log(0.1) /
        (dealii::numbers::PI * 0.5 * 0.5 * 0.1) *
        std::exp(std::log(0.1) * 1.0e-4 * 1.0e-4 / 0.25) *
        (-3.0 * 0.01 * 0.01 / 0.1 / 0.1 + 2.0 * 0.01 / 0.1 + 1.0);
    BOOST_TEST(values[0] == expected_value_1);
  }
}

} // namespace adamantine
