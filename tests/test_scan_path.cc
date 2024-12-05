/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2021, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE ScanPath

#include <ScanPath.hh>

#include "main.cc"

namespace utf = boost::unit_test;

namespace adamantine
{

class ScanPathTester
{
public:
  std::vector<ScanPathSegment> get_segment_format_list()
  {
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    ScanPath scan_path("scan_path.txt", "segment", units_optional_database);
    return scan_path._segment_list;
  };
  std::vector<ScanPathSegment> get_event_series_format_list()
  {
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    ScanPath scan_path("scan_path_event_series.inp", "event_series",
                       units_optional_database);
    return scan_path._segment_list;
  };
};

BOOST_AUTO_TEST_CASE(scan_path, *utf::tolerance(1e-12))
{
  ScanPathTester tester;

  // Test the segments from a ScanPathFileFormat::segment file
  std::vector<ScanPathSegment> segment_format_list =
      tester.get_segment_format_list();

  BOOST_TEST(segment_format_list.size() == 2);

  BOOST_TEST(segment_format_list[0].end_time == 1.0e-6);
  BOOST_TEST(segment_format_list[0].end_point[0] == 0.0);
  BOOST_TEST(segment_format_list[0].end_point[1] == 0.1);
  BOOST_TEST(segment_format_list[0].power_modifier == 0.0);

  BOOST_TEST(segment_format_list[1].end_time == (1.0e-6 + 0.002 / 0.8));
  BOOST_TEST(segment_format_list[1].end_point[0] == 0.002);
  BOOST_TEST(segment_format_list[1].end_point[1] == 0.1);
  BOOST_TEST(segment_format_list[1].power_modifier == 1.0);

  // Test the segments from a ScanPathFileFormat::event_series file
  std::vector<ScanPathSegment> segment_event_series_list =
      tester.get_event_series_format_list();

  BOOST_TEST(segment_event_series_list.size() == 3);

  BOOST_TEST(segment_event_series_list[0].end_time == 0.1);
  BOOST_TEST(segment_event_series_list[0].end_point[0] == 0.0);
  BOOST_TEST(segment_event_series_list[0].end_point[1] == 0.0);
  BOOST_TEST(segment_event_series_list[0].power_modifier == 0.0);

  BOOST_TEST(segment_event_series_list[1].end_time == 1.0);
  BOOST_TEST(segment_event_series_list[1].end_point[0] == 0.5);
  BOOST_TEST(segment_event_series_list[1].end_point[1] == 0.0);
  BOOST_TEST(segment_event_series_list[1].power_modifier == 1.0);

  BOOST_TEST(segment_event_series_list[2].end_time == 2.0);
  BOOST_TEST(segment_event_series_list[2].end_point[0] == 0.5);
  BOOST_TEST(segment_event_series_list[2].end_point[1] == 0.5);
  BOOST_TEST(segment_event_series_list[2].power_modifier == 0.5);
}

BOOST_AUTO_TEST_CASE(scan_path_location, *utf::tolerance(1e-10))
{
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  ScanPath scan_path("scan_path.txt", "segment", units_optional_database);
  double time = 1.0e-7;
  dealii::Point<3> p1 = scan_path.value(time);

  BOOST_TEST(p1[0] == 0.0);
  BOOST_TEST(p1[1] == 0.1);
  BOOST_TEST(p1[2] == 0.1);

  time = 0.001001;

  dealii::Point<3> p2 = scan_path.value(time);

  BOOST_TEST(p2[0] == 8.0e-4);
  BOOST_TEST(p2[1] == 0.1);
  BOOST_TEST(p2[2] == 0.1);

  time = 100.0;
  dealii::Point<3> p3 = scan_path.value(time);
  BOOST_TEST(p3[0] == std::numeric_limits<double>::lowest());
  BOOST_TEST(p3[1] == std::numeric_limits<double>::lowest());
  BOOST_TEST(p3[2] == std::numeric_limits<double>::lowest());
  double power = scan_path.get_power_modifier(time);
  BOOST_TEST(power == 0.0);
}

} // namespace adamantine
