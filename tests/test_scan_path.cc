/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ScanPath

#include <ScanPath.hh>

#include "main.cc"

namespace adamantine
{

class ScanPathTester
{
public:
  std::vector<ScanPathSegment> get_segment_format_list()
  {
    ScanPath scan_path("scan_path.txt", "segment");
    return scan_path._segment_list;
  };
  std::vector<ScanPathSegment> get_event_series_format_list()
  {
    ScanPath scan_path("scan_path_event_series.inp", "event_series");
    return scan_path._segment_list;
  };
};

BOOST_AUTO_TEST_CASE(scan_path)
{
  ScanPathTester tester;

  // Test the segments from a ScanPathFileFormat::segment file
  std::vector<ScanPathSegment> segment_format_list =
      tester.get_segment_format_list();

  double const tolerance = 1e-12;

  BOOST_CHECK(segment_format_list.size() == 2);

  BOOST_CHECK_CLOSE(segment_format_list[0].end_time, 1.0e-6, tolerance);
  BOOST_CHECK_CLOSE(segment_format_list[0].end_point[0], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_format_list[0].end_point[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_format_list[0].power_modifier, 0.0, tolerance);

  BOOST_CHECK_CLOSE(segment_format_list[1].end_time, 1.0e-6 + 0.002 / 0.8,
                    tolerance);
  BOOST_CHECK_CLOSE(segment_format_list[1].end_point[0], 0.002, tolerance);
  BOOST_CHECK_CLOSE(segment_format_list[1].end_point[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_format_list[1].power_modifier, 1.0, tolerance);

  // Test the segments from a ScanPathFileFormat::event_series file
  std::vector<ScanPathSegment> segment_event_series_list =
      tester.get_event_series_format_list();

  BOOST_CHECK(segment_event_series_list.size() == 3);

  BOOST_CHECK_CLOSE(segment_event_series_list[0].end_time, 0.1, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[0].end_point[0], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[0].end_point[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[0].power_modifier, 0.0,
                    tolerance);

  BOOST_CHECK_CLOSE(segment_event_series_list[1].end_time, 1.0, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[1].end_point[0], 0.5, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[1].end_point[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[1].power_modifier, 1.0,
                    tolerance);

  BOOST_CHECK_CLOSE(segment_event_series_list[2].end_time, 2.0, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[2].end_point[0], 0.5, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[2].end_point[1], 0.5, tolerance);
  BOOST_CHECK_CLOSE(segment_event_series_list[2].power_modifier, 0.5,
                    tolerance);
}

BOOST_AUTO_TEST_CASE(scan_path_location)
{
  double const tolerance = 1e-10;

  ScanPath scan_path("scan_path.txt", "segment");
  double time = 1.0e-7;
  dealii::Point<3> p1 = scan_path.value(time);

  BOOST_CHECK_CLOSE(p1[0], 0.0, tolerance);
  BOOST_CHECK_CLOSE(p1[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(p1[2], 0.0, tolerance);

  time = 0.001001;

  dealii::Point<3> p2 = scan_path.value(time);

  BOOST_CHECK_CLOSE(p2[0], 8.0e-4, tolerance);
  BOOST_CHECK_CLOSE(p2[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(p2[2], 0.0, tolerance);
}

} // namespace adamantine
