/* Copyright (c) 2016 - 2020, the adamantine authors.
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
  std::vector<ScanPathSegment> get_segment_list()
  {
    ScanPath scan_path("scan_path.txt");
    return scan_path._segment_list;
  };
};

BOOST_AUTO_TEST_CASE(scan_path)
{
  ScanPathTester tester;
  std::vector<ScanPathSegment> segment_list = tester.get_segment_list();

  double const tolerance = 1e-12;

  BOOST_CHECK(segment_list.size() == 2);

  BOOST_CHECK_CLOSE(segment_list[0].end_time, 1.0e-6, tolerance);
  BOOST_CHECK_CLOSE(segment_list[0].end_point[0], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_list[0].end_point[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_list[0].power_modifier, 0.0, tolerance);

  BOOST_CHECK_CLOSE(segment_list[1].end_time, 1.0e-6 + 0.002 / 0.8, tolerance);
  BOOST_CHECK_CLOSE(segment_list[1].end_point[0], 0.002, tolerance);
  BOOST_CHECK_CLOSE(segment_list[1].end_point[1], 0.0, tolerance);
  BOOST_CHECK_CLOSE(segment_list[1].power_modifier, 1.0, tolerance);
}

BOOST_AUTO_TEST_CASE(scan_path_location)
{
    double const tolerance = 1e-10;

    ScanPath scan_path("scan_path.txt");
    dealii::Point<1> time(1.0e-7);
    double x1 = scan_path.value(time,0);
    double y1 = scan_path.value(time,2);

    BOOST_CHECK_CLOSE(x1, 0.0, tolerance);
    BOOST_CHECK_CLOSE(y1, 0.0, tolerance);

    time[0] = 0.001001;
    double x2 = scan_path.value(time,0);
    double y2 = scan_path.value(time,2);

    BOOST_CHECK_CLOSE(x2, 8.0e-4, tolerance);
    BOOST_CHECK_CLOSE(y2, 0.0, tolerance);
}

} // namespace adamantine
