/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

 #define BOOST_TEST_MODULE ParseScanPath

 //#include <ParseScanPath.hh>
 #include <HeatSource.hh>

 #include "main.cc"

namespace adamantine {

class HeatSourceTester{
public:
    std::vector<ScanPathSegment> get_segment_list(){
        boost::property_tree::ptree database;

        database.put("depth", 0.1);
        database.put("absorption_efficiency", 0.1);
        database.put("diameter", 1.0);
        database.put("max_power", 10.);
        database.put("abscissa", "t");
        HeatSource<2> heat_source(database);
        return heat_source._segment_list;
    };
};

 BOOST_AUTO_TEST_CASE(parse_scan_path)
 {
     HeatSourceTester tester;
     std::vector<ScanPathSegment> segment_list = tester.get_segment_list();

     double const tolerance = 1e-12;

     BOOST_CHECK(segment_list.size() == 2);

     BOOST_CHECK_CLOSE(segment_list[0].end_time, 1.0e-6, tolerance);
     BOOST_CHECK_CLOSE(segment_list[0].end_point[0], 0.0, tolerance);
     BOOST_CHECK_CLOSE(segment_list[0].end_point[1], 0.0, tolerance);
     BOOST_CHECK_CLOSE(segment_list[0].power_modifier, 0.0, tolerance);

     BOOST_CHECK_CLOSE(segment_list[1].end_time, 1.0e-6 + 0.002/0.8, tolerance);
     BOOST_CHECK_CLOSE(segment_list[1].end_point[0], 0.002, tolerance);
     BOOST_CHECK_CLOSE(segment_list[1].end_point[1], 0.0, tolerance);
     BOOST_CHECK_CLOSE(segment_list[1].power_modifier, 1.0, tolerance);
 }
}
