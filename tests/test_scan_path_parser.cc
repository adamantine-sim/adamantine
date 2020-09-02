/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

 #define BOOST_TEST_MODULE ScanPathParser

 #include <ScanPathParser.hh>

 #include "main.cc"

 BOOST_AUTO_TEST_CASE(scan_path_parser)
 {
     std::string scan_path_file = "";
     std::vector<ScanPathSegment> segment_list = ParseScanPath(scan_path_file);

     BOOST_CHECK(segment_list.size() == 5);
 }
