/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef SCANPATHPARSER_HH
#define SCANPATHPARSER_HH

#include <vector>
#include <istream>

enum class ScanPathSegmentType {line, point};

struct ScanPathSegment {
    ScanPathSegmentType type;
    double segment_start_time;
    double segment_end_time;
    double velocity;
    double power_modifier;
};

std::vector<ScanPathSegment> ParseScanPath(std::string scan_path_file) {

    std::vector<ScanPathSegment> segments;

    // Open the file


    return segments;

};

#endif
