---
layout: default
parent: Run
title: Scan path
nav_order: 2
usemathjax: true
---

# Scan path
*adamantine* supports two kinds of scan path input: the `segment` format and the
`event` format.

## Segment format
After the self-explainatory tree-line header, the column descriptions are:
* Column 1: mode 0 for line mode, mode 1 for spot mode
* Columns 2 to 4: (x,y,z) coordinates in units of m. For line mode, this
is the ending position of the the line.
* Column 5: the coefficient for the nominal power. Usually this is either
0 or 1, but sometimes intermediate values are used when turning a corner.
* Column 6: in spot mode, this is the dwell time in seconds, in line mode
this is the velocity in m/s.

The first entry must be a spot. If it was a line, there would be no way
to know where the line starts (since the coordinates are the ending coordinates).
By convention, we avoid using a zero second dwell time for the first spot
and instead choose some small positive number.

An example of such a file can be found
[here](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/scan_path_L.txt).

## Event format
For an event series the first segment is a point, then the rest are lines.
The column descriptions are:
* Column 1: segment endtime
* Columns 2 to 4: (x,y,z) coordinates in units of m. This is the ending
position of the line.
* Column 5: the coefficient for the nominal power. Usually this is either
0 or 1, but sometimes intermediate values are used when turning a corner.

An example of such a file can be found
[here](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/scan_path_event_series.inp).
