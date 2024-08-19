---
layout: default
parent: Run
title: Material deposition
nav_order: 3
usemathjax: true
---

# Material deposition
*adamantine* supports two ways to deposit material: based on the scan path and
based on a separate material deposition file.

## Scan-path-based deposition
If the material deposition is based on the scan path, then material is added
according to the `deposition_length`, `deposition_width`, `deposition_height`, 
`deposition_lead_time`, and `deposition_time` input parameters in the `geometry` 
input block. Cells are activated if they are crossed by a rectangular prism 
(rectangle in 2D) traveling along the scan path. In 3D the rectangular prism 
is centered with the (x,y) values of the scan path and the top of the rectangular 
prism is at the z value of the scan path (i.e. the scan path height gives the 
new height of the material after deposition). Near the end of segments, the 
length of the rectangular prism is truncated so as to not deposit material 
past the edge of the segment. Material can be deposited with a lead time ahead 
of the heat source (controlled by `deposition_lead_time`). Depositing material 
requires modifying the simulation mesh, which can be computationally intensive. 
To reduce the cost, material can be added in "lumps", with the time between 
lumps set by `deposition_time`.

## File-based deposition
The material deposition can also be set by boxes defined in a separate
file. The format of this file is as follows.

The first entry of the file is the dimension the problem: 2 or 3.
* For 2D problems, the column descriptions are:
  * Column 1 to 2: (x,y) coordinates of the center of the deposition box in m.
  * Column 3 to 4: (x,y) length of deposition box in m.
  * Column 5: deposition time in s.
  * Column 6: angle of material deposition.
* For 3D problems, the column descriptions are:
  * Column 1 to 3: (x,y,z) coordinates of the center of the deposition box in m.
  * Column 4 to 6: (x,y,z) length of deposition box in m.
  * Column 7: deposition time in s.
  * Column 6: angle of material deposition.
