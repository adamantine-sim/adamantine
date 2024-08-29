---
layout: page
title: Visualization
nav_order: 4
---

# Visualization
*adamantine* produces many output files with different extensions *.vtu*,
*.pvtu*, and *.pvd*. These files can be opened using 
[ParaView](https://www.paraview.org/) or
[VisIt](https://visit-dav.github.io/visit-website/). The *.vtu* files contain
the output data for a given time step and a given processor. The *.pvtu* files
contain metadata that points to the *.vtu*. There is one *.pvtu* for each time
step. Finally, the *.pvd* file contains metadata that points to the *.pvtu*
files. By opening this single file in *ParaView* or *VisIt*, you will be able to
visualize the entire simulation.

## VisIt
When visualizing the output, the entire domain including cells that have not
been activated yet are visible. To only visualize the activated cells, do the
following:
 - Add the **Threshold** operator. This operator is part of the **Selection**
 operator.
 - Open the **Threshild operator attributes** and delete the default selected
 variable.
 - Add a new variable. Choose **Scalars**, then **temperature**.
 - Modify the **Lower bound** to 1.
 - Select **Apply**, then click **Draw**.

This will filter out all the cells with a temperature under one. Since the
inactive cells have a temperature of zero, they will not be shown.
