---
layout: default
parent: Examples
title: HourGlass
nav_order: 1
---

<head>
<style>
* {
  box-sizing: border-box;
}

.column {
  float: left;
  width: 50%;
  padding: 5px;
}

/* Clearfix (clear floats) */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>
</head>


# HourGlass
This example is composed of the following files:
 * **HourGlass\_AOP.info:** the input file
 * **HourGlass\_AOP.vtk:** the mesh
 * **HourGlass\_AOP\_scan\_path.txt:** the scan path of the heat source

The domain is an hour glass with an hole in the center.

<div class="row">
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_mesh_side.png?raw=true" style="width:100%">
 </div>
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_mesh_top.png?raw=true" style="width:100%">
 </div>
</div> 
<div class="row">
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_mesh_slice.png?raw=true" style="width:100%">
 </div>
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_mesh_slice_2.png?raw=true" style="width:100%">
 </div>
</div> 

This example is a purely thermal problem. By changing the
`time_stepping.duration` parameter, you can choose how much of the manufacturing
process is simulated. To build the entire hour glass, adjust the `duration` to
40610 seconds.
