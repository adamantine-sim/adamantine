---
layout: default
parent: Examples
title: Demo316
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

# Demo316
This example shows thermal simulation of material deposition on a plate.

The example is composed of the following files:
 * **demo_316_short.info:** the input file
 * **demo_316_short_scan_path.txt:** the scan path of the heat source

Below are snapshots of the temperature at different times:

<div class="row">
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/demo_316/demo_316_0.png?raw=true" style="width:100%">
   Temperature at t = 0s
 </div>
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/demo_316/demo_316_1.png?raw=true" style="width:100%">
   Temperature at t = 6e-4s
 </div>
</div> 
<div class="row">
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/demo_316/demo_316_2.png?raw=true" style="width:100%">
   Temperature at t = 12e-4s
 </div>
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/demo_316/demo_316_3.png?raw=true" style="width:100%">
   Temperature at t = 18e-4s
 </div>
</div> 
<div class="row">
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/demo_316/demo_316_4.png?raw=true" style="width:100%">
   Temperature at t = 24e-4s
 </div>
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/demo_316/demo_316_5.png?raw=true" style="width:100%">
   Temperature at t = 20e-4s
 </div>
</div> 




# Demo316 AMR
This example is similar to the previous one but it uses adaptive mesh refinement
(AMR). 

The example is composed of the following files:
 * **demo_316_short_amr.info:** the input file
 * **demo_316_short_scan_path_amr.txt:** the scan path of the heat source

The main difference with *Demo316* concerns the refinement input:
```
refinement
{
  n_refinements 1                 ; Number of time the cells on the paths of the beams are refined.
  time_steps_between_refinement 5 ; Number of time steps after which the refinement process is
                                  ; performed.
  coarsen_after_beam true         ; The cells are coarsen once the beam has passed
}
```
The other differences with *Demo316* are:
 * the domain size is reduced and the number of cells is reduced accordingly
 * the time step is increased from 6e-5 *s* to 1e-4 *s*
 * the order of the finite element is decrease from 3 to 2

# Demo316 anisotropic
This is similar to *Demo316* but the thermal conductivity of the material is
anisotropic. The conductivity in the deposition direction is increased.

This example is composed of the following files:
 * **demo_316_short_anisotropic.info:** the input file
 * **demo_316_short_scan_path.txt:** the scan path of the heat source

The differences with *Demo316* are:
 * the domain is reduced
 * the boundary conditions are changed from convective and radiative to
 adiabatic
 * the thermal conductivity in the deposition direction, *x*, is increased for
 the solid, liquid, and powder phases. This is not physical but it is done to
 test the anisotropic capabilities of *adamantine*
 * the duration of the simulation is reduced from 4e-3 *s* to 1e-4 *s*
 * the time step is reduced from 6e-5 *s* to 1e-5 *s*
 * the order of the finite element is decrease from 3 to 2
