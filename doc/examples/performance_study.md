---
layout: default
parent: Examples
title: Performance Study
nav_order: 2
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


# Performance Study
We will use this example to understand which parameters affects the performance
of *adamantine*.

This example is composed of the following files:
 * [HourGlass_AOP.info](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/HourGlass_AOP.info): the input file
 * [HourGlass_AOP.vtk](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/HourGlass_AOP.vtk): the mesh
 * [HourGlass_AOP_scan_path.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/HourGlass_AOP_scan_path.txt): the scan path of the heat source

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

To understand where most of the time is spent, we will use [Caliper](http://software.llnl.gov/Caliper).
First, we run **HourGlass_AOP.info** using a single processor: 
`mpirun -np 1 ./adamantine -i HourGlass_AOP.info`.
*Caliper* returns the following results:

|Path                                                                 | Min time/rank | Max time/rank | Avg time/rank | Time % | 
|:--------------------------------------------------------------------|:-------------:|:-------------:|:-------------:|:------:|
|main                                                                 |    106.35     |    106.35     |    106.35     | 100.00 | 
|&nbsp;&nbsp;run                                                      |    106.35     |    106.35     |    106.35     |  99.99 | 
|&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                                  |      0.17     |      0.17     |      0.17     |   0.16 | 
|&nbsp;&nbsp;&nbsp;&nbsp;main_loop                                    |    101.01     |    101.01     |    101.01     |  94.97 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine_mesh                      |      0.06     |      0.06     |      0.06     |   0.06 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;add_material                     |     78.80     |     78.80     |     78.80     |  74.09 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine triangulation |     31.98     |     31.98     |     31.98     |  30.07 |
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;evaluate_thermal_physics         |      0.78     |      0.78     |      0.78     |   0.73 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                      |     16.57     |     16.57     |     16.57     |  15.58 | 

We see that the simulation took 105 seconds of which 74% is spent in `add_material`.
The core of computation is `evaluate_thermal_physics` which took less than 1% of
the total simulation time. Writing the output files, `output_pvtu`, took another
15%. Neither `add_material` or `output_pvtu` scale well with the number of processors.
Using more processors will not improve the performance significantly. Moreover
there are only 1241 degrees of freedom which is too small to take advantage of
more processors. 

Let's modified **HourGlass_AOP.info**:
 - add `deposition_time 25` inside `geometry`. Using this option, the material 
 is deposited in such a way that it will take 20 seconds for the heat source 
 to reach the end of the deposition. This decreases the number of time, material
 is added which will greatly speedup the simulation.
 - change `time_steps_between_output` from 10 to 500.
 - change `duration` to 500 s. By running the simulation for a longer time, more
 cells will be activated and we will have more degrees of freedom to spread
 between the processors.

When using one processor, we get:

|Path                                                                 | Min time/rank | Max time/rank | Avg time/rank | Time % | 
|:--------------------------------------------------------------------|:-------------:|:-------------:|:-------------:|:------:|
|main                                                                 |    144.40     |     144.40    |     144.40    | 100.00 | 
|&nbsp;&nbsp;run                                                      |    144.40     |     144.40    |     144.40    |  99.99 | 
|&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                                  |      0.18     |       0.18    |       0.18    |   0.12 | 
|&nbsp;&nbsp;&nbsp;&nbsp;main_loop                                    |    139.07     |     139.07    |     139.07    |  96.31 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine_mesh                      |      0.06     |       0.06    |       0.06    |   0.04 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;add_material                     |     35.96     |      35.96    |      35.96    |  24.90 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine triangulation |     14.68     |      14.68    |      14.68    |  10.16 |
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;evaluate_thermal_physics         |     81.45     |      81.45    |      81.45    |  56.40 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                      |     16.81     |      16.81    |      16.81    |  11.64 |

At the end of simulation, we have 3932 degrees of freedom. We now spend the
majority of the computation time in `evaluate_thermal_physics`. We can compare
the results when using two processors (`mpirun -np 2 ./adamantine -i HourGlass_AOP.info`):

|Path                                                                 | Min time/rank | Max time/rank | Avg time/rank | Time % | 
|:--------------------------------------------------------------------|:-------------:|:-------------:|:-------------:|:------:|
|main                                                                 |    104.54     |    104.55     |   104.55      | 100.00 | 
|&nbsp;&nbsp;run                                                      |    104.54     |    104.55     |   104.55      |  99.99 | 
|&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                                  |      0.09     |      0.10     |     0.10      |   0.09 | 
|&nbsp;&nbsp;&nbsp;&nbsp;main_loop                                    |     99.78     |     99.78     |    99.78      |  95.44 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine_mesh                      |      0.03     |      0.03     |     0.03      |   0.03 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;add_material                     |     33.24     |     33.90     |    33.57      |  32.11 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine triangulation |     13.63     |     13.65     |    13.64      |  13.05 |
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;evaluate_thermal_physics         |     46.17     |     59.29     |    52.73      |  50.43 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                      |      5.11     |     15.74     |    10.43      |   9.97 |

By using two processors instead of one, we get a speed up of 1.38. Notice that
the max time/rank for `add_material` and `output_pvtu` is almost the same 
whether we use one processor or two. The speedup for `add_material` and
`output_pvtu` is about 1.06. We will explain later on why these functions 
scale so poorly. The speedup for `evaluate_thermal_physics` is 1.37. This is 
far away from a perfect speedup of two but it is due to relatively small number
of degrees of freedom during the simulation. The timings show are aggregated 
over the entire simulation and therefore the speedup is negatively impacted by
the first few steps which have very small number of degrees of freedom. The 
longer the simulation runs the better the speedup, we get. To show this, we 
will change the `duration` to 1500 s.

When using one processor, we get:

|Path                                                                 | Min time/rank | Max time/rank | Avg time/rank | Time % | 
|:--------------------------------------------------------------------|:-------------:|:-------------:|:-------------:|:------:|
|main                                                                 |    626.06     |    626.06     |    626.06     | 100.00 | 
|&nbsp;&nbsp;run                                                      |    626.06     |    626.06     |    626.06     |  99.99 | 
|&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                                  |      0.17     |      0.17     |      0.17     |   0.02 | 
|&nbsp;&nbsp;&nbsp;&nbsp;main_loop                                    |    620.71     |    620.71     |    620.71     |  99.14 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine_mesh                      |      0.06     |      0.06     |      0.06     |   0.01 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;add_material                     |    108.49     |    108.49     |    108.49     |  17.33 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine triangulation |     43.96     |     43.96     |     43.96     |   7.02 |
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;evaluate_thermal_physics         |    447.18     |    447.18     |    447.18     |  71.42 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                      |     50.18     |     50.18     |     50.18     |   8.01 |

At the end of simulation, we have 9471 degrees of freedom. We spend 71% of the
time in `evaluate_thermal_physics` while it was only 56% for the 500 s 
simulation. This is because the amount of work that is performed in `add_material` 
and `output_pvtu` does not increase as quickly with the number of degrees of
freedom as it does in `evaluate_thermal_physics`. Therefore `add_material` and
`output_pvtu` become relatively cheaper over the course of the simulation. Now
let's take a look at the results when using two processors:

|Path                                                                 | Min time/rank | Max time/rank | Avg time/rank | Time % | 
|:--------------------------------------------------------------------|:-------------:|:-------------:|:-------------:|:------:|
|main                                                                 |    435.47     |    435.47     |    435.47     | 100.00 | 
|&nbsp;&nbsp;run                                                      |    435.47     |    435.47     |    435.47     |  99.99 | 
|&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                                  |      0.08     |      0.09     |      0.09     |   0.02 | 
|&nbsp;&nbsp;&nbsp;&nbsp;main_loop                                    |    430.74     |    430.74     |    430.74     |  98.91 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine_mesh                      |      0.03     |      0.03     |      0.03     |   0.00 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;add_material                     |    100.17     |    101.89     |    101.03     |  23.20 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;refine triangulation |     41.07     |     41.18     |     41.13     |   9.44 |
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;evaluate_thermal_physics         |    268.94     |    307.77     |    288.36     |  66.21 | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output_pvtu                      |     16.05     |     47.03     |     31.54     |   7.24 |

For the entire simulation, we have a speedup of 1.43. For `add_material` and
`output_pvtu`, the speedup is unchanged at 1.06. This is consistent with the
claim that the work done in these functions does not scale with the number of
degrees of freedom. Finally, the speedup from `evaluate_thermal_physics` is
1.45. As expected, the speedup increases with the duration of the simulation.

Now let's discuss the difference in scaling between `add_material`,
`output_pvtu`, and `evaluate_thermal_physics`. The main reason of the poor
scaling of the first two functions is due to load balancing. There
are two different strategies to partition the mesh:
 1. partition all the cells equally between the processors: the issue with this
    strategy is that all the active cells may be attributed to a single
    processor. The other processors have no active cells and no work to perform
    in `evaluate_thermal_physics`. In the worse case, a single processor will
    have active cells until half the simulation is done.
 2. partition all the **active** cells equally between the processors: this is
    the strategy that we have chosen. In this case, all the processors will have
    work to do in `evaluate_thermal_physics`. The issue is that the work done in 
    `add_material` and `output_pvtu` is on the entire mesh. Their work is not restricted
    to the active cells. This explains why there are not as sensitive to the
    number of degrees of freedom, i.e., the number of active cells as
    `evaluate_thermal_physics. Since the partitioning algorithm only cares about
    the active cells, all the non-active cells are attributed to one 
    processor which has a lot more operations to do. This explains the poor
    scaling of these functions. 

We can see the effect describe above by looking the partitioning of the mesh at
the end of the 1500 s simulation. First, we look at the partitioning of the
active cells:
<div class="row">
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_subdomain_2d_bottom.png?raw=true" style="width:100%">
   Bottom view of the mesh
 </div>
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_subdomain_2d_top.png?raw=true" style="width:100%">
   Top view of the mesh
 </div>
</div> 
The cells are equally distributed between the two processors. Now let's look at
the partitioning of the entire mesh:
<div class="row">
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_subdomain_3d_side_1.png?raw=true" style="width:100%">
 </div>
 <div class="column">
   <img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/hour_glass/hour_glass_subdomain_3d_side_2.png?raw=true" style="width:100%">
 </div>
</div> 
All the inactive cells have been given to processor 1.

Over the course of the simulation, more and more cells are activated and the
load balancing improves.
