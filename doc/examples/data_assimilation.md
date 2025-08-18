---
layout: default
parent: Examples
title: Data Assimilation
nav_order: 3
usemathjax: true
---

# Data Assimilation
These examples are very similar to the [thermal examples]({{site.baseurl}}/doc/examples/thermal_simulation) 
but we will use them to show how data assimilation works in
*adamantine*.

## Bare plate L: ensemble simulation
This example shows how to run an ensemble of simulations. Ensemble simulations are 
used to compute the simulation covariance matrix (see 
[Data Assimilation]({{site.baseurl}}/doc/data_assimilation)).

The example is composed of the following files:
 * [bare_plate_L_ensemble.info](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_ensemble.info): the input file
 * [bare_plate_L_scan_path.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_scan_path.txt): the scan path of the heat source

It can be run using `./adamantine -i bare_plate_L_data.info`.

The interesting part of the input file is:
```
ensemble
{
    ensemble_simulation true
    ensemble_size 3
    materials 
    {
      initial_temperature_stddev 10.0
    }
}
```
The first parameter, `ensemble_simulation`, enables the ensemble simulations. The
second parameter, `ensemble_size`, determines the number of simulations that
will be run, in this case three. The third paremeter, `initial_temperature_stddev`, 
sets the initial temperature to a normal distribution with a standard
distribution of ten. The mean of the distribution is 300 K.

When launching *adamantine*, three simulations using different initial
temperature will be run. The initial temperatures are chosen randomly using a
normal distribution with a standard deviation of ten. The values choses are
written in the files `output_mX_data.txt` where `X` is 0, 1, or 2.The output 
files are named `output_mX.*`. Each number is associated with a single
simulation. In *paraview* or *VisIt*, we will open `output_m0.pvd`,
`output_m1.pvd`, and `output_m2.pvd`.

The first simulation, `m0`, has an initial temperature of 301.3 K. The maximum
temperature at the end of the simulation is 676.2 K.
<img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/bare_plate_L/bare_plate_L_m0.png?raw=true" style="width:100%">

The second simulation, `m1`, has an initial temperature of 298.5 K. The maximum
temperature at the end of the simulation is 673.4 K.
<img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/bare_plate_L/bare_plate_L_m1.png?raw=true" style="width:100%">

The third simulation, `m2`, has an initial temperature of 304.6 K. The maximum
temperature at the end of the simulation is 679.5 K.
<img src="https://github.com/adamantine-sim/website-assets/blob/master/examples/bare_plate_L/bare_plate_L_m2.png?raw=true" style="width:100%">


## Bare plate L: data assimilation
This example shows how *adamantine* can use experimental data through data
assimilation.

The example is composed of the following files:
 * [bare_plate_L_da.info](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_da.info): the input file
 * [bare_plate_L_scan_path.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_scan_path.txt): the scan path of the heat source
 * [bare_plate_L_expt_data_0_0.csv](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_expt_data_0_0.csv) and [bare_plate_L_expt_data_0_1.csv](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_expt_data_0_1.csv):
 point cloud synthetic data that represent experimental data
 * [bare_plate_L_expt_log.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_expt_log.txt): the log file that lists the timestamps for each
 frame from each camera

It can be run using `./adamantine -i bare_plate`.

The interesting parts of the input files are:
```
experiment
{
  read_in_experimental_data true
  log_filename bare_plate_L_expt_log.txt
  file bare_plate_L_expt_data_#camera_#frame.csv
  format point_cloud
  first_frame 0
  last_frame 1
  first_camera_id 0
  last_camera_id 0
  first_frame_temporal_offset 0.0105
  estimated_uncertainty 5.0
}
```
and
```
data_assimilation
{
  assimilate_data true
  localization_cutoff_function gaspari_cohn
  localization_cutoff_distance 1.0e-3
  solver
  {
    max_number_of_temp_vectors 10
    convergence_tolerance 1.0e-8
  }
}
```

The `experiment` input contains the filename and the format of the experimental
data. It is also here that the uncertainty used to build the observation error
covariance, $$R$$, is set.

The `data_assimilation` input contains the parameters for the localization
cutoff and for the solver used to compute the Kalman gain, $$K$$.

## Bare plate L: augmented data assimilation
This example shows how data assimilation can be used to determine simulation 
parameters such as material properties or the power of the heat source.

The example is composed of the following files:
 * [bare_plate_L_da_augmented.info](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_da_augmented.info): the input file
 * [bare_plate_L_scan_path.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_scan_path.txt): the scan path of the heat source
 * [bare_plate_L_da_aug_ref_data_0_0.csv](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_da_aug_ref_data_0_0.csv) and 
 [bare_plate_L_da_aug_ref_data_0_1.csv](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_da_aug_ref_data_0_1.csv):
 point cloud synthetic data that represent experimental data
 * [bare_plate_L_da_aug_ref_log.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/bare_plate_L_da_aug_ref_log.txt): the log file that lists the timestamps for each

The interesting part of the input files are:
```
ensemble
{
  ensemble_simulation true
  ensemble_size 3
  sources
  {
    beam_0
    {
      absorption_efficiency_stddev 0.1
    }
  }
}
```
and
```
data_assimilation
{
  assimilate_data true
  localization_cutoff_function gaspari_cohn
  localization_cutoff_distance 1.0e-3
  augment_with_beam_0_absorption true
  solver
  {
    max_number_of_temp_vectors 10
    convergence_tolerance 1.0e-8
  }
}
```

In `ensemble`, we have replaced replace the uncertainty of the initial
temperature with the uncertainty on the absorption of the heat source.

In `data_assimilation`, we have added `augment_with_beam_0_absorption true`.
With this option enabled, *adamantine* will try to improved on the value of the
absorption of the heat source.
