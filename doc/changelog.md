---
layout: page
title: Changelog
nav_order: 6
---

# Changelog

## [Version 1.1](https://github.com/adamantine-sim/adamantine/tree/release/1.1) (2025-01-23)

### General Enhancements
 - Add capability to read updates to the scan path during a simulation: https://github.com/adamantine-sim/adamantine/pull/291 and https://github.com/adamantine-sim/adamantine/pull/294 
 - Output warning on the screen instead of crashing if no experimental point is found during data assimilation https://github.com/adamantine-sim/adamantine/pull/315
 - Skip empty lines in scan path file https://github.com/adamantine-sim/adamantine/pull/330
 - Support MPI for (thermo-)mechanical simulations without adaptivity https://github.com/adamantine-sim/adamantine/pull/301 and https://github.com/adamantine-sim/adamantine/pull/333
 - Relicense the code from 3-clause BSD to Apache 2.0 with LLVM-exception https://github.com/adamantine-sim/adamantine/pull/346
 - Do not exit when unknown command line arguments are passed https://github.com/adamantine-sim/adamantine/pull/351
 - Make sure that ensemble variables are not correlated and that they are always positive https://github.com/adamantine-sim/adamantine/pull/353 
 - Write in a file the variables used by each ensemble member https://github.com/adamantine-sim/adamantine/pull/353
 - Speedup adaptive mesh refinement and remove `beam_cutoff` https://github.com/adamantine-sim/adamantine/pull/356
 - Require deal.II 9.6 instead of version 9.5 https://github.com/adamantine-sim/adamantine/pull/359
 - Extend data assimilation to all the scalar inputs https://github.com/adamantine-sim/adamantine/pull/361
 - Stop compiling adamantine  with `-ffast-math` when building the docker image https://github.com/adamantine-sim/adamantine/pull/370
 - Improve data assimilation when the cells are not aligned with the axis https://github.com/adamantine-sim/adamantine/pull/375
 - Add Nix build https://github.com/adamantine-sim/adamantine/pull/340 and https://github.com/adamantine-sim/adamantine/pull/365
 - By default, output the temperature just before and after data assimilation. This new behavior can be turned off using the `postprocessor.output_on_data_assimilation` input https://github.com/adamantine-sim/adamantine/pull/377
 - Vectorize the computation of the heat source. Use `-march=native` when compiling deal.II and adamantine https://github.com/adamantine-sim/adamantine/pull/383
 - Output temperature gradients and cooling rates at the liquidus in a text file https://github.com/adamantine-sim/adamantine/pull/384
 - Allow different boundary conditions on faces with different boundary ids https://github.com/adamantine-sim/adamantine/pull/388
 - Introduce new `clamped` boundary conditions for (thermo-)mechanical simulations https://github.com/adamantine-sim/adamantine/pull/393
 - Introduce new `traction_free` boundary conditions for (thermo-)mechanical simulations and a new `printed_surface` boundary https://github.com/adamantine-sim/adamantine/pull/395
 - Implicit time stepping methods have been removed https://github.com/adamantine-sim/adamantine/pull/403
 - Update format of the experimental log file to use absolute time  https://github.com/adamantine-sim/adamantine/pull/399
 - Add volume integration of the total heat input to the system for monitoring and verification.https://github.com/adamantine-sim/adamantine/pull/427
 - Output `von Mises` stress instead of the norm of the stress https://github.com/adamantine-sim/adamantine/pull/429

### Bug Fixes
 - Fix a bug when using more than than one MPI process by ensemble member where the data on MPI ranks different than zero was cutoff by the localization function https://github.com/adamantine-sim/adamantine/pull/339
 - Fix a bug where the number of material states was off by one https://github.com/adamantine-sim/adamantine/pull/349
 - Fix a bug where all the processors would write the `pvd` file which would cause the file to be ill formed https://github.com/adamantine-sim/adamantine/pull/355
 - Fix table input format by changing the delimiter from `;` to `|` https://github.com/adamantine-sim/adamantine/pull/358
 - Fix a bug where too many processors would try to write the experimental data to files which would then be ill formed https://github.com/adamantine-sim/adamantine/pull/378
 - Fix a bug where the average state ratios were not computed over all the quadrature points https://github.com/adamantine-sim/adamantine/pull/386
 - Fix a bug where radiative and convective boundary where multiplied by the temperature when they should not have been  https://github.com/adamantine-sim/adamantine/pull/389
 - Fix a division by zero in spot mode when the source is turned on https://github.com/adamantine-sim/adamantine/pull/397
 - Add missing data to checkpoint/restart file. The missing data would sometimes trigger a segfault on restart https://github.com/adamantine-sim/adamantine/pull/406
 - Solve the mechanical problem at each time step instead of solving only at output https://github.com/adamantine-sim/adamantine/pull/423


## [Version 1.0](https://github.com/adamantine-sim/adamantine/tree/release/1.1) (2024/10/01)
* Initial Release
