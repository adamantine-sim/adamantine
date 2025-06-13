---
layout: default
parent: Run
title: Input file
nav_order: 1
usemathjax: true
---

# Input file
*adamantine* supports Boost INFO format and json. The input file is assumed to use the INFO format unless the file extension is `.json`.
The following options are available:

## boundary (required)
* **type**: type of boundary: `adiabatic`, `radiative`, or `convective`. Multiple types
can be chosen simultaneously by separating them by comma (required)

## physics (required):
* **thermal**: thermal simulation: true or false (required)
* **mechanical**: mechanical simulation: true or false (required)
* if both thermal and mechanical parameters are true, solve a coupled thermo-mechanics problem

## discretization (required):
* **thermal** (required if *physics.thermal* is true):
  * **fe\_degree**: degree of the finite element used (required)
  * **quadrature**: quadrature used: `gauss` or `lobatto` (default value: `gauss`)
* **mechanical** (required if *physics.mechanical* is true):
  * **fe\_degree**: degree of the finite element used (required)

## geometry (required):
* **dim**: the dimension of the problem (2 or 3, required)
* **material\_height**: below this height the domain contains material. Above this
height the domain is empty. The height is in meters (default value: 1e9)
* **use\_powder**: the additive manufacturing process use powder: true or false
(default value: false)
* if use\_powder is true:
  * **powder\_layer**: thickness of the initial layer of powder in meters (required)
* **material\_deposition**: material is deposed during the simulation: true or
false (default value: false)
* if material\_deposition is true:
  * **material\_deposition\_method**: `file` or `scan_paths`
  * if material\_deposition\_method is file:
      * **material\_deposition\_file**: material deposition filename
  * if material\_deposition\_method is scan\_paths:
      * **deposition\_length**: length of material deposition boxes along the scan direction in meters (required)
      * **deposition\_width**: width of material deposition boxes (in the plane of the material, normal to the scan direction, 3D only) in meters (required)
      * **deposition\_height**: height of material deposition boxes (out of the plane of the material) in meters (required)
      * **deposition\_lead\_time**: amount of time before the scan path reaches a point that the material is added in seconds (required)
      * **deposition\_time**: add the material in bigger lumps in seconds (optional)
* **import\_mesh**: true or false (required)
* if import\_mesh is true:
  * **mesh\_file**: The filename for the mesh file (required)
  * **mesh\_format**: `abaqus`, `assimp`, `unv`, `ucd`, `dbmesh`, `gmsh`, `tecplot`, `xda`, `vtk`,
  `vtu`, `exodus`, or `default`, i.e., use the file suffix to try to determine the
  mesh format (required)
  * **mesh\_scale\_factor**: Apply a uniform scaling factor to the mesh (e.g. if the mesh is defined
  in mm or inches instead of m) (default value: 1, <span style="color:red">removed in 1.1, use `units.mesh` instead</span>)
  * **reset\_material\_id**: Clear the material IDs defined in the mesh and set them all to zero so 
    all material properties are given by the `material_0` input block: true or false (default value: false)
* if import\_mesh is false:
  * **length**: the length of the domain in meters (required)
  * **height**: the height of the domain in meters (required)
  * **width**: the width of the domain in meters (only in 3D)
  * **length\_origin**: the reference location in the length direction (default value: 0)
  * **height\_origin**: the reference location in the height direction (default value: 0)
  * **width\_origin**: the reference location in the width direction (only in 3D) (default value: 0)
  * **length\_divisions**: number of cell layers in length (default value: 10)
  * **height\_divisions**: number of cell layers in the height (default value: 10)
  * **width\_divisions**: number of cell layers in width (only in 3D) (default value: 10)

## materials (required):
* **n\_materials**: number of materials (required)
* **property\_format**: format of the material property: `table` or `polynomial`. For `table`, the format of the matieral properties is as follows: `temperature_1,value_1|temperature_2,value_2|...` with `temperature_1 < temperature_2`. For `polynomial`, the format is as follows: `coeff_0,coeff_1,coeff_2` where `coeff_0` is the coefficient of `T^0`, `coeff_1` is the coefficient of `T^1`, etc (required)
* **initial\_temperature**: initial temperature of all the materials in kelvins (default value: 300)
* **new\_material\_temperature**: temperature of all the material that is being added during the process in kelvins (default value: 300)
* **material\_X**: property tree for the material with number X
* **material\_X.Y**: property tree where Y is either liquid, powder, or solid
(one is required)
* **material\_X.Y.Z**: Z is either `density` in kg/m^3, `specific_heat` in J/(K\*kg),
`thermal_conductivity_x`, resp. `y` or `z`, in the direction `x`, resp. `y` or `z` (in 2D only `x` and `z` are used), in `W/(m\*K)`, `emissivity`,
or `convection_heat_transfer_coef` in `W/(m^2\*K)` (optional)
* **material\_X.A**: A is either `solidus` in kelvins, `liquidus` in kelvins, `latent_heat`
in `J/kg`, `radiation_temperature_infty` in kelvins, or `convection_temperature_infty`
  in kelvins (optional)

## memory\_space (optional): 
* `device` (use GPU if Kokkos was compiled with GPU support) or `host` (use CPU) (default value: host)

## post\_processor (required):
* **filename\_prefix**: prefix of output files (required)
* **time\_steps\_between\_output**: number of time steps between the fields being written to the output files (default value: 1)
* **additional\_output\_refinement**: additional levels of refinement for the output (default: 0)
* **output\_on\_data\_assimilation**: output fields just before and just after data assimilation (default: true, <span style="color:green">since 1.1<span>)

## refinement (required):
* **n\_refinements**: number of times the cells on the paths of the beams
are refined (default value: 2)
* **beam\_cutoff**: the cutoff value of the heat source terms above which beam-based refinement occurs (default value: 1e-15, <span style="color:red">removed in 1.1)
* **coarsen\_after\_beam**: whether to coarsen cells where the beam has already passed (default value: false)
* **time\_steps\_between\_refinement**: number of time steps after which the
  refinement process is performed (default value: 2)

## sources (required):
* **n\_beams**: number of heat source beams (required)
* **beam\_X**: property tree for the beam with number X
* **beam\_X.type**: type of heat source: `goldak`, `electron_beam`, or `cube` (required)
* **beam\_X.scan\_path\_file**: scan path filename (required)
* **beam\_X.scan\_path\_file\_format**: format of the scan path: `segment` or
`event_series` (required)
* **beam\_X.max\_power**: maximum power of the beam in watts (required)
* **beam\_X.depth**: maximum depth reached by the electron beam in meters (required)
* **beam\_X.absorption\_efficiency**: absorption efficiency of the beam equivalent
to `energy_conversion_efficiency * control_efficiency` for electon beam. Number
between 0 and 1 (required).
* **beam\_X.diameter**: diameter of the beam in meters (default value: 2e-3)

## time\_stepping (required):
* **method**: name of the method to use for the time integration: `forward_euler`,
`rk_third_order`, `rk_fourth_order`, `backward_euler`, `implicit_midpoint`, `crank_nicolson`, or
`sdirk2` (required)
* **scan\_path\_for\_duration**: if the flag is true, the duration of the simulation is determined by the duration of the scan path. In this case the scan path file needs to contain SCAN\_PATH\_END to terminate the simulation. If the flag is false, the duration of the simulation is determined by the duration input (default value: false, <span style="color:green">since 1.1<span>)
* **duration**: duration of the simulation in seconds (<span style="color:red">required for 1.0,</span> <span style="color:green">since 1.1 only required if scan\_path\_for\_duration is false</span>)
* **time\_step**: length of the time steps used for the simulation in seconds (required)
* for implicit method:
  * **max\_iteration**: mamximum number of the iterations of the linear solver
  (default value: 1000)
  * **tolerance**: tolerance of the linear solver (default value: 1e-12)
  * **n\_tmp\_vectors**: maximum number of vectors used by GMRES (default value:
  30)
  * **right\_preconditioner**: use left or right preconditioning for the linear
  solver (default value: false)
  * **newton\_max\_iteration**: maximum number of iterations of Newton solver
  (default value: 100)
  * **newton\_tolerance**: tolerance of the Newton solver (default value: 1e-6)
  * **jfnk**: use Jacobian-Free Newton Krylov method (default value: false)

## experiment (optional):
* **read\_in\_experimental\_data**: whether to read in experimental data (default: false)
* **file**: format of the file names. The format is pretty arbitrary, the keywords \#frame
and \#camera are replaced by the frame and the camera number. The format of
the file itself should be csv. (required)
* **format**: format of the experimental data, either `point_cloud`, with
    `(x, y, z, value)` per line, or `ray`, with `(pt0_x, pt0_y, pt0_z,
     pt1_x, pt1_y, pt1_z, value )` per line, where the ray starts at `pt0`
    and passes through `pt1` (required)
* **first\_frame**: number associated to the first frame (default value: 0)
* **last\_frame**: number associated to the last frame (required)
* **first\_camera\_id**: number associated to the first camera (required)
* **last\_camera\_id**: number associated to the last camera (required)
* **log\_filename**: the (full) filename of the log file that lists the timestamps for each frame from each camera. Note that the timestamps are not assumed to match the simulation time frame. The `first_frame_temporal_offset` parameter controls the simulation time corresponding to the first camera frame (required)
* **first\_frame\_temporal\_offset**: a uniform shift to the timestamps from all cameras to match the simulation time (default value: 0.0)
* **estimated\_uncertainty**: the estimate of the uncertainty in the experimental data points as given by a standard deviation 
  (under the simplifying assumption that the error is normally distributed and independent for each data point) (default value: 0.0).
* **output\_experiment\_on\_mesh**: whether to output the experimental data
    projected onto the simulation mesh at each experiment time stamp (default: true)

## ensemble (optional):
* **ensemble\_simulation**: whether to perform an ensemble of simulations (default value: false)
* **ensemble\_size**: number of ensemble members for the ensemble Kalman filter (EnKF) (default value: 5)
* **initial\_temperature\_stddev**: standard deviation for the initial temperature of the material (default value: 0.0, <span style="color:red">removed in 1.1)
* **new\_material\_temperature\_stddev**: standard deviation for the temperature of material added during the process (default value: 0.0, <span style="color:red">removed in 1.1)
* **beam\_0\_max\_power\_stddev**: standard deviation for the max power for beam 0 (if it exists) (default value: 0.0, <span style="color:red">removed in 1.1)
* **beam\_0\_absorption\_efficiency\_stddev**: standard deviation for the absorption efficiency for beam 0 (if it exists) (default value: 0.0, <span style="color:red">removed in 1.1)
* **variable_stddev**: standard deviation associated to `variable`. `variable` is an other entry in the input file, for instance `sources.beam_0.max_power`. The input file accepts multiple `variable_stddev` at once. Note that this only works for scalar value and therefore it does not work for temperature dependent variables (<span style="color:green">since 1.1</span>).

## data\_assimilation (optional):
* **assimilate\_data**: whether to perform data assimilation (default value: false)
* **localization\_cutoff\_function**: function used to decrease the sample covariance as the relevant points become farther away: gaspari\_cohn, step\_function, none (default: none)
* **localization\_cutoff\_distance**: distance at which sample covariance entries are set to zero (default: infinity)
* **augment\_with\_beam\_0\_absorption**: whether to augment the state vector with the beam 0 absorption efficiency (default: false)
* **augment\_with\_beam\_0\_max_power**: whether to augment the state vector with the beam 0 max power (default: false)
* **solver**:
  * **max\_number\_of\_temp\_vectors**: maximum number of temporary vectors for the GMRES solve (optional)
  * **max\_iterations**: maximum number of iterations for the GMRES solve (optional)
  * **convergence\_tolerance**: convergence tolerance for the GMRES solve (optional)

## profiling (optional):
* **timer**: output timing information (default value: false)
* **caliper**: configuration string for Caliper (optional)

## checkpoint (optional):
* **time\_steps\_between\_checkpoint**: number of time steps after which checkpointing is performed (required)
* **filename\_prefix**: prefix of the checkpoint files (required)
* **overwrite\_files**: if true the checkpoint files are overwritten by newer ones. If false, the time steps is added to the filename prefix (required)

## restart (optional):
* **filename\_prefix**: prefix of the restart files (required)

## units (optional): 
Change the unit of some inputs (<span style="color:green">since 1.1</span>)
* **mesh**: unit used for the mesh. Either millimeter, centimeter, inch, or meter (default value: meter)
* **heat\_source** (optional):
  * **power**: unit used for the power of the heat sources. Either milliwatt or
    watt (default value: watt)
  * **velocity**: unit used for the velocity of the heat sources. Either
    millimeter/second, centimeter/second, or meter/second (default value: meter/second)
  * **dimension**: unit used for the dimension of the heat sources. Either
    millimeter, centimeter, inch, or meter (default value: meter)
  * **scan\_path**: unit used for the scan path of the heat sources. Either
    millimeter, centimeter, inch, or meter (default value: meter)

## microstructure (optional): 
* **filename\_prefix**: prefix of the output file of the temperature gradient, the cooling rate, and the interface velocity at the liquidus. The format of the file is x y (z) temperature gradient (K/m) cooling rate (K/s) the inteface velocity (m/s)  (required, <span style="color:green">since 1.1</span>)

## verbose\_output (optional): 
* true or false (default value: false)
