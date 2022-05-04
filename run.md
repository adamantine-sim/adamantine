---
layout: page
title: Run
nav_order: 3
---

# Run
After compiling `adamantine`, you can run a simulation using
```bash
mpirun -n 2 ./adamantine --input-file=input.info
```
Note that the name of the input file is totally arbitrary, `my_input_file` is as
valid as `input.info`.

There is a [known bug](https://github.com/adamantine-sim/adamantine/issues/130)
when using multithreading. To deactivate multithreading use
```bash
export DEAL_II_NUM_THREADS=1
```
If you use our Docker image, the variable is already set.

## Input file
The following options are available:
* **boundary**:
  * **type**: type of boundary: `adiabatic`, `radiative`, or `convective`. Multiple types
  can be chosen simultaneously by separating them by comma (required)
* **discretization**:
  * **fe\_degree**: degree of the finite element used (required)
  * **quadrature**: quadrature used: `gauss` or `lobatto` (default value: `gauss`)
* **geometry**:
  * **dim**: the dimension of the problem (2 or 3, required)
  * **material\_height**: below this height the domain contains material. Above this
  height the domain is empty (default value: 1e9)
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
        * **deposition\_length**: length of material deposition boxes along the scan direction
        * **deposition\_width**: width of material deposition boxes (in the plane of the material, normal to the scan direction, 3D only)
        * **deposition\_height**: height of material deposition boxes (out of the plane of the material)
        * **deposition\_lead\_time**: amount of time before the scan path reaches a point that the material is added
  * **import\_mesh**: true or false (required)
  * if import\_mesh is true:
    * **mesh\_file**: The filename for the mesh file (required)
    * **mesh\_format**: `abaqus`, `assimp`, `unv`, `ucd`, `dbmesh`, `gmsh`, `tecplot`, `xda`, `vtk`,
    `vtu`, `exodus`, or `default`, i.e., use the file suffix to try to determine the
    mesh format (required)
  * if import\_mesh is false:
    * **length**: the length of the domain in meters (required)
    * **height**: the height of the domain in meters (required)
    * **width**: the width of the domain in meters (only in 3D)
    * **length\_divisions**: number of cell layers in length (default value: 10)
    * **height\_divisions**: number of cell layers in the height (default value: 10)
    * **width\_divisions**: number of cell layers in width (only in 3D) (default value: 10)
* **materials**:
  * **n\_materials**: number of materials (required)
  * **property\_format**: format of the material property: table or polynomial (required)
  * **initial\_temperature**: initial temperature of all the materials (default value: 300)
  * **new\_material\_temperature**: temperature of all the material that is being added during the process (default value: 300)
  * **material\_X**: property tree for the material with number X
  * **material\_X.Y**: property tree where Y is either liquid, powder, or solid
  (one is required)
  * **material\_X.Y.Z**: Z is either `density` in kg/m^3, `specific_heat` in J/(K\*kg),
  `thermal_conductivity_x`, resp. `y` or `z`, in the direction `x`, resp. `y` or `z` (in 2D only `x` and `z` are used), in `W/(m\*K)`, `emissivity`,
  or `convection_heat_transfer_coef` in `W/(m^2\*K)` (optional)
  * **material\_X.A**: A is either `solidus` in kelvin, `liquidus` in kelvin, `latent_heat`
  in `J/kg`, `radiation_temperature_infty` in kelvin, or `convection_temperature_infty`
  in kelvin (optional)
* **memory\_space**: `device` (use GPU) or `host` (use CPU) (default value: host)
* **post\_processor** (required):
  * **filename\_prefix**: prefix of output files (required)
  * **time\_steps\_between\_output**: number of time steps between the
  fields being written to the output files (default value: 1)
* **refinement** (required):
  * **n\_heat\_refinements**: number of coarsening/refinement to execute (default value: 2)
  * **heat\_cell\_ratio**: this is the ratio (n new cells)/(n old cells) after heat
  refinement (default value: 1)
  * **n\_beam\_refinements**: number of times the cells on the paths of the beams
  are refined (default value: 2)
  * **beam\_cutoff**: the cutoff value of the heat source terms above which beam-based refinement occurs (default value: 1e-15)
  * **coarsen\_after\_beam**: whether to coarsen cells where the beam has already passed (may conflict with heat refinement, default value: false)
  * **max\_level**: maximum number of times a cell can be refined
  * **time\_steps\_between\_refinement**: number of time steps after which the
  refinement process is performed (default value: 2)
  * **verbose**: true or false (default value: false)
* **sources** (required):
  * **n\_beams**: number of heat source beams (required)
  * **beam\_X**: property tree for the beam with number X
  * **beam\_X.type**: type of heat source: `goldak`, `electron_beam`, or `cube` (required)
  * **beam\_X.scan\_path\_file**: scan path filename (required)
  * **beam\_X.scan\_path\_file\_format**: format of the scan path: `segment` or
  `event_series` (required)
  * **beam\_X.depth**: maximum depth reached by the electron beam in meters (required)
  * **beam\_X.absorption\_efficiency**: absorption efficiency of the beam equivalent
  to `energy_conversion_efficiency * control_efficiency` for electon beam. Number
  between 0 and 1 (required).
  * **beam\_X.diameter**: diameter of the beam in meters (default value: 2e-3)
* **time\_stepping** (required):
  * **method**: name of the method to use for the time integration: `forward_euler`,
  `rk_third_order`, `rk_fourth_order`, `heun_euler`, `bogacki_shampine`, `dopri`,
  `fehlberg`, `cash_karp`, `backward_euler`, `implicit_midpoint`, `crank_nicolson`, or
  `sdirk2` (required)
  * **duration**: duration of the simulation in seconds (required)
  * **time\_step**: length of the time steps used for the simulation in seconds (required)
  * for embedded methods:
    * **coarsening\_parameter**: coarsening of the time step when the error is small
    enough (default value: 1.2)
    * **refining\_parameter**: refining of the time step when the error is too large
    (default value: 0.8)
    * **min\_time\_step**: minimal time step (default value: 1e-14)
    * **max\_time\_step**: maximal time step (default value: 1e100)
    * **refining\_tolerance**: if the error is above the threshold, the time step is
    refined (default value: 1e-8)
    * **coarsening\_tolerance**: if the error is under the threshold, the time step
    is coarsen (default value: 1e-12)
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
* **experiment**: (optional)
  * *read\_in\_experimental\_data*: whether to read in experimental data (default: false)
  * **file**: format of the file names. The format is pretty arbitrary, the keywords \#frame
  and \#camera are replaced by the frame and the camera number. The format of
  the file itself should be csv. (required)
  * **first\_frame**: number associated to the first frame (default value: 0)
  * **last\_frame**: number associated to the last frame (required)
  * **first\_camera\_id**: number associated to the first camera (required)
  * **last\_camera\_id**: number associated to the last camera (required)
  * **data\_columns**: columns associated with x, y, T (in 2D) and x, y, z, T (in 3D) (required)
  * **log\_filename**: the (full) filename of the log file that lists the timestamp for each frame from each camera. (required)
  * **first\_frame\_temporal\_offset**: a uniform shift to the timestamps from all cameras to match the simulation time (default value: 0.0)
  * **estimated\_uncertainty**: the estimate of the uncertainty in the experimental data points as given by a standard deviation 
    (under the simplifying assumption that the error is normally distributed and independent for each data point) (default value: 0.0).
* **ensemble**: (optional)
  * **ensemble\_simulation**: whether to perform an ensemble of simulations (default value: false)
  * **ensemble\_size**: number of ensemble members for the ensemble Kalman filter (EnKF) (default value: 5)
  * **initial\_temperature\_stddev**: standard deviation for the initial temperature of the material (default value: 0.0)
  * **new\_material\_temperature\_stddev**: standard deviation for the temperature of material added during the process (default value: 0.0)
  * **beam\_0\_max\_power\_stddev**: standard deviation for the max power for beam 0 (if it exists) (default value: 0.0)
  * **beam\_0\_absorption\_efficiency\_stddev**: standard deviation for the absorption efficiency for beam 0 (if it exists) (default value: 0.0)
* **data\_assimilation**: (optional)
  * **assimilate\_data**: whether to perform data assimilation (default value: false)
  * **localization\_cutoff\_function**: function used to decrease the sample covariance as the relevant points become farther away: gaspari\_cohn, step\_function, none (default: none)
  * **localization\_cutoff\_distance**: distance at which sample covariance entries are set to zero (default: infinity)
  * **augment\_with\_beam\_0\_absorption**: whether to augment the state vector with the beam 0 absorption efficiency (default: false)
  * **augment\_with\_beam\_0\_max_power**: whether to augment the state vector with the beam 0 max power (default: false)
  * **solver**:
    * **max\_number\_of\_temp\_vectors**: Maximum number of temporary vectors for the GMRES solve (optional)
    * **max\_iterations**: Maximum number of iterations for the GMRES solve (optional)
    * **convergence\_tolerance**: Convergence tolerance for the GMRES solve (optional)
* **profiling** (optional):
  * **timer**: output timing information (default value: false)
  * **caliper**: configuration string for Caliper (optional)


## Scan path
`adamantine` supports two kinds of scan path input: the `segment` format and the
`event` format.
### Segment format
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
### Event format
For an event series the first segment is a point, then the rest are lines.
The column descriptions are:
* Column 1: segment endtime
* Columns 2 to 4: (x,y,z) coordinates in units of m. This is the ending
position of the line.
* Column 5: the coefficient for the nominal power. Usually this is either
0 or 1, but sometimes intermediate values are used when turning a corner.

## Material deposition
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
