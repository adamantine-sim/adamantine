---
layout: default
parent: Examples
title: Thermal Simulation
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

# Thermal Simulation
This example shows thermal simulation of material deposition on a plate.

The example is composed of the following files:
 * [demo_316_short.info](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/demo_316_short.info): the input file
 * [demo_316_short_scan_path.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/demo_316_short_scan_path.txt): the scan path of the heat source

It can be run using `./adamantine -i demo_316_short.info`.

## Understanding the input file

An *adamantine* INFO file is organized into named blocks. A block starts with a
name followed by braces, and each line inside the braces sets one input. For
example, `geometry { ... }` is the `geometry` block, while
`geometry { length }` would refer to the `length` input inside that block. Text
after a semicolon is a comment and is ignored by the program. Values written
with `e` notation are scientific notation: for example, `10e-3` means
`0.010`, or 10 millimeters when the unit is meters.

The complete list of inputs and their accepted values is available in the
[input-file reference]({{site.baseurl}}/doc/input_file.html). This explanation
focuses on what each value does in this particular example.

The sections below follow the order in the input file. Together they answer
four basic questions: what is being simulated, what material and heat source
are used, how the equations are advanced in time, and what files are written.

### 1. Geometry

```text
geometry
{
  import_mesh false
  dim 3
  length 10e-3
  height 2.0e-3
  width 10e-3
  length_divisions 50
  height_divisions 8
  width_divisions 50
  material_deposition true
  material_deposition_method scan_paths
  deposition_length 0.0003
  deposition_height 0.0002
  deposition_width 0.0003
  deposition_lead_time 0.0005
  material_height 0.75e-3
}
```

The `geometry` block defines the computational domain and how material is
initially placed in it.

* `import_mesh false` tells *adamantine* to generate a regular mesh from the
  dimensions below. If this were `true`, the block would instead need a mesh
  filename and mesh format.
* `dim 3` selects a three-dimensional simulation. Therefore all three domain
  dimensions and all three division counts are used.
* `length 10e-3`, `height 2.0e-3`, and `width 10e-3` make a box that is 10 mm
  long, 2 mm high, and 10 mm wide. The input-file units for these values are
  meters.
* `length_divisions 50`, `height_divisions 8`, and `width_divisions 50` split
  the box into 50, 8, and 50 cell layers in the corresponding directions.
  More divisions generally give a finer spatial resolution, but also increase
  the computational cost. These are cell counts, not physical lengths.
* `material_deposition true` enables adding material while the simulation is
  running. This is what makes the example a deposition process rather than a
  heat source moving over a permanently filled block.
* `material_deposition_method scan_paths` says that deposition is controlled by
  the beam scan path. The alternative method reads deposition information from
  a separate material-deposition file.
* `deposition_length 0.0003`, `deposition_width 0.0003`, and
  `deposition_height 0.0002` define the size of each material-deposition box:
  0.3 mm along the scan direction, 0.3 mm across the scan direction, and
  0.2 mm vertically. `deposition_width` is used because this is a 3D problem.
* `deposition_lead_time 0.0005` means that material is added 0.5 ms before the
  heat source reaches the corresponding scan location. The lead time gives
  the newly deposited material a spatial and temporal relationship to the
  moving beam.
* `material_height 0.75e-3` says that the initial material occupies the part
  of the domain below 0.75 mm. The region above that height starts empty and
  can be filled by the deposition process.

There is no `use_powder` input in this block, so it keeps its default value of
`false`. The example therefore does not turn on the separate initial-powder
layer option described in the input-file reference.

### 2. Boundary

```text
boundary
{
  type convective,radiative
}
```

The `boundary` block describes how heat crosses the external boundary of the
domain. The comma-separated value selects two boundary mechanisms at once:

* `convective` models heat exchanged with the surrounding environment through
  convection. The convection coefficient and ambient temperature come from
  the material block below.
* `radiative` models thermal radiation from the surface to the environment.
  Its strength depends on the emissivity and the surrounding radiation
  temperature.

Because no `boundary_X` or `printed_surface` sub-block is supplied, the same
general boundary selection is used rather than assigning different boundary
types to particular face IDs or to the printed surface.

### 3. Physics

```text
physics
{
  thermal true
  mechanical false
}
```

This block selects the equations that *adamantine* solves.

* `thermal true` enables the transient heat equation, so the temperature field
  changes as the beam moves and as heat diffuses through the material.
* `mechanical false` disables the mechanical calculation. The example does not
  compute displacement, stress, or strain.

Since only the thermal flag is enabled, this is a purely thermal simulation.
If both flags were `true`, *adamantine* would solve a coupled thermo-mechanical
problem.

### 4. Refinement

```text
refinement
{
  n_refinements 0
  time_steps_between_refinement 2000000000
}
```

The `refinement` block controls adaptive mesh refinement near the heat-source
paths.

* `n_refinements 0` requests no additional refinement levels. The generated
  mesh is therefore used at its original resolution.
* `time_steps_between_refinement 2000000000` is the interval at which the
  refinement procedure would be considered. It is intentionally enormous for
  this example, so refinement is effectively disabled even aside from the
  zero refinement levels.

### 5. Materials

```text
materials
{
  n_materials 1
  property_format polynomial
  material_0
  {
    ...
  }
}
```

The `materials` block supplies the physical properties used by the heat
equation.

* `n_materials 1` declares one material definition, named `material_0`. The
  number in the name is zero-based: the first material is material 0.
* `property_format polynomial` says how temperature-dependent properties are
  represented. A property is interpreted as polynomial coefficients, starting
  with the coefficient of `T^0`. Because the values in this example are single
  numbers, they represent constant properties rather than properties that
  vary with temperature.

The `material_0` block contains three possible material states:

```text
material_0
{
  solid { ... }
  powder { ... }
  liquid { ... }
  solidus 1675
  liquidus 1708
  latent_heat 290000
  radiation_temperature_infty 300
  convection_temperature_infty 300
}
```

The state blocks allow *adamantine* to use different properties before melting,
while material is powder, and after melting. The phase-change temperatures are
in kelvins:

* `solidus 1675` is the temperature below which the material is treated as
  solid.
* `liquidus 1708` is the temperature above which it is treated as liquid.
* Between the solidus and liquidus temperatures, the material is in the
  transition interval. `latent_heat 290000` supplies the energy per kilogram
  associated with melting.

The environmental temperatures are both 300 K:

* `radiation_temperature_infty 300` is the surrounding temperature used by the
  radiative boundary condition.
* `convection_temperature_infty 300` is the surrounding temperature used by
  the convective boundary condition.

The global `initial_temperature` and `new_material_temperature` inputs are
omitted, so both keep their documented default of 300 K. This means that the
initial material and material added during deposition start at the same
ambient temperature unless another input changes them.

#### Solid properties

```text
solid
{
  density 7904
  specific_heat 714
  thermal_conductivity_x 31.4
  thermal_conductivity_y 31.4
  thermal_conductivity_z 31.4
  convection_heat_transfer_coef 100
  emissivity 0.15
}
```

The solid has density `7904 kg/m^3`, specific heat `714 J/(K*kg)`, and
thermal conductivity `31.4 W/(m*K)` in each direction. Equal conductivity in
all directions makes this material thermally isotropic. The convection heat
transfer coefficient is `100 W/(m^2*K)`, and the surface emissivity is `0.15`.
Those last two values are used by the boundary conditions selected above.

#### Powder properties

```text
powder
{
  specific_heat 714
  density 7904
  thermal_conductivity_x 0.314
  thermal_conductivity_y 0.314
  thermal_conductivity_z 0.314
  convection_heat_transfer_coef 100
  emissivity 0.15
}
```

The powder uses the same density and specific heat as the solid, but its
thermal conductivity is `0.314 W/(m*K)` in every direction, one hundredth of
the solid value. This represents the poorer heat conduction of a loose powder
region. The powder state is defined for the material model, although this
input does not enable the separate `use_powder` initial-layer option.

#### Liquid properties

```text
liquid
{
  specific_heat 847
  density 7904
  thermal_conductivity_x 37.3
  thermal_conductivity_y 37.3
  thermal_conductivity_z 37.3
  convection_heat_transfer_coef 100
  emissivity 0.15
}
```

The liquid has density `7904 kg/m^3`, specific heat `847 J/(K*kg)`, and
thermal conductivity `37.3 W/(m*K)` in each direction. As with the other
states, convection and radiation use a coefficient of `100 W/(m^2*K)` and an
emissivity of `0.15`.

Mechanical material properties such as Lamé parameters, thermal expansion,
plastic modulus, and elastic limit are not included because the physics block
turns mechanics off.

### 6. Heat source

```text
sources
{
  n_beams 1
  beam_0
  {
    type goldak
    depth 0.5e-3
    diameter 0.6e-3
    scan_path_file demo_316_short_scan_path.txt
    scan_path_file_format segment
    absorption_efficiency 0.3
    max_power 400.0
  }
}
```

The `sources` block describes the moving heat input.

* `n_beams 1` creates one heat source, named `beam_0`.
* `type goldak` selects the Goldak heat-source model, which is the laser-like
  source used in this example.
* `depth 0.5e-3` limits the source's maximum penetration depth to 0.5 mm.
* `diameter 0.6e-3` gives the source a diameter of 0.6 mm.
* `scan_path_file demo_316_short_scan_path.txt` points to the file that tells
  the source where to start and how to move. The file must be available from
  the directory in which *adamantine* is run.
* `scan_path_file_format segment` tells *adamantine* to interpret that file as a
  sequence of path segments. This is different from the event-series format.
* `absorption_efficiency 0.3` says that 30% of the beam power is absorbed by
  the object and contributes heat; the rest is not deposited as thermal input.
* `max_power 400.0` sets the beam's maximum power to 400 W.

### 7. Time stepping

```text
time_stepping
{
  method forward_euler
  duration 0.004
  time_step 0.6e-4
}
```

This block controls how the transient thermal equation is advanced.

* `method forward_euler` selects the first-order explicit Forward Euler time
  integration method.
* `duration 0.004` runs the simulation for 0.004 s, or 4 ms, unless the run is
  stopped earlier for another reason.
* `time_step 0.6e-4` uses a time step of `0.00006 s`, or 60 microseconds. The
  time step must be small enough for the chosen mesh, material properties, and
  explicit integration method to remain stable.

The scan path does not determine the duration here because
`scan_path_for_duration` is omitted and therefore keeps its default value of
`false`.

### 8. Output

```text
post_processor
{
  filename_prefix output
  time_steps_between_output 10
}
```

The `post_processor` block controls field output.

* `filename_prefix output` makes generated output files start with the prefix
  `output`.
* `time_steps_between_output 10` writes the temperature and other requested
  fields every ten simulation time steps instead of after every step. This
  reduces the number of output files and the amount of disk space used.

No additional output refinement is requested, so output uses the default
additional refinement level of zero.

### 9. Thermal discretization

```text
discretization
{
  thermal
  {
    fe_degree 3
    quadrature gauss
  }
}
```

The `discretization` block describes how the thermal field is represented on
the mesh.

* `fe_degree 3` uses third-degree finite-element basis functions. Higher degree
  can represent smoother spatial variation within each cell, but generally
  costs more to solve.
* `quadrature gauss` uses Gauss quadrature to evaluate the integrals needed by
  the finite-element method. `gauss` is also the default, so this line makes
  the choice explicit.

There is no `mechanical` discretization block because mechanical physics is
disabled.

### 10. Profiling

```text
profiling
{
  timer false
  caliper "spot(profile.mpi),loop-report,runtime-report"
}
```

The `profiling` block is about measuring program performance, not changing the
physics.

* `timer false` disables *adamantine*'s built-in timing output.
* `caliper ...` provides a Caliper configuration string. If Caliper profiling
  is enabled in the build and activated by the run environment, this string
  requests spot profiling, loop reporting, and runtime reporting.

### 11. Memory space

```text
memory_space host
```

`host` tells the Kokkos-backed parts of the program to use host memory, which
corresponds to the CPU in this example. The alternative `device` value is used
when *adamantine* has been built with GPU support and the calculation should run
in device memory.

### Putting the configuration together

In plain language, this file asks *adamantine* to simulate a 3D, 10 mm by 10 mm
by 2 mm domain whose initial material fills the bottom 0.75 mm. A single
Goldak heat source moves along the segments in the scan-path file. The model
tracks heat transfer, melting, latent heat, convection, and radiation, while
material is added ahead of the moving source. The calculation uses an
unrefined, third-degree finite-element mesh, advances for 4 ms with 60
microsecond Forward Euler steps, and writes fields every ten steps.

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
 * [demo_316_short_amr.info](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/demo_316_short_amr.info): the input file
 * [demo_316_short_scan_path_amr.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/demo_316_short_scan_path_amr.txt): the scan path of the heat source

It can be run using `./adamantine -i demo_316_short_amr.info`.

The main difference with the previous example concerns the refinement input:
```text
refinement
{
  n_refinements 1                 ; Number of time the cells on the paths of the beams are refined.
  time_steps_between_refinement 5 ; Number of time steps after which the refinement process is performed.
  coarsen_after_beam true         ; The cells are coarsen once the beam has passed
}
```
The other differences are:
 * the domain size is reduced and the number of cells is reduced accordingly
 * the time step is increased from 6e-5 *s* to 1e-4 *s*
 * the order of the finite element is decrease from 3 to 2

# Demo316 anisotropic
This is similar to *Demo316* but the thermal conductivity of the material is
anisotropic. The conductivity in the deposition direction is increased.

This example is composed of the following files:
 * [demo_316_short_anisotropic.info](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/demo_316_short_anisotropic.info): the input file
 * [demo_316_short_scan_path.txt](https://github.com/adamantine-sim/adamantine/blob/master/tests/data/demo_316_short_scan_path.txt): the scan path of the heat source

It can be run using `./adamantine -i demo_316_short_anisotropic.info`.

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
