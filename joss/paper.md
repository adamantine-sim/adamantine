---
title: 'Adamantine 1.0: A Thermomechanical Simulator for Additive Manufacturing'
tags:
  - C++
  - additive manufacturing
  - data assimilation
authors:
  - name: Bruno Turcksin
    orcid: 0000-0001-5954-6313
    corresponding: true
    affiliation: 1
  - name: Stephen DeWitt
    orcid: 0000-0002-9550-293X
    affiliation: 1
affiliations:
  - name: Oak Ridge National Laboratory, Oak Ridge, TN, USA
    index: 1
date: 1 July 2024
bibliography: paper.bib
---

# Summary
`Adamantine` is a thermomechanical simulation code that is written in C++ and built on 
top of deal.II [@dealII95], p4est [@p4est], ArborX [@arborx2020], Trilinos [@trilinos-website], 
and Kokkos [@kokkos2022]. `Adamantine` was developed with additive manufacturing in mind and it is
particularly well adapted to simulate fused filament fabrication, directed energy 
deposition, and powder bed fusion. `Adamantine` employs the finite element
method with adaptive mesh refinement to solve a nonlinear anisotropic heat equation, enabling
support for various additive manufacturing processes. It can also perform
elastoplastic and thermoelastoplastic simulations. It can handle materials in
three distinct phases (solid, liquid, and powder) to accurately reflect the
physical state during different stages of the manufacturing process. To enhance
simulation accuracy, `adamantine` incorporates data assimilation techniques [@da2016].
This allows it to integrate experimental data from sensors like thermocouples
and infrared (IR) cameras. This combined approach helps account for errors
arising from input parameters, material properties, models, and numerical
calculations, leading to more realistic simulations that reflect what occurs in a particular print.

# Statement of Need
Manufacturing "born-qualified" components, i.e., parts ready for critical
applications straight from the printer, requires a new approach to additive
manufacturing (AM). This vision demands not only precise simulations for 
planning the build but also real-time adjustments throughout the process 
to obtain the desired thermomechanical evolution of the part. Currently, 
setting AM process parameters is an expert-driven, often trial-and-error 
process. Material changes and geometry complexities can lead to unpredictable 
adjustments in parameters, making a purely empirical approach slow and expensive. We can overcome 
this by using advanced simulations for both planning and adaptive control.

`Adamantine`, a thermomechanical simulation tool, offers a solution to process
parameter planning and adjustment in AM. During the
planning phase, its capabilities can be leveraged to predict the
thermomechanical state and optimize process parameters for the desired outcome. 
For adaptive control, `adamantine` utilizes data from IR cameras and 
thermocouples. This data is integrated using the Ensemble Kalman Filter (EnKF) method,
allowing the simulation to constantly adapt and reflect the actual build process.

With a continuously refined simulation, `adamantine` can predict the final thermomechanical state 
of the object with greater accuracy. This simulation-enhanced monitoring capability enables a human operator or an adaptive control algorithm to adjust to the build parameters 
mid-print, if needed, to ensure that printed parts conform to the necessary tolerances. 

While other open-source software like AdditiveFOAM [@additivefoam] excels at heat
and mass transfer simulations in additive manufacturing, and commercial options
like Abaqus [@abaqus] and Ansys [@ansys] offer comprehensive thermomechanical capabilities,
`adamantine` stands out for its unique ability to incorporate real-world data
through data assimilation. This feature allows for potentially more accurate
simulations, leading to better process optimization and final part quality.

# Simulated Physics

## Thermal simulation
`Adamantine` solves an anisotropic version of standard continuum heat transfer model used in additive manufacturing simulations [@Megahed2016; @KELLER2017244]. The model includes the change of phases between powder, liquid, and solid and accounts for latent heat release for melting/solidification phase transformations. It assumes the presence of a "mushy" zone, i.e., the liquidus and the solidus are different, as is generally the case for alloys. The heat input by the laser, electron beam, electric-arc, or other process-specific heat source is introduced using a volumetric source term [@Goldak1984; @KNAPP2023111904]. Adiabatic, convective, and radiative boundary conditions are implemented, with the option to combine convective and radiative boundary conditions. 

## Mechanical simulation
`Adamantine` can perform elastoplastic simulations. The plastic model is the linear combination of
the isotropic and kinematic hardening described in @borja2013. This allows us to 
model both the change in yield stress and the Bauschinger effect.

## Thermomechanical simulation
Thermomechanical simulations in `adamantine` are performed with one-way coupling from the temperature evolution to the mechanical evolution. We neglect the effect of deformation on the thermal simulation. An extra term in the mechanical simulation accounts for the eigenstrain associated with by thermal expansion of the material [@fung2001; @Megahed2016].

# Data Assimilation
Data assimilation "is the approximation of a true state of some physical system
at a given time by combining time-distributed observations with a dynamic model
in an optimal way" [@da2016]. `Adamantine` leverages this technique to enhance
the accuracy of simulations during and after prints with in-situ characterization. It also ties the simulation results to the particular events (e.g. resulting for stochastic processes) for a specific print.

We have implemented a data assimilation algorithm called the Ensemble Kalman
Filter [@da2016]. This statistical technique incorporates experimental observations into a simulation to provide the best estimate (in the Bayesian sense) of the state of the system that reflects uncertainties from both data sources. EnKF requires to perform an ensemble of 
simulations with slightly different input model parameters and/or initial conditions. The EnKF calculation and the coordination of simulations of ensemble 
members are done from inside `adamantine`.  

# Algorithmic Choices

## Time integration
`Adamantine` includes several options for time integration methods that it inherits from the deal.II library [@dealII95]. These are: forward Euler, 3rd order explicit Runge-Kutta, 4th order explicit Runge-Kutta, backward Euler, implicit midpoint, Crank-Nicolson, and singly diagonally implicit Runge-Kutta. 

## Matrix-free finite element formulation
`Adamantine` uses a variable-order finite element spatial discretization with a matrix-free approach [@kronbichler2012]. This approach calculates the action of an operator directly, rather than explicitly storing the full (sparse) system matrix. This matrix-free approach significantly reduces computational cost, especially for higher-degree finite elements.

## MPI support
While mechanical and thermomechanical simulations are limited to serial
execution, thermal and EnKF ensemble simulations can use MPI. Thermal
simulations can be performed using an arbitrary number of processors. For EnKF
ensemble simulations, the partitioning scheme works as follows:

 * If the number of processors (Nproc) is less than or equal to the number of
 EnKF ensemble members (N), `adamantine` distributes the simulations evenly
 across the processors. All processors except the first will handle the same
 number of simulations. The first processor might take on a larger workload if a
 perfect split is not possible
 * `Adamantine` can leverage more processors than there are simulations, but
 only if Nproc is a multiple of N. This ensures that all the simulations are
 partitioned in the same way.

MPI support for mechanical and thermomechanical simulations are a subject of ongoing work.

## GPU support
`Adamantine` includes partial support for GPU-accelerated calculations through the use of the Kokkos library. 
The evaluation of the thermal operator can be performed on the GPU. The heat
source is computed on the CPU. The mechanical simulation is CPU only.
Performing the entire computation on the GPU is the subject of ongoing work.

# Mesh
`Adamantine` uses a purely hexahedral mesh. It has limited internal capabilities to
generate meshes. For complex geometries, `adamantine` can load meshes created by
mesh generators. The following formats are supported: `unv` format from the SALOME mesh
generator (SMESH) [@smesh], `UCD`, `VTK` [@vtk], Abaqus [@abaqus] file format, DB mesh,
`msh` file from Gmsh [@gmsh], `mphtxt` format from COMSOL [@comsol], Tecplot [@tecplot], 
assimp [@assimp], and ExodusII [@exodusii]. The generated mesh should be conformal. 
During the simulation, `adamantine` can adaptively refine the mesh near the heat source 
using the forest of octrees approach [@dealII95; @p4est], where each element in the initial 
mesh can be refined as an octree.

# Additional Information
An in-depth discussion of the governing equations and examples showcasing the
capabilities of`adamantine` can be found at https://adamantine-sim.github.io/adamantine

# Acknowledgments
This manuscript has been authored by UT-Battelle, LLC, under contract
DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government
retains and the publisher, by accepting the article for publication, acknowledges 
that the US government retains a nonexclusive, paid-up, irrevocable, worldwide 
license to publish or reproduce the published form of this manuscript, or allow 
others to do so, for US government purposes. DOE will provide public access to 
these results of federally sponsored research in accordance with the DOE Public 
Access Plan (https://www.energy.gov/doe-public-access-plan).

This research is sponsored by the INTERSECT Initiative and the SEED Program as
part of the Laboratory Directed Research and Development Program of Oak Ridge 
National Laboratory, managed by UT-Battelle, LLC, for the US Department of 
Energy under contract DE-AC05-00OR22725.

This research used resources of the Compute and Data Environment for Science
(CADES) at the Oak Ridge National Laboratory, which is supported by the Office 
of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

# References
