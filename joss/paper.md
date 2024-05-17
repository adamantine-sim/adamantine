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
`Adamantine` is a thermomechanical C++ code build on top of deal.II [@dealII95], 
ArborX [@arborx2020], Trilinos [@trilinos-website], and Kokkos [@kokkos2022].
`Adamantine` was developed with additive manufacturing in mind and it is
particularly well adapted to simulate fused filament fabrication, direct energy 
deposition, and powder bed fusion. In order to support these different processes, 
`adamantine` solve the anisotropic heat equation. It can handle materials in
three distinct phases (solid, liquid, and powder) to accurately reflect the
physical state during different stages of the manufacturing process. To enhance
simulation accuracy, `adamantine` incorporates data assimilation techniques [@da2016].
This allows it to integrate experimental data from sensors like thermocouples
and Infra-Red (IR) cameras. This combined approach helps account for errors
arising from input parameters, material properties, models, and numerical
calculations, leading to more realistic simulations.

# Statement of Need
Manufacturing "born-qualified" objects, i.e., parts ready for critical
applications straight from the printer, requires a new approach to additive
manufacturing (AM). This vision demands not only precise simulations for 
planning the build but also real-time adjustments throughout the process 
to obtained the desired thermomechanical evolution of the part. Currently, 
setting AM process parameters is an expert-driven, oftentrial-and-error 
process. Material changes and geometry complexities can lead to unpredictable 
adjustments in parameters, making it slow and expensive. We can overcome 
this by using advanced simulations for both planning and adaptive control.

`Adamantine`, a thermomechanical simulation tool, offers a solution. During the
planning phase, its capabilities can be leveraged to predict the
thermomechanical state and optimize process parameters for the desired outcome. 
For adaptive control, `adamantine` utilizes data from infrared (IR) cameras and 
thermocouples. This data is integrated using the Ensemble Kalman Filter (EnKF) method,
allowing the simulation to constantly adapt and reflect the actual build process.

With a continuously refined simulation, `adamantine` can predict the final state 
of the object with greater accuracy. This enables adjustments to the build parameters 
mid-print, if needed, ensuring the creation of born-qualified parts.

While other open-source software like AdditiveFOAM [@additivefoam] excels at heat
and mass transfer simulations in additive manufacturing, and commercial options
like Abaqus [@abaqus] and Ansys offer comprehensive thermomechanical capabilities,
`adamantine` stands out for its unique ability to incorporate real-world data
through data assimilation. This feature allows for potentially more accurate
simulations, leading to better process optimization and final part quality.

# Simulated Physics

## Thermal Simulation
We solve the heat equation with change of phases: powder $\Rightarrow$ liquid, 
solid $\Rightarrow$ liquid, and liquid $\Rightarrow$ solid. The simulation 
requires the presence of a "mushy" zone, i.e., the liquidus and the solidus are 
different. This is generally the case for alloys. The following boundary conditions 
are implemented: adiabatic, convective, and radiative.

## Mechanical Simulation
We solve an elasto-plastic problem. The plastic model is the linear combination of
the isotropic and kinematic hardening describe in [@borja2013]. This allows us to 
model both the change in yield stress and the Bauschinger effect.

## Thermomechanical Simulation
We solve the thermomechanical problem where the thermal and the mechanical
simulations are coupled. In this case, we have an extra term in the mechanial
simulation which takes into account the thermal expansion of the material [@fung2001]. 
We assume a one-way coupling and neglect the effect of the deformation on the thermal
simulation.

# Data Assimilation
Data assimilation "is the approximation of a true state of some physical system
at a given time by combining time-distributed observations with a dynamic model
in an optimal way" [@da2016]. `Adamantine` leverages this technique to enhance
the precision of its simulations.

We have implemented a data assimilation algorithm called the Ensemble Kalman
Filter (EnKF). This statistical technique seamlessly combines the simulation's
predictions with experimental data. EnKF requires to perform an ensemble of 
simulations with slightly different input simulations. These EnKF ensemble 
simulations are done inside a unique instance of `adamantine`.  This allows 
experimental data from IR cameras and thermocouples ton be assimilated in order
to provide insight into the actual AM process.

# Algorithmic Choices

## Matrix-free
We have made the choice of using explicit time-stepping schemes and thus,
it is necessary to perform each time step efficiently. Since a time step 
correspond to the evaluation of an operator, we use a matrix-free technique 
[@kronbichler2012]. This technique avoids explicitly storing the full system matrix, 
which becomes computationally expensive for higher-degree finite elements. Instead, 
`adamantine` calculates the operator's action directly, significantly reducing 
computational cost compared to traditional matrix-vector multiplication.

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

## GPU support
There is partial support for GPU through the use of the Kokkos library. Part of 
the thermal simulation can be performed on the GPU but the mechanical simulation is CPU only.
Performing the entire computation on the GPU is part of our future plan.

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
