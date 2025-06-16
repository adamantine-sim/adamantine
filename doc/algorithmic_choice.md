---
layout: default
parent: Home
title: Algorithmic Choices
nav_order: 2
usemathjax: true
---

# Algorithmic Choices
## Matrix-free implementation
The implementation is done matrix-free for the following reasons:
* New architecture have little memory per core and so not having to store
    the memory is very interesting.
* Because the latency of the memory, a very important part of our problem
    is memory bound. It is therefore interesting to decrease memory access even
    at the cost of more computation.
* Because we have time-dependent nonlinear problem, we would need to
    rebuild the matrix at least every time step. Since the assembly needs to be
    redone so often, storing the matrix is not advantageous.

## Adaptive mesh refinement
Usually, the powder layer is about 50 microns thick but the piece that is being
built is tens of centimeters long. Moreover, since the material is melted using
an electron beam or a laser, the melting zone is very localized. This means that
a uniform mesh would require a very large number of cells in places where nothing
happens (material not heated yet or already cooled). Using AMR, we can refine
the zones that are of interest for a given point in time.

## Element activation
To simulate the addition on material, we use the hp-capability of 
[deal.II](https://www.dealii.org). Deal.II supports a special kind of finite 
element called `FE_Nothing`. `FE_Nothing` is a finite element that does with
zero degree of freedom. This can be used to represent empty cells in a mesh on
which no degrees of freedom should be allocated. To simulate the addition of
material, we replace the `FE_Nothing` associated with a cell with a regular finite 
element, i.e., we activate an element. Using this technique, the addition of 
material can be done cell-wise. By coupling element activation and AMR, we can
add arbitrary small amount of material.

## MPI support
While mechanical and thermomechanical simulations are limited to serial
execution, thermal and EnKF (see [Data Assimilation section]({{site.baseurl}}/doc/data_assimilation)) 
ensemble simulations can use MPI. Thermal simulations can be performed using an 
arbitrary number of processors. MPI support for mechanical and thermomechanical 
simulations are a subject of ongoing work.

### Thermal simulation
The mesh is partitioned using [p4est](https://www.p4est.org/). The partitioning
takes into account that there is nothing to do on inactive cells. It tries to
distribute evenly the active cells between all the processors. If at the beginning
of the simulation very few cells are active, each processor will own a very
small number of active cells. In this case, the communication cost can be important
compared to the computation cost. Once more cells are activated this ceases to
be a problem. For a more in-depth discussion take a look at the [data
assimilation example]({{site.baseurl}}/doc/examples/data_assimilation).

### EnKF
For EnKF ensemble simulations, the partitioning scheme works as follows:
* If the number of processors (Nproc) is less than or equal to the number of EnKF 
    ensemble members (N), adamantine distributes the simulations evenly across the
    processors. All processors except the first will handle the same number of 
    simulations. The first processor might take on a larger workload if a perfect 
    split is not possible.
* *adamantine* can leverage more processors than there are simulations, but
    only if Nproc is a multiple of N. This ensures that all the simulations
    are partitioned in the same way.

### Checkpoint/restart
Thermal simulations and EnKF periodically write to files the current states of
the simulation and restart from these files later on. The number of processors
used by the restarted simulation does not have to be the same as the number of
processors used initially to checkpoint the simulation.

## GPU Support
There is partial support for GPU-accelerated calculations through the use of 
the Kokkos library. Part of the thermal simulation can be performed on the GPU. 
Performing the entire computation on the GPU is the subject of ongoing work.
