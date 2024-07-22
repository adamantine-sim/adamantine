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
