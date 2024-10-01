---
layout: default
title: Home
nav_order: 1
has_children: true
usemathjax: true
---

# adamantine
*adamantine* is a thermomechanical code for additive manufacturing. It is based on
[deal.II](https://www.dealii.org), [ArborX](https://github.com/arborx/ArborX), 
[Trilinos](https://trilinos.github.io), and [Kokkos](https://kokkos.org).
*adamantine* can simulate the thermomechanical evolution an object undergoes during the
manufacturing process.  It can handle materials in three distinct phases (solid, liquid, 
and powder) to accurately reflect the physical state during manufacturing.
Experimental data can be used to improve the simulation through the use of 
[Ensemble Kalman filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
