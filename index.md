---
layout: default
title: Home
nav_order: 1
has_children: true
usemathjax: true
---

# adamantine

## What is adamantine?

*adamantine* is a thermomechanical code for additive manufacturing. It is based on
[deal.II](https://www.dealii.org), [ArborX](https://github.com/arborx/ArborX), 
[Trilinos](https://trilinos.github.io), and [Kokkos](https://kokkos.org).
*adamantine* can simulate the thermomechanical evolution an object undergoes during the
manufacturing process.  It can handle materials in three distinct phases (solid, liquid, 
and powder) to accurately reflect the physical state during manufacturing.
Experimental data can be used to improve the simulation through the use of 
[Ensemble Kalman filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).

## New here? Run your first simulation

The fastest way to see *adamantine* in action is the
[Docker quickstart]({{site.baseurl}}/doc/quickstart). It uses a short thermal example and takes you from installation to a result
you can inspect in VisIt.

1. Pull the verified Docker image.
2. Download the example input and scan-path files.
3. Run the simulation.
4. Open the generated `.pvd` file in VisIt.
5. Compare the temperature field with the expected screenshot.
