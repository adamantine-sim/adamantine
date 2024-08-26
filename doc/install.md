---
layout: page
title: Installation
nav_order: 2
---

# Installation

## Manual Installation
Installing *adamantine* requires:
* MPI
* A compiler that support C++17
* CMake: 3.15 or later
* Boost: 1.70.0 or later
* ArborX: 1.4.1 or later
* Trilinos: 14.4.0 or later
* deal.II: 9.5 or later

You need to compile ArborX with MPI support and deal.II with MPI, P4EST, ArborX, and Trilinos support. If you want to use Exodus file, you also need Trilinos with SEACAS support.
*adamantine* also optionally supports profiling through [Caliper](https://github.com/llnl/Caliper).

An example on how to install all the dependencies can be found in
`ci/Dockerfile`.

To configure *adamantine* use:
```CMake
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/dealii \
  -DBOOST_DIR=/path/to/boost \
/path/to/source/dir
```
Then simply use `make`. This will compile *adamantine* and create an executable
in a newly created `bin` subdirectory. You will find in this subdirectory the
executable and an example of input files.

The list of configuration options is:
* ADAMANTINE\_ENABLE\_ADIAK=ON/OFF
* ADAMANTINE\_ENABLE\_CALIPER=ON/OFF
* ADAMANTINE\_ENABLE\_COVERAGE=ON/OFF
* ADAMANTINE\_ENABLE\_TESTS=ON/OFF
* BOOST\_DIR=/path/to/boost
* CMAKE\_BUILD\_TYPE=Debug/Release
* CALIPER\_DIR=/path/to/caliper (optional)
* DEAL\_II\_DIR=/path/to/dealii

## Docker
You can pull Docker image containing *adamantine* using
```
docker pull rombur/adamantine:latest
```
*adamantine* can be found in `/home/adamantine/bin`
