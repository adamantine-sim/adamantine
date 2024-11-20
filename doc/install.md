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

You need to compile ArborX with MPI support and deal.II with MPI, p4est, ArborX, and Trilinos support. If you want to use Exodus file, you also need Trilinos with SEACAS support.
*adamantine* also optionally supports profiling through [Caliper](https://github.com/llnl/Caliper).

An example on how to install all the dependencies can be found in
`ci/Dockerfile`. They can also be installed through
[spack](https://github.com/spack/spack).

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
* CALIPER\_DIR=/path/to/caliper (only if you enabled CALIPER)
* DEAL\_II\_DIR=/path/to/dealii

## Docker
You can pull the [Docker](https://en.wikipedia.org/wiki/Docker_(software)) image containing the version of *adamantine* on master using:
``` bash
docker pull rombur/adamantine:latest
```
The 1.0 release version is available using:
``` bash
docker pull rombur/adamantine:1.0
```
To start an interactive container use:
``` bash
docker run --rm -it rombur/adamantine:latest bash
```
or
``` bash
docker run --rm -it rombur/adamantine:1.0 bash
```

You will find *adamantine* in `/home/adamantine/bin`. More `docker run` options 
can be found in the Docker [documentation](https://docs.docker.com/reference/cli/docker/container/run/).

There are two methods to move file to/from the Docker container:
1. Mount a volume using the option `-v`. You launch the container using:
``` bash
docker run --rm -it -v /path/to/computer/folder:/path/to/image/folder rombur/adamantine:1.0 bash
```
Every file in `/path/to/computer/folder` (resp. `/path/to/image/folder`) will 
be visible in `/path/to/image/folder` (resp. `/path/to/computer/folder`). Note
that any file created inside the Docker container is created as *root* not as
a regular user. 
2. Use `docker cp` to copy the files (see
   [here](https://docs.docker.com/reference/cli/docker/container/cp/)).

The Docker images cannot use GPUs.

## NIX
You can use [NIX](https://nixos.org) to install the development version and the
latest release of *adamantine*. You need to enable [Flakes](https://nixos.wiki/wiki/Flakes). 
To install the vesion on master, use:
``` bash
nix develop github:adamantine-sim/adamantine
```
The 1.0 release version is available using:
``` bash
nix develop github:adamantine-sim/adamantine#release
```
