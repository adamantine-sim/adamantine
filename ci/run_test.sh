#!/usr/bin/env bash

# Number of processors with default value
: ${N_PROCS:=2}

# To use module we need to source the bashrc and make sure than bashrc does not
# return because the shell is not interactive
sed -i '/return/d' ~/.bashrc
source ~/.bashrc
. /home/docker/spack/share/spack/setup-env.sh

# Load the modules
spack load openmpi
spack load boost
spack load dealii
spack load metis

# Install adamantine
mkdir -p /home/docker/build
cd /home/docker/build
cmake \
  -D CMAKE_BUILD_TYPE=Debug \
  -D ADAMANTINE_ENABLE_TESTS=ON \
  -D ADAMANTINE_ENABLE_COVERAGE=ON \
  -D CMAKE_CXX_FLAGS="-Wall -std=c++14" \
../adamantine

make -j${N_PROCS}

# indent_code is not a real test. Do not run it because it would required to
# change the permission of the adamantine directory
ctest -j${N_PROCS} -R test_

# Check code coverage
make coverage
