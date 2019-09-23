#!/usr/bin/env bash

export PATH=/usr/local/bin:$PATH

# Install adamantine
mkdir -p /home/docker/build
cd /home/docker/build
cmake \
  -D CMAKE_BUILD_TYPE=Debug \
  -D ADAMANTINE_ENABLE_TESTS=ON \
  -D ADAMANTINE_ENABLE_COVERAGE=ON \
  -D CMAKE_CXX_FLAGS="-Wall" \
  -D DEAL_II_DIR=/opt/dealii \
../adamantine

make

# Variable only work with openmpi 4.0.2 and later
# export OMPI_ALLOW_RUN_AS_ROOT=1 
# export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# indent_code is not a real test. Do not run it because it would required to
# change the permission of the adamantine directory
ctest -V -R test_

# Check code coverage
make coverage
