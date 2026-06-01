ARG BASE=nvidia/cuda:12.6.2-devel-ubuntu24.04
FROM $BASE

ARG N_PROCS=8
ARG CUDA=ON

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
      cmake \
      gcc \
      gfortran \
      clang-format \
      build-essential \
      wget \
      curl \
      libboost-all-dev \
      libcurl4-gnutls-dev \
      bison \
      python3-dev \
      git \
      lcov \
      zlib1g-dev \
      libopenblas-dev \
      libdw-dev \
      libpfm4-dev \
      libunwind-dev \
      libgmp-dev \
      openmpi-bin \
      libopenmpi-dev \
      libp4est-dev \
      libhdf5-dev \
      libnetcdf-dev \
      libsundials-dev \
      && \
      apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PREFIX=/scratch
ENV ARCHIVE_DIR=${PREFIX}/archive
ENV SOURCE_DIR=${PREFIX}/source
ENV BUILD_DIR=${PREFIX}/build
ENV INSTALL_DIR=/opt

RUN mkdir -p ${PREFIX} && \
    cd ${PREFIX} && \
    mkdir archive && \
    mkdir source && \
    mkdir build

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_btl_base_warn_component_unused='0'
ENV OMPI_CC=cc
ENV OMPI_CXX=c++

# Install Trilinos 14.4.0
RUN export TRILINOS_HASH=975307431d60d0859ebaa27c9169cbb1d4287513 && \
    export TRILINOS_URL=https://github.com/trilinos/Trilinos/archive/${TRILINOS_HASH}.tar.gz && \
    export TRILINOS_ARCHIVE=${ARCHIVE_DIR}/trilinos.tar.gz && \
    export TRILINOS_SOURCE_DIR=${SOURCE_DIR}/trilinos && \
    export TRILINOS_BUILD_DIR=${BUILD_DIR}/trilinos && \
    export TRILINOS_INSTALL_DIR=${INSTALL_DIR}/trilinos && \
    wget --quiet ${TRILINOS_URL} --output-document=${TRILINOS_ARCHIVE} && \
    mkdir -p ${TRILINOS_SOURCE_DIR} && \
    tar -xf ${TRILINOS_ARCHIVE} -C ${TRILINOS_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${TRILINOS_INSTALL_DIR} && \
    cp -r ${TRILINOS_SOURCE_DIR}/packages/kokkos/bin ${TRILINOS_INSTALL_DIR} && \
    export OMPI_CXX=${TRILINOS_INSTALL_DIR}/bin/nvcc_wrapper && \
    mkdir ${TRILINOS_BUILD_DIR} && \
    cd ${TRILINOS_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=ON \
        -DTPL_ENABLE_MPI=ON \
        -DTPL_ENABLE_BLAS=ON \
        -DTPL_ENABLE_LAPACK=ON \
        -DTPL_ENABLE_Boost=ON \
        -DTPL_ENABLE_BoostLib=ON \
        -DTrilinos_ENABLE_ALL_PACKAGES=OFF \
        -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
        -DTrilinos_ENABLE_TESTS=OFF \
        -DTrilinos_ENABLE_EXAMPLES=OFF \
        -DTrilinos_ENABLE_Amesos=ON \
        -DTrilinos_ENABLE_AztecOO=ON \
        -DTrilinos_ENABLE_Epetra=ON \
        -DTrilinos_ENABLE_Ifpack=ON \
        -DTrilinos_ENABLE_Kokkos=ON \
        -DTrilinos_ENABLE_ML=ON \
        -DTrilinos_ENABLE_SEACAS=ON \
        -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
        -DKokkos_ENABLE_CUDA=$CUDA \
        -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
        -DKokkos_ARCH_AMPERE80=ON \
        -DBLAS_LIBRARY_NAMES="openblas" \
        -DBLAS_LIBRARY_DIRS=${OPENBLAS_DIR}/lib \
        -DLAPACK_LIBRARY_NAMES="openblas" \
        -DLAPACK_LIBRARY_DIRS=${OPENBLAS_DIR}/lib \
        -DTPL_ENABLE_Netcdf=ON \
        -DTPL_ENABLE_Matio=OFF \
        -DTPL_ENABLE_X11=OFF \
        -DCMAKE_INSTALL_PREFIX=${TRILINOS_INSTALL_DIR} \
        ${TRILINOS_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${TRILINOS_ARCHIVE} && \
    rm -rf ${TRILINOS_BUILD_DIR} && \
    rm -rf ${TRILINOS_SOURCE_DIR}
ENV TRILINOS_DIR=${INSTALL_DIR}/trilinos
ENV OMPI_CXX=${TRILINOS_DIR}/bin/nvcc_wrapper

# Install Adiak
RUN export ADIAK_VERSION=0.4.0 && \
    export ADIAK_URL=https://github.com/LLNL/Adiak/releases/download/v${ADIAK_VERSION}/adiak-${ADIAK_VERSION}.tar.gz && \
    export ADIAK_ARCHIVE=${ARCHIVE_DIR}/adiak.tar.gz && \
    export ADIAK_SOURCE_DIR=${SOURCE_DIR}/adiak && \
    export ADIAK_BUILD_DIR=${BUILD_DIR}/adiak && \
    export ADIAK_INSTALL_DIR=${INSTALL_DIR}/adiak && \
    wget --quiet ${ADIAK_URL} --output-document=${ADIAK_ARCHIVE} && \
    mkdir -p ${ADIAK_SOURCE_DIR} && \
    tar -xf ${ADIAK_ARCHIVE} -C ${ADIAK_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${ADIAK_BUILD_DIR} && \
    cd ${ADIAK_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=${ADIAK_INSTALL_DIR} \
        ${ADIAK_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${ADIAK_ARCHIVE} && \
    rm -rf ${ADIAK_BUILD_DIR} && \
    rm -rf ${ADIAK_SOURCE_DIR}
ENV ADIAK_DIR=${INSTALL_DIR}/adiak

# Install Caliper 2.11.0
RUN export CALIPER_VERSION=2.11.0 && \
    export CALIPER_URL=https://github.com/LLNL/Caliper/archive/refs/tags/v${CALIPER_VERSION}.tar.gz && \
    export CALIPER_ARCHIVE=${ARCHIVE_DIR}/caliper.tar.gz && \
    export CALIPER_SOURCE_DIR=${SOURCE_DIR}/caliper && \
    export CALIPER_BUILD_DIR=${BUILD_DIR}/caliper && \
    export CALIPER_INSTALL_DIR=${INSTALL_DIR}/caliper && \
    wget --quiet ${CALIPER_URL} --output-document=${CALIPER_ARCHIVE} && \
    mkdir -p ${CALIPER_SOURCE_DIR} && \
    tar -xf ${CALIPER_ARCHIVE} -C ${CALIPER_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${CALIPER_BUILD_DIR} && \
    cd ${CALIPER_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DWITH_ADIAK=ON \
        -DWITH_LIBDW=ON \
        -DWITH_LIBPFM=ON \
        -DWITH_LIBUNWIND=ON \
        -DWITH_MPI=ON \
        -DWITH_SAMPLER=ON \
        -DCMAKE_INSTALL_PREFIX=${CALIPER_INSTALL_DIR} \
        ${CALIPER_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${CALIPER_ARCHIVE} && \
    rm -rf ${CALIPER_BUILD_DIR} && \
    rm -rf ${CALIPER_SOURCE_DIR}
ENV CALIPER_DIR=${INSTALL_DIR}/caliper

# Install ArborX 1.5   
RUN export ARBORX_HASH=e026f8237dbd0aaf2ddf2e3bf0e492071db854a2 && \
    export ARBORX_URL=https://github.com/arborx/ArborX/archive/${ARBORX_HASH}.tar.gz && \
    export ARBORX_ARCHIVE=${ARCHIVE_DIR}/arborx.tar.gz && \
    export ARBORX_SOURCE_DIR=${SOURCE_DIR}/arborx && \
    export ARBORX_BUILD_DIR=${BUILD_DIR}/arborx && \
    export ARBORX_INSTALL_DIR=${INSTALL_DIR}/arborx && \
    wget --quiet ${ARBORX_URL} --output-document=${ARBORX_ARCHIVE} && \
    mkdir -p ${ARBORX_SOURCE_DIR} && \
    tar -xf ${ARBORX_ARCHIVE} -C ${ARBORX_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${ARBORX_BUILD_DIR} && \
    cd ${ARBORX_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_CXX_COMPILER=${TRILINOS_DIR}/bin/nvcc_wrapper \
        -DCMAKE_CXX_EXTENSIONS=OFF \
        -DARBORX_ENABLE_MPI=ON \
        -DCMAKE_PREFIX_PATH=${TRILINOS_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ARBORX_INSTALL_DIR} \
        ${ARBORX_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${ARBORX_ARCHIVE} && \
    rm -rf ${ARBORX_BUILD_DIR} && \
    rm -rf ${ARBORX_SOURCE_DIR}
ENV ARBORX_DIR=${INSTALL_DIR}/arborx

# Install deal.II 9.6.2
RUN export DEAL_II_VERSION=9.6.2 && \
    export DEAL_II_URL=https://github.com/dealii/dealii/releases/download/v${DEAL_II_VERSION}/dealii-${DEAL_II_VERSION}.tar.gz && \
    export DEAL_II_ARCHIVE=${ARCHIVE_DIR}/dealii.tar.gz && \
    export DEAL_II_SOURCE_DIR=${SOURCE_DIR}/dealii && \
    export DEAL_II_BUILD_DIR=${BUILD_DIR}/dealii && \
    export DEAL_II_INSTALL_DIR=${INSTALL_DIR}/dealii && \
    wget --quiet ${DEAL_II_URL} --output-document=${DEAL_II_ARCHIVE} && \
    mkdir -p ${DEAL_II_SOURCE_DIR} && \
    tar -xf ${DEAL_II_ARCHIVE} -C ${DEAL_II_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${DEAL_II_BUILD_DIR} && cd ${DEAL_II_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=DebugRelease \
        -DCMAKE_CXX_COMPILER=${TRILINOS_DIR}/bin/nvcc_wrapper \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_EXTENSIONS=OFF \
        -DDEAL_II_EARLY_DEPRECATIONS=ON \
        -DDEAL_II_WITH_TBB=OFF \
        -DDEAL_II_WITH_64BIT_INDICES=ON \
        -DDEAL_II_WITH_COMPLEX_VALUES=OFF \
        -DDEAL_II_WITH_MPI=ON \
        -DDEAL_II_WITH_SUNDIALS=ON \
        -DDEAL_II_WITH_P4EST=ON \
        -DDEAL_II_WITH_ARBORX=ON \
        -DARBORX_DIR=${ARBORX_DIR} \
        -DDEAL_II_WITH_TRILINOS=ON \
        -DTRILINOS_DIR=${TRILINOS_DIR} \
        -DDEAL_II_TRILINOS_WITH_SEACAS=ON \
        -DDEAL_II_COMPONENT_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=${DEAL_II_INSTALL_DIR} \
        ${DEAL_II_SOURCE_DIR}  && \
    make -j${N_PROCS} install && \
    rm -rf ${DEAL_II_ARCHIVE} && \
    rm -rf ${DEAL_II_BUILD_DIR} 
# We keep the source file for debugging purpose
ENV DEAL_II_DIR=${INSTALL_DIR}/dealii
