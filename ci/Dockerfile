ARG BASE=nvidia/cuda:12.6.2-devel-ubuntu24.04
FROM $BASE

ARG N_PROCS=8
ARG CUDA=ON

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
      gcc \
      gfortran \
      clang-format \
      build-essential \
      wget \
      curl \
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

# Install CMake
RUN export CMAKE_VERSION=3.30.5 && \
    export CMAKE_SHA256=f747d9b23e1a252a8beafb4ed2bc2ddf78cff7f04a8e4de19f4ff88e9b51dc9d && \
    export CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    export CMAKE_ARCHIVE=${ARCHIVE_DIR}/cmake.tar.gz && \
    export CMAKE_BUILD_DIR=${BUILD_DIR}/cmake && \
    wget --quiet ${CMAKE_URL} --output-document=${CMAKE_ARCHIVE} && \
    echo "${CMAKE_SHA256} ${CMAKE_ARCHIVE}" | sha256sum -c && \
    mkdir -p ${CMAKE_BUILD_DIR} && \
    tar xf ${CMAKE_ARCHIVE} -C ${CMAKE_BUILD_DIR} --strip-components=1 && \
    mv ${CMAKE_BUILD_DIR} ${INSTALL_DIR} && \
    rm -rf ${CMAKE_ARCHIVE} && \
    rm -rf ${CMAKE_BUILD_DIR}
ENV PATH=${INSTALL_DIR}/cmake/bin:$PATH

# Install OpenMPI
RUN export OPENMPI_VERSION=5.0.5 && \
    export OPENMPI_VERSION_SHORT=5.0 && \
    export OPENMPI_SHA256=6588d57c0a4bd299a24103f4e196051b29e8b55fbda49e11d5b3d32030a32776 && \
    export OPENMPI_URL=https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSION_SHORT}/downloads/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    export OPENMPI_ARCHIVE=${ARCHIVE_DIR}/openmpi.tar.bz2 && \
    export OPENMPI_SOURCE_DIR=${SOURCE_DIR}/openmpi && \
    export OPENMPI_BUILD_DIR=${BUILD_DIR}/openmpi && \
    export OPENMPI_INSTALL_DIR=${INSTALL_DIR}/openmpi && \
    wget --quiet ${OPENMPI_URL} --output-document=${OPENMPI_ARCHIVE} && \
    echo "${OPENMPI_SHA256} ${OPENMPI_ARCHIVE}" | sha256sum -c && \
    mkdir -p ${OPENMPI_SOURCE_DIR} && \
    tar -xf ${OPENMPI_ARCHIVE} -C ${OPENMPI_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${OPENMPI_BUILD_DIR} && \
    cd ${OPENMPI_BUILD_DIR} && \
    if [ $CUDA = ON ]; \
    then \
    ${OPENMPI_SOURCE_DIR}/configure --with-cuda=/usr/local/cuda --prefix=${OPENMPI_INSTALL_DIR};  \
    else \
    ${OPENMPI_SOURCE_DIR}/configure --prefix=${OPENMPI_INSTALL_DIR}; \
    fi && \
    make -j${N_PROCS} install && \
    rm -rf ${OPENMPI_ARCHIVE} && \
    rm -rf ${OPENMPI_BUILD_DIR} && \
    rm -rf ${OPENMPI_SOURCE_DIR}
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_btl_base_warn_component_unused='0'
ENV OMPI_CC=cc
ENV OMPI_CXX=c++
ENV PATH=${INSTALL_DIR}/openmpi/bin:$PATH

# Install Boost
RUN export BOOST_VERSION=1.86.0 && \
    export BOOST_VERSION_U=1_86_0 && \
    export BOOST_URL=https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_U}.tar.bz2 && \
    export BOOST_SHA256=1bed88e40401b2cb7a1f76d4bab499e352fa4d0c5f31c0dbae64e24d34d7513b && \
    export BOOST_ARCHIVE=${ARCHIVE_DIR}/boost_${BOOST_VERSION_U}.tar.bz2 && \
    export BOOST_SOURCE_DIR=${SOURCE_DIR}/boost && \
    export BOOST_BUILD_DIR=${BUILD_DIR}/boost && \
    export BOOST_INSTALL_DIR=${INSTALL_DIR}/boost && \
    wget --quiet ${BOOST_URL} --output-document=${BOOST_ARCHIVE} && \
    echo "${BOOST_SHA256} ${BOOST_ARCHIVE}" | sha256sum -c && \
    mkdir -p ${BOOST_SOURCE_DIR} && \
    tar -xf ${BOOST_ARCHIVE} -C ${BOOST_SOURCE_DIR} --strip-components=1 && \
    cd ${BOOST_SOURCE_DIR} && \
    ./bootstrap.sh --prefix=${BOOST_INSTALL_DIR}; \
    echo "using mpi ;" >> project-config.jam && \
    ./b2 -j${N_PROCS}\
        --build-dir=${BOOST_BUILD_DIR} \
        hardcode-dll-paths=true dll-path=${BOOST_INSTALL_DIR}/lib \
        link=shared \
        variant=release \
        install \
        && \
    rm -rf ${BOOST_ARCHIVE} && \
    rm -rf ${BOOST_BUILD_DIR} && \
    rm -rf ${BOOST_SOURCE_DIR}
ENV BOOST_DIR=${INSTALL_DIR}/boost

# Install p4est
RUN export P4EST_VERSION=2.3.3 && \
    export P4EST_URL=http://p4est.github.io/release/p4est-${P4EST_VERSION}.tar.gz && \
    export P4EST_ARCHIVE=${ARCHIVE_DIR}/p4est-${P4EST_VERSION}.tar.gz && \
    export P4EST_SOURCE_DIR=${SOURCE_DIR}/p4est && \
    export P4EST_BUILD_DIR=${BUILD_DIR}/p4est && \
    export P4EST_INSTALL_DIR=${INSTALL_DIR}/p4est && \
    wget --quiet ${P4EST_URL} --output-document=${P4EST_ARCHIVE} && \
    mkdir -p ${P4EST_SOURCE_DIR} && \
    cd ${P4EST_SOURCE_DIR} && \
    wget --quiet https://raw.githubusercontent.com/dealii/dealii/master/doc/external-libs/p4est-setup.sh && \
    bash ./p4est-setup.sh ${P4EST_ARCHIVE} ${P4EST_INSTALL_DIR} && \
    rm -rf ${P4EST_ARCHIVE} && \
    rm -rf ${P4EST_BUILD_DIR} && \
    rm -rf ${P4EST_SOURCE_DIR}
ENV P4EST_DIR=${INSTALL_DIR}/p4est

RUN export HDF5_VERSION=1.12.0 && \
    export HDF5_URL=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.bz2 && \
    export HDF5_MD5=1fa68c4b11b6ef7a9d72ffa55995f898 && \
    export HDF5_ARCHIVE=${ARCHIVE_DIR}/hdf5.tar.bz2 && \
    export HDF5_SOURCE_DIR=${SOURCE_DIR}/hdf5 && \
    export HDF5_BUILD_DIR=${BUILD_DIR}/hdf5 && \
    export HDF5_INSTALL_DIR=${INSTALL_DIR}/hdf5 && \
    wget --quiet ${HDF5_URL} --output-document=${HDF5_ARCHIVE} && \
    echo "${HDF5_MD5} ${HDF5_ARCHIVE}" | md5sum -c && \
    mkdir -p ${HDF5_SOURCE_DIR} && \
    tar -xf ${HDF5_ARCHIVE} -C ${HDF5_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${HDF5_BUILD_DIR} && \
    cd ${HDF5_BUILD_DIR} && \
    ${HDF5_SOURCE_DIR}/configure \
        --prefix=${HDF5_INSTALL_DIR} \
        --enable-shared \
        --disable-static \
        --enable-parallel \
        && \
    make -j${NPROC} install && \
    rm -rf ${HDF5_ARCHIVE} && \
    rm -rf ${HDF5_BUILD_DIR} && \
    rm -rf ${HDF5_SOURCE_DIR}
ENV HDF5_DIR=/opt/hdf5

# Install NetCDF
RUN export NETCDF_VERSION=4.7.4 && \
    export NETCDF_URL=https://github.com/Unidata/netcdf-c/archive/v${NETCDF_VERSION}.tar.gz && \
    export NETCDF_ARCHIVE=${ARCHIVE_DIR}/netcdf.tar.gz && \
    export NETCDF_SOURCE_DIR=${SOURCE_DIR}/netcdf && \
    export NETCDF_BUILD_DIR=${BUILD_DIR}/netcdf && \
    export NETCDF_INSTALL_DIR=${INSTALL_DIR}/netcdf && \
    wget --quiet ${NETCDF_URL} --output-document=${NETCDF_ARCHIVE} && \
    mkdir -p ${NETCDF_SOURCE_DIR} && \
    tar -xf ${NETCDF_ARCHIVE} -C ${NETCDF_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${NETCDF_BUILD_DIR} && \
    cd ${NETCDF_BUILD_DIR} && \
    ${NETCDF_SOURCE_DIR}/configure \
        --prefix=${NETCDF_INSTALL_DIR} \
        --enable-netcdf-4 \
        --enable-shared \
        --disable-static \
        CC=mpicc \
        CFLAGS="-I${HDF5_DIR}/include" \
        LDFLAGS="-L${HDF5_DIR}/lib -lhdf5" \
        && \
    make -j${NPROC} install && \
    rm -rf ${NETCDF_ARCHIVE} && \
    rm -rf ${NETCDF_BUILD_DIR} && \
    rm -rf ${NETCDF_SOURCE_DIR}
ENV NETCDF_DIR=/opt/netcdf

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
        -DBoost_INCLUDE_DIRS=${BOOST_DIR}/include \
        -DTPL_ENABLE_BoostLib=ON \
        -DBoostLib_INCLUDE_DIRS=${BOOST_DIR}/include \
        -DBoostLib_LIBRARY_DIRS=${BOOST_DIR}/lib \
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
        -DNetcdf_INCLUDE_DIRS=$NETCDF_DIR/include \
        -DNetcdf_LIBRARY_DIRS=$NETCDF_DIR/lib \
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
        -DDEAL_II_WITH_P4EST=ON \
        -DP4EST_DIR=${P4EST_DIR} \
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
