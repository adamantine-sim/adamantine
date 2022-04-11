ARG BASE=nvidia/cuda:11.6.1-devel-ubuntu20.04
FROM $BASE

ARG N_PROCS=8
ARG CUDA=ON

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
      gcc \
      gfortran \
      clang-format \
      clang \
      build-essential \
      cmake \
      wget \
      curl \
      libcurl4-gnutls-dev \
      bison \
      python2.7-dev \
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
RUN export CMAKE_VERSION=3.21.0 && \
    export CMAKE_SHA256=d54ef6909f519740bc85cec07ff54574cd1e061f9f17357d9ace69f61c6291ce && \
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
RUN export OPENMPI_VERSION=4.1.3 && \
    export OPENMPI_VERSION_SHORT=4.1 && \
    export OPENMPI_SHA256=3d81d04c54efb55d3871a465ffb098d8d72c1f48ff1cbaf2580eb058567c0a3b && \
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
    if [ $CUDA ]; \
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
ENV PATH=${INSTALL_DIR}/openmpi/bin:$PATH

# Install Boost
RUN export BOOST_VERSION=1.76.0 && \
    export BOOST_VERSION_U=1_76_0 && \
    export BOOST_URL=https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_U}.tar.bz2 && \
    export BOOST_SHA256=f0397ba6e982c4450f27bf32a2a83292aba035b827a5623a14636ea583318c41 && \
    export BOOST_ARCHIVE=${ARCHIVE_DIR}/boost_${BOOST_VERSION_U}.tar.bz2 && \
    export BOOST_SOURCE_DIR=${SOURCE_DIR}/boost && \
    export BOOST_BUILD_DIR=${BUILD_DIR}/boost && \
    export BOOST_INSTALL_DIR=${INSTALL_DIR}/boost && \
    wget --quiet ${BOOST_URL} --output-document=${BOOST_ARCHIVE} && \
    echo "${BOOST_SHA256} ${BOOST_ARCHIVE}" | sha256sum -c && \
    mkdir -p ${BOOST_SOURCE_DIR} && \
    tar -xf ${BOOST_ARCHIVE} -C ${BOOST_SOURCE_DIR} --strip-components=1 && \
    cd ${BOOST_SOURCE_DIR} && \
    ./bootstrap.sh \
        --prefix=${BOOST_INSTALL_DIR} \
        && \
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
RUN export P4EST_VERSION=2.3.2 && \
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
    export HDF5_URL=http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.bz2 && \
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

# We want to enable SEACAS to be able to read EXODUS file but it clashes with ADOL-C. For now,
# we disable it
# Install Trilinos 13.2.0
RUN export TRILINOS_HASH=4a5f7906a6420ee2f9450367e9cc95b28c00d744 && \
    export TRILINOS_URL=https://github.com/trilinos/Trilinos/archive/${TRILINOS_HASH}.tar.gz && \
    export TRILINOS_ARCHIVE=${ARCHIVE_DIR}/trilinos.tar.gz && \
    export TRILINOS_SOURCE_DIR=${SOURCE_DIR}/trilinos && \
    export TRILINOS_BUILD_DIR=${BUILD_DIR}/trilinos && \
    export TRILINOS_INSTALL_DIR=${INSTALL_DIR}/trilinos && \
    wget --quiet ${TRILINOS_URL} --output-document=${TRILINOS_ARCHIVE} && \
    mkdir -p ${TRILINOS_SOURCE_DIR} && \
    tar -xf ${TRILINOS_ARCHIVE} -C ${TRILINOS_SOURCE_DIR} --strip-components=1 && \
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
        -DTrilinos_ENABLE_SEACAS=OFF \
        -DTrilinos_ENABLE_Sacado=ON \
        -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
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

# Install Adiak
RUN export ADIAK_VERSION=0.2.1 && \
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

# Install Caliper 2.7.0
RUN export CALIPER_HASH=1b7943f1854a5c285793c350339f31a9f2959bb0 && \
    export CALIPER_URL=https://github.com/LLNL/caliper/archive/${CALIPER_HASH}.tar.gz && \
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

# Install ArborX 1.2
RUN export ARBORX_HASH=dc372874ce4d1ec756de2503a30c293e2fd0776b && \
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
        -DCMAKE_CXX_EXTENSIONS=OFF \
        -DCMAKE_PREFIX_PATH=${TRILINOS_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ARBORX_INSTALL_DIR} \
        ${ARBORX_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${ARBORX_ARCHIVE} && \
    rm -rf ${ARBORX_BUILD_DIR} && \
    rm -rf ${ARBORX_SOURCE_DIR}
ENV ARBORX_DIR=${INSTALL_DIR}/arborx

# For now dealii-weak_forms requires Sacado, ADOL-C, and symengine
# Install ADOL-C 2.7.2
RUN export ADOLC_HASH=0e8f8a62715b7f71f05cbb3b371157aef5d24f94 && \
    export ADOLC_URL=https://github.com/coin-or/ADOL-C/archive/${ADOLC_HASH}.tar.gz && \
    export ADOLC_ARCHIVE=${ARCHIVE_DIR}/adolc.tar.gz && \
    export ADOLC_SOURCE_DIR=${SOURCE_DIR}/adolc && \
    export ADOLC_BUILD_DIR=${BUILD_DIR}/adolc && \
    export ADOLC_INSTALL_DIR=${INSTALL_DIR}/adolc && \
    wget --quiet ${ADOLC_URL} --output-document=${ADOLC_ARCHIVE} && \
    mkdir -p ${ADOLC_SOURCE_DIR} && \
    tar -xf ${ADOLC_ARCHIVE} -C ${ADOLC_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${ADOLC_BUILD_DIR} && \
    cd ${ADOLC_BUILD_DIR} && \
    ${ADOLC_SOURCE_DIR}/configure --with-boost=${BOOST_DIR} --prefix=${ADOLC_INSTALL_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${ADOLC_ARCHIVE} && \
    rm -rf ${ADOLC_BUILD_DIR} && \
    rm -rf ${ADOLC_SOURCE_DIR}
ENV ADOLC_DIR=${INSTALL_DIR}/adolc

# Install SymEngine 0.9.0
RUN export SYMENGINE_HASH=7b1880824c2cce98787ae29a317682ba6c294484 && \
    export SYMENGINE_URL=https://github.com/symengine/symengine/archive/${SYMENGINE_HASH}.tar.gz && \
    export SYMENGINE_ARCHIVE=${ARCHIVE_DIR}/symengine.tar.gz && \
    export SYMENGINE_SOURCE_DIR=${SOURCE_DIR}/symengine && \
    export SYMENGINE_BUILD_DIR=${BUILD_DIR}/symengine && \
    export SYMENGINE_INSTALL_DIR=${INSTALL_DIR}/symengine && \
    wget --quiet ${SYMENGINE_URL} --output-document=${SYMENGINE_ARCHIVE} && \
    mkdir -p ${SYMENGINE_SOURCE_DIR} && \
    tar -xf ${SYMENGINE_ARCHIVE} -C ${SYMENGINE_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${SYMENGINE_BUILD_DIR} && \
    cd ${SYMENGINE_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DCMAKE_INSTALL_PREFIX=${SYMENGINE_INSTALL_DIR} \
        ${SYMENGINE_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${SYMENGINE_ARCHIVE} && \
    rm -rf ${SYMENGINE_BUILD_DIR} && \
    rm -rf ${SYMENGINE_SOURCE_DIR}
ENV SYMENGINE_DIR=${INSTALL_DIR}/symengine

# Install deal.II 
RUN export DEAL_II_HASH=8197bc4197029df173581bc09c18030e9d824104 && \
    export DEAL_II_URL=https://github.com/dealii/dealii/archive/${DEAL_II_HASH}.tar.gz && \
    export DEAL_II_ARCHIVE=${ARCHIVE_DIR}/dealii.tar.gz && \
    export DEAL_II_SOURCE_DIR=${SOURCE_DIR}/dealii && \
    export DEAL_II_BUILD_DIR=${BUILD_DIR}/dealii && \
    export DEAL_II_INSTALL_DIR=${INSTALL_DIR}/dealii && \
    wget --quiet ${DEAL_II_URL} --output-document=${DEAL_II_ARCHIVE} && \
    mkdir -p ${DEAL_II_SOURCE_DIR} && \
    tar -xf ${DEAL_II_ARCHIVE} -C ${DEAL_II_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${DEAL_II_BUILD_DIR} && cd ${DEAL_II_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=DebugRelease \
        -DCMAKE_CXX_FLAGS="-std=c++17" \
        -DCMAKE_CUDA_FLAGS="-arch=sm_70 -std=c++17 --extended-lambda" \
        -DCMAKE_CXX_STANDARD=17 \
        -DDEAL_II_WITH_64BIT_INDICES=ON \
        -DDEAL_II_WITH_MPI=ON \
        -DDEAL_II_WITH_P4EST=ON \
        -DP4EST_DIR=${P4EST_DIR} \
        -DDEAL_II_WITH_CUDA=${CUDA} \
        -DDEAL_II_WITH_ARBORX=ON \
        -DARBORX_DIR=${ARBORX_DIR} \
        -DDEAL_II_WITH_TRILINOS=ON \
        -DTRILINOS_DIR=${TRILINOS_DIR} \
        -DDEAL_II_TRILINOS_WITH_SACADO=ON \
        -DDEAL_II_TRILINOS_WITH_SEACAS=OFF \
        -DDEAL_II_WITH_ADOLC=ON \
        -DADOLC_DIR=${ADOLC_DIR} \
        -DDEAL_II_WITH_SYMENGINE=ON \
        -DSYMENGINE_DIR=${SYMENGINE_DIR} \
        -DDEAL_II_COMPONENT_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=${DEAL_II_INSTALL_DIR} \
        ${DEAL_II_SOURCE_DIR}  && \
    make -j${N_PROCS} install && \
    rm -rf ${DEAL_II_ARCHIVE} && \
    rm -rf ${DEAL_II_BUILD_DIR} 
# We keep the source file for debugging purpose
ENV DEAL_II_DIR=${INSTALL_DIR}/dealii

# There is a bug when using multithreading. Disabling it for now.
ENV DEAL_II_NUM_THREADS=1

# Install dealii-weak_forms
RUN export DEALII_WEAK_FORMS_HASH=962180dd9a60d40125e4cfbadb4b7e58a1a6bb08 && \
    export DEALII_WEAK_FORMS_URL=https://github.com/jppelteret/dealii-weak_forms/archive/${DEALII_WEAK_FORMS_HASH}.tar.gz && \
    export DEALII_WEAK_FORMS_ARCHIVE=${ARCHIVE_DIR}/dealii-weak_forms.tar.gz && \
    export DEALII_WEAK_FORMS_SOURCE_DIR=${SOURCE_DIR}/dealii-weak_forms && \
    export DEALII_WEAK_FORMS_BUILD_DIR=${BUILD_DIR}/dealii-weak_forms && \
    export DEALII_WEAK_FORMS_INSTALL_DIR=${INSTALL_DIR}/dealii-weak_forms && \
    wget --quiet ${DEALII_WEAK_FORMS_URL} --output-document=${DEALII_WEAK_FORMS_ARCHIVE} && \
    mkdir -p ${DEALII_WEAK_FORMS_SOURCE_DIR} && \
    tar -xf ${DEALII_WEAK_FORMS_ARCHIVE} -C ${DEALII_WEAK_FORMS_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${DEALII_WEAK_FORMS_BUILD_DIR} && cd ${DEALII_WEAK_FORMS_BUILD_DIR} && \
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${DEALII_WEAK_FORMS_INSTALL_DIR} \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_DOCUMENTATION=OFF \
      -DBUILD_TESTS=OFF \
      -DDEAL_II_DIR=${DEAL_II_DIR} \
      ${DEALII_WEAK_FORMS_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${DEALII_WEAK_FORMS_ARCHIVE} && \
    rm -rf ${DEALII_WEAK_FORMS_BUILD_DIR} 
# We keep the source file for debugging purpose
ENV DEALII_WEAK_FORMS_DIR=${INSTALL_DIR}/dealii-weak_forms
