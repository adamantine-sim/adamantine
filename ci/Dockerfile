FROM nvidia/cuda:11.0-devel

ARG N_PROCS=12

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
      gcc \
      gfortran \
      clang-format \
      build-essential \
      cmake \
      wget \
      curl \
      bison \
      python2.7-dev \
      git \
      lcov \
      zlib1g-dev \
      libopenblas-dev \
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
RUN export CMAKE_VERSION=3.18.3 && \
    export CMAKE_VERSION_SHORT=3.18 && \
    export CMAKE_SHA256=eb23bac95873cc0bc3946eed5d1aea6876ac03f7c0dcc6ad3a05462354b08228 && \
    export CMAKE_URL=https://cmake.org/files/v${CMAKE_VERSION_SHORT}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
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
RUN export OPENMPI_VERSION=4.0.5 && \
    export OPENMPI_VERSION_SHORT=4.0 && \
    export OPENMPI_SHA256=c58f3863b61d944231077f344fe6b4b8fbb83f3d1bc93ab74640bf3e5acac009 && \
    export OPENMPI_URL=https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSION_SHORT}/downloads/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    export OPENMPI_ARCHIVE=${ARCHIVE_DIR}/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    export OPENMPI_SOURCE_DIR=${SOURCE_DIR}/openmpi && \
    export OPENMPI_BUILD_DIR=${BUILD_DIR}/openmpi && \
    export OPENMPI_INSTALL_DIR=${INSTALL_DIR}/openmpi && \
    wget --quiet ${OPENMPI_URL} --output-document=${OPENMPI_ARCHIVE} && \
    echo "${OPENMPI_SHA256} ${OPENMPI_ARCHIVE}" | sha256sum -c && \
    mkdir -p ${OPENMPI_SOURCE_DIR} && \
    tar -xf ${OPENMPI_ARCHIVE} -C ${OPENMPI_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${OPENMPI_BUILD_DIR} && \
    cd ${OPENMPI_BUILD_DIR} && \
    ${OPENMPI_SOURCE_DIR}/configure --with-cuda=/usr/local/cuda --prefix=${OPENMPI_INSTALL_DIR} && \
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
RUN export BOOST_VERSION=1.73.0 && \
    export BOOST_VERSION_U=1_73_0 && \
    export BOOST_URL=https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_U}.tar.bz2 && \
    export BOOST_SHA256=4eb3b8d442b426dc35346235c8733b5ae35ba431690e38c6a8263dce9fcbb402 && \
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
ENV BOOST_ROOT=${INSTALL_DIR}/boost

# Install p4est
RUN export P4EST_VERSION=2.2 && \
    export P4EST_URL=http://p4est.github.io/release/p4est-${P4EST_VERSION}.tar.gz && \
    export P4EST_ARCHIVE=${ARCHIVE_DIR}/p4est-${P4EST_VERSION}.tar.gz && \
    export P4EST_SOURCE_DIR=${SOURCE_DIR}/p4est && \
    export P4EST_BUILD_DIR=${BUILD_DIR}/p4est && \
    export P4EST_INSTALL_DIR=${INSTALL_DIR}/p4est && \
    wget --quiet ${P4EST_URL} --output-document=${P4EST_ARCHIVE} && \
    mkdir -p ${P4EST_SOURCE_DIR} && \
    cd ${P4EST_SOURCE_DIR} && \
    wget --quiet https://www.dealii.org/9.0.0/external-libs/p4est-setup.sh && \
    bash ./p4est-setup.sh ${P4EST_ARCHIVE} ${P4EST_INSTALL_DIR} && \
    rm -rf ${P4EST_ARCHIVE} && \
    rm -rf ${P4EST_BUILD_DIR} && \
    rm -rf ${P4EST_SOURCE_DIR}
ENV P4EST_DIR=${INSTALL_DIR}/p4est

# Install deal.II 
RUN export DEAL_II_HASH=551e93708c2f0b26f916795bd15f2f0e575e295c && \
    export DEAL_II_URL=https://github.com/Rombur/dealii/archive/${DEAL_II_HASH}.tar.gz && \
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
        -DCMAKE_CUDA_FLAGS="-std=c++17" \
        -DCMAKE_CXX_STANDARD=17 \
        -DDEAL_II_WITH_64BIT_INDICES=ON \
        -DDEAL_II_WITH_MPI=ON \
        -DDEAL_II_WITH_P4EST=ON \
        -DP4EST_DIR=${P4EST_DIR} \
        -DDEAL_II_WITH_CUDA=ON \
        -DCMAKE_INSTALL_PREFIX=${DEAL_II_INSTALL_DIR} \
        ${DEAL_II_SOURCE_DIR}  && \
    make -j${N_PROCS} install && \
    rm -rf ${DEAL_II_ARCHIVE} && \
    rm -rf ${DEAL_II_BUILD_DIR} && \
    rm -rf ${DEAL_II_SOURCE_DIR}
ENV DEAL_II_DIR=${INSTALL_DIR}/dealii
