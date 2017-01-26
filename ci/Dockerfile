FROM ubuntu

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
      gcc \
      gfortran \
      clang-format-3.7 \
      build-essential \
      cmake \
      wget \
      curl \
      environment-modules \
      bison \
      python \
      git \
      lcov

# Create a user
RUN useradd -ms /bin/bash docker
USER docker
WORKDIR /home/docker

RUN echo 'module() { eval `/usr/bin/modulecmd bash $*`; }' >> ~/.bashrc
RUN bash -c "source ~/.bashrc"

# Use spack to install the latest version of the libraries
RUN cd /home/docker && git clone https://github.com/llnl/spack.git
ENV PATH=/home/docker/spack/bin:$PATH
RUN spack install boost +mpi +python +icu
RUN spack install dealii@develop +int64 ~arpack ~gsl ~hdf5 ~metis +mpi ~netcdf ~oce +p4est ~petsc +python ~slepc ~trilinos ^boost+mpi+python+icu
