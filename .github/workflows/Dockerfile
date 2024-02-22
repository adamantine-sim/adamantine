FROM rombur/adamantine-stack:no_gpu-latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
      wget \
      curl \
      bison \
      bash-completion \
      gdb \
      zlib1g-dev \
      python3-dev \
      python3-setuptools \
      vim-nox \
      ninja-build \
      && \
      apt-get clean && rm -rf /var/lib/apt/lists/

RUN cd /home && git clone https://github.com/adamantine-sim/adamantine && \
    cd adamantine && \
    mkdir build && cd build && \
    cmake \
      -DDEAL_II_DIR=${DEAL_II_DIR} \
      -DCMAKE_BUILD_TYPE=Release\
      -DCMAKE_CXX_FLAGS="-g -ffast-math" \
      -DADAMANTINE_ENABLE_ADIAK=ON \
      -DADAMANTINE_ENABLE_CALIPER=ON \
    ../ && \
    make && mv bin ../ && cd ../ && rm -r build
