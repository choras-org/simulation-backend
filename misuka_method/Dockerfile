FROM ubuntu:22.04 
#python 3.11 image statt ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    gnupg \
    lsb-release \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add LLVM repo for Clang 17
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" \
    > /etc/apt/sources.list.d/llvm.list

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    clang-17 \
    libc++-17-dev \
    libc++abi-17-dev \
    cmake \
    ninja-build \
    git \
    build-essential \
    zlib1g-dev \
    libpng-dev \
    libjpeg-dev \
    gmsh \
    && rm -rf /var/lib/apt/lists/*

ENV CC=clang-17
ENV CXX=clang++-17

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3.11 -m pip install --upgrade pip

# Install typing_extensions so stub generator works on Python < 3.11
RUN python3.11 -m pip install typing_extensions

# Clone misuka with ALL submodules
RUN git clone --recursive \
    https://github.com/misuka-renderer/misuka.git \
    /opt/misuka && \
    cd /opt/misuka && \
    git checkout 9845db9359ed7f489b5633195df3ce0acc09afe4 && \
    git submodule update --init --recursive

# Build misuka from source, explicitly pointing cmake to Python 3.10
RUN cd /opt/misuka && \
    mkdir build && cd build && \
    cmake -GNinja .. \
        -DMI_DEFAULT_VARIANTS="scalar_rgb;llvm_ad_rgb;scalar_acoustic;llvm_ad_acoustic" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE=/usr/bin/python3.11 \
        -DPython3_EXECUTABLE=/usr/bin/python3.11 && \
    ninja

# Add misuka build output to Python path
ENV PYTHONPATH="/opt/misuka/build/python"

# Install remaining Python dependencies


WORKDIR /app

COPY misuka_method /app/misuka_method
RUN python3.11 -m pip install /app/misuka_method

WORKDIR /app/misuka_method

CMD ["python3.11", "-m", "misuka_interface"]