# ---------- Stage 1: builder ----------
FROM ubuntu:22.04 AS builder

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

RUN for i in 1 2 3 4 5; do \
        add-apt-repository ppa:deadsnakes/ppa && break; \
        echo "add-apt-repository failed (attempt $i/5), retrying in $((i*5))s..." >&2; \
        sleep $((i*5)); \
        [ "$i" = "5" ] && exit 1; \
    done && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    clang-17 \
    libc++-17-dev \
    libc++abi-17-dev \
    libllvm17 \
    cmake \
    ninja-build \
    ccache \
    git \
    build-essential \
    zlib1g-dev \
    libpng-dev \
    libjpeg-dev \
    gmsh \
    && rm -rf /var/lib/apt/lists/*

ENV CC=clang-17
ENV CXX=clang++-17
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_C_COMPILER_LAUNCHER=ccache

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3.11 -m pip install --upgrade pip typing_extensions

# Shallow clone of misuka from the lgpl branch
RUN mkdir -p /opt/misuka && cd /opt/misuka && \
    git init && \
    git remote add origin https://github.com/misuka-renderer/misuka.git && \
    git fetch --depth 1 origin lgpl && \
    git checkout FETCH_HEAD && \
    git submodule update --init --recursive --depth 1

# Build misuka from source, with ccache to speed up rebuilds
RUN --mount=type=cache,target=/root/.ccache \
    cd /opt/misuka && \
    mkdir build && cd build && \
    cmake -GNinja .. \
        -DMI_DEFAULT_VARIANTS="scalar_rgb;llvm_ad_rgb;llvm_ad_acoustic" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE=/usr/bin/python3.11 \
        -DPython3_EXECUTABLE=/usr/bin/python3.11 && \
    ninja && \
    ccache -s

WORKDIR /app
COPY misuka_method /app/misuka_method
RUN python3.11 -m pip install --target=/opt/misuka_python \
        numpy \
        requests \
        meshio \
        'pyfar<0.10.0' \
        'pyrato>=1.0.0' \
        pandas && \
    python3.11 -m pip install --no-deps --target=/opt/misuka_python /app/misuka_method

# ---------- Stage 2: runtime ----------
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common ca-certificates gnupg wget lsb-release && \
    for i in 1 2 3 4 5; do \
        add-apt-repository ppa:deadsnakes/ppa && break; \
        echo "add-apt-repository failed (attempt $i/5), retrying in $((i*5))s..." >&2; \
        sleep $((i*5)); \
        [ "$i" = "5" ] && exit 1; \
    done && \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libc++1-17 \
    libc++abi1-17 \
    libllvm17 \
    libpng16-16 \
    libjpeg-turbo8 \
    libglu1-mesa \
    gmsh \
    python3-gmsh \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Misuka's compiled build output (libmitsuba.so, libdrjit-extra.so, plugins/, etc.)
COPY --from=builder /opt/misuka/build /opt/misuka/build

# Python package and dependencies installed in the builder stage.
COPY --from=builder /opt/misuka_python /opt/misuka_python

ENV LD_LIBRARY_PATH="/opt/misuka/build"
ENV PYTHONPATH="/opt/misuka_python:/opt/misuka/build/python:/usr/lib/python3/dist-packages"
RUN ln -sf $(find /usr/lib -name "libLLVM-17.so" | head -1) /usr/local/lib/libLLVM-17.so
ENV DRJIT_LIBLLVM_PATH="/usr/local/lib/libLLVM-17.so"

WORKDIR /app/misuka_method
CMD ["python3.11", "-m", "misuka_interface"]
