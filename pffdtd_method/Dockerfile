# CUDA devel base: needs nvcc at build time to compile Hamilton's c_cuda
# binary. At runtime the binary only requires the CUDA runtime libs which
# the same image ships. On hosts without GPU passthrough at runtime, the
# binary is never invoked -- the wrapper auto-falls-back to the pure-Python
# numba CPU engine. So this image works on every host CHORAS supports.
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Python 3.11 (Ubuntu 22.04 default is 3.10; PFFDTD method package floor is
# 3.11). libhdf5-dev needed to compile PFFDTD's c_cuda binary against HDF5.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        gmsh \
        libhdf5-dev \
        libglu1-mesa \
        libxcursor1 \
        libxinerama1 \
        curl \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Ubuntu apt-shipped pip (22.0.2) doesn't parse PEP 621 [project] tables.
# Upgrade before installing the method package or it registers as UNKNOWN
# with zero deps and the next pip run fails on missing imports.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone PFFDTD (Brian Hamilton, MIT). The repo has no setup.py / pyproject.toml;
# python/ subtree is added to PYTHONPATH. sed-fix the deprecated numpy
# aliases (np.float -> np.float64 was removed in numpy>=1.20; np.bool8 etc.
# in numpy>=2.0, hence we also pin numpy<2 in pyproject.toml).
ENV PFFDTD_REPO=https://github.com/bsxfun/pffdtd.git
ENV PFFDTD_REF=aa319f6
ENV PFFDTD_SRC=/opt/pffdtd_src

RUN git clone --depth 50 ${PFFDTD_REPO} ${PFFDTD_SRC} \
 && git -C ${PFFDTD_SRC} checkout ${PFFDTD_REF} \
 && ln -s ${PFFDTD_SRC}/python /app/pffdtd_python \
 && find ${PFFDTD_SRC}/python -name "*.py" -exec sed -i \
        -e 's/np\.finfo(np\.float)/np.finfo(np.float64)/g' \
        -e 's/np\.bool8/np.bool_/g' \
        -e 's/np\.complex_\b/np.complex128/g' \
        {} +

COPY pffdtd_method /app/pffdtd_method

# Install method package + verify PFFDTD python imports work
RUN pip install --no-cache-dir /app/pffdtd_method \
 && PYTHONPATH=/app/pffdtd_python python -c "from materials.adm_funcs import fit_to_Sabs_oct_11; from sim_setup import sim_setup; from fdtd.sim_fdtd import SimEngine; print('PFFDTD python imports OK')"

# Compile Hamilton's c_cuda GPU binary as a multi-arch CUDA fatbinary
# (sm_60 .. sm_90 + PTX for forward compat). Symlink fdtd_gpu -> the
# double-precision binary (works with raw impulse sources; single-precision
# binary asserts diff=true which doesn't apply here). On hosts without GPU
# passthrough the binary exists but is never invoked.
RUN cd ${PFFDTD_SRC}/c_cuda \
 && sed -i 's|-arch=sm_35|-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90|' Makefile \
 && make gpu \
 && ln -sf ${PFFDTD_SRC}/c_cuda/fdtd_main_gpu_double.x /usr/local/bin/fdtd_gpu

# PFFDTD's Python sources on PYTHONPATH so `python -c` debugging works inside
# the container even before the interface module's runtime sys.path insert.
ENV PYTHONPATH=/app/pffdtd_python

WORKDIR /app/pffdtd_method

CMD ["python", "-m", "pffdtd_interface"]
