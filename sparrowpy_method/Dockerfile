FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for mesh generation and scientific computing
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
    && rm -rf /var/lib/apt/lists/*

# Copy method package directory
COPY sparrowpy_method /app/sparrowpy_method

# Install the method package
RUN pip install --no-cache-dir /app/sparrowpy_method

WORKDIR /app/sparrowpy_method

# Run one small simulation to fill numba's compilation cache, so
# simulations in the container skip the ~30 s JIT compilation. The cache
# is compiled for the build host's CPU; on a different CPU numba ignores
# it and recompiles. Set NUMBA_CPU_NAME=generic if the image is built on
# one machine and run on another.
RUN python -m sparrowpy_interface.warmup tests/test_input_sparrowpy.json

# Default command to run the containerized sparrowpy method
CMD ["python", "-m", "sparrowpy_interface"]
