FROM python:3.11.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

WORKDIR /app

# System dependencies for mesh generation and scientific Python builds.
# python3-gmsh provides the gmsh Python module (+ libgmsh) on platforms with no
# gmsh pip wheel (linux aarch64 / Apple Silicon). Its dist-packages dir holds
# only gmsh.py, so exposing it via PYTHONPATH (below) cannot shadow venv deps.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
    python3-gmsh \
    && rm -rf /var/lib/apt/lists/*

# Copy both packages so `[tool.uv.sources] dg_interface = { path = "../dg_method" }` resolves
COPY dg_method /app/dg_method
COPY deeponet_method /app/deeponet_method

WORKDIR /app/deeponet_method

# Avoid hardlink warnings on Docker overlay filesystems
ENV UV_LINK_MODE=copy

# Install from uv.lock (honors [tool.uv.sources] and override-dependencies)
RUN uv sync --frozen --no-dev

# Use the project venv as the default Python
ENV PATH="/app/deeponet_method/.venv/bin:$PATH"

# Let the venv python import the apt-provided gmsh (dist-packages holds only
# gmsh.py). Needed on linux aarch64 where no gmsh pip wheel exists; harmless on
# x86_64 (same gmsh 4.13.1).
ENV PYTHONPATH="/usr/lib/python3/dist-packages"

# for TensorBoard
EXPOSE 6006

# Default command to run the containerized deeponet method
CMD ["python", "-m", "deeponet_interface"]
