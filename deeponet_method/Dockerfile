FROM python:3.11.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

WORKDIR /app

# System dependencies for mesh generation and scientific Python builds
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
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

# for TensorBoard
EXPOSE 6006

# Default command to run the containerized deeponet method
CMD ["python", "-m", "deeponet_interface"]
