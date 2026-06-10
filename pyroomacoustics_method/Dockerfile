# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Install system dependencies for mesh generation and scientific computing
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
    libgmsh-dev \
    python3-gmsh \
    && rm -rf /var/lib/apt/lists/*

# Make system python3-gmsh available to the pip environment
# This creates a symlink so pip-installed packages can import gmsh
RUN SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \
    ln -sf /usr/lib/python3/dist-packages/gmsh* "$SITE_PACKAGES/"

# Set working directory
WORKDIR /app

# Copy method package
COPY pyroomacoustics_method /app/pyroomacoustics_method

# Install the method package
RUN pip install --no-cache-dir /app/pyroomacoustics_method

WORKDIR /app/pyroomacoustics_method

# Call pyroomacoustics cli when the container starts
CMD ["python", "-m", "pyroomacoustics_interface"]
