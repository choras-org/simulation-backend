# Use Python 3.11 as base image to match the main backend
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

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

# Copy method package directory
COPY de_method /app/de_method

# Install the method package
RUN pip install --no-cache-dir /app/de_method

WORKDIR /app/de_method

# Default command to run the containerized DE method
CMD ["python", "-m", "de_interface"]
