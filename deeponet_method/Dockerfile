FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for mesh generation and scientific computing
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
    && rm -rf /var/lib/apt/lists/*

# Copy method package directory
COPY deeponet_method /app/deeponet_method

# Install the method package
RUN pip install --no-cache-dir /app/deeponet_method

WORKDIR /app/deeponet_method

# Default command to run the containerized deeponet method
CMD ["python", "-m", "deeponet_interface"]
