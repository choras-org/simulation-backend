# Use Python 3.11 as base image to match the main backend
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing (gmsh is no longer
# needed: deism>=2.2.1.16 reads the pre-generated .msh via meshio)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy method package directory
COPY deism_method /app/deism_method

# Install the method package
RUN pip install --no-cache-dir /app/deism_method

WORKDIR /app/deism_method

# Default command to run the containerized DEISM method
CMD ["python", "-m", "deism_interface"]
