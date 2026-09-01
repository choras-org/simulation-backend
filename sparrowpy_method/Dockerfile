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
COPY sparrowpy_method /app/sparrowpy_method

# Install the method package
RUN pip install --no-cache-dir /app/sparrowpy_method

WORKDIR /app/sparrowpy_method

# Default command to run the containerized sparrowpy method
CMD ["python", "-m", "sparrowpy_interface"]
