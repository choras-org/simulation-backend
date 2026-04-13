# Use Python 3.11 as base image to match the main backend
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for mesh generation and scientific computing
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY deeponet_method/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the DE method interface and shared components
COPY deeponet_method/DeepONetInterface.py .
COPY dg_method/DGinterface.py .
COPY common/MeasurementRoom.geo .
COPY common/MeasurementRoom.msh .
COPY common/HelperFunctions.py .
COPY common/simulation_method_interface.py .


# Default command to run the containerized DE method
CMD ["python", "DeepONetInterface.py"]