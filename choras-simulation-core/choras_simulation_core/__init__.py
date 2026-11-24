"""CHORAS Simulation Core Package.

This package provides shared base classes and utilities for all CHORAS
simulation methods. It defines the common interface that all simulation
methods must implement and provides structured exception types for
meaningful error reporting.

"""

from choras_simulation_core.base import SimulationMethod
from choras_simulation_core import exceptions

__version__ = "0.1.0"

__all__ = [
    "SimulationMethod",
    "exceptions",
]
