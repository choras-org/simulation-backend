"""Exception hierarchy for CHORAS simulation methods.

This module defines a structured exception hierarchy that simulation methods
can use to provide meaningful, user-friendly error messages that will be
propagated to the frontend.
"""


class SimulationError(Exception):
    """Base exception for all simulation-related errors.

    This is the base class for all custom exceptions raised by simulation
    methods. It ensures that errors can be caught generically while still
    maintaining specific error types.

    Parameters
    ----------
    message : str
        A user-friendly description of what went wrong.

    """

    def __init__(self, message: str):
        """Initialize the simulation error.

        Parameters
        ----------
        message : str
            A user-friendly error message.

        """
        self.message = message
        super().__init__(self.message)


class ConfigurationError(SimulationError):
    """Exception raised for configuration-related errors.

    Use this exception when the simulation fails due to invalid configuration,
    missing parameters, invalid JSON structure, or incorrect settings.

    Examples
    --------
    >>> raise ConfigurationError(
    ...     f"Absorption coefficient format for {material_id} incorrect - "
    ...     "Check material assignments"
    ... )

    """

    pass


class GeometryError(SimulationError):
    """Exception raised for geometry-related errors.

    Use this exception when the simulation fails due to geometry issues
    such as missing mesh files, invalid geometry, malformed meshes, or
    geometry processing errors.

    Examples
    --------
    >>> raise GeometryError(
    ...     "The provided geometry is invalid - "
    ...     "Please verify the geometry file was uploaded correctly"
    ... )

    """

    pass


class ComputationError(SimulationError):
    """Exception raised for computation/solver errors.

    Use this exception when the simulation fails during the actual
    computation phase, such as solver divergence, numerical instability,
    or algorithm-specific failures.

    Examples
    --------
    >>> raise ComputationError(
    ...     "Solver diverged after 1000 iterations - "
    ...     "Try reducing time step or increasing damping"
    ... )

    """

    pass


class ResourceError(SimulationError):
    """Exception raised for resource-related errors.

    Use this exception when the simulation fails due to insufficient
    resources such as memory, disk space, file access permissions, or
    other system resource issues.

    Examples
    --------
    >>> raise ResourceError(
    ...     "Insufficient memory to allocate resources "
    ...     "Try reducing mesh density or use cloud resources"
    ... )

    """

    pass
