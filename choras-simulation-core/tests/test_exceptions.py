"""Tests for the custom exception classes."""

import pytest

from choras_simulation_core.exceptions import (
    ComputationError,
    ConfigurationError,
    GeometryError,
    ResourceError,
    SimulationError,
)


@pytest.mark.parametrize(
    ("exception_class", "error_message"),
    [
        (ConfigurationError, "Test configuration error"),
        (GeometryError, "Test geometry error"),
        (ComputationError, "Test computation error"),
        (ResourceError, "Test resource error"),
    ],
    ids=[
        "ConfigurationError",
        "GeometryError",
        "ComputationError",
        "ResourceError",
    ],
)
def test_exception_can_be_raised_and_caught(exception_class, error_message):
    """Test correctly raising the exception type and message."""
    with pytest.raises(exception_class, match=error_message):
        raise exception_class(error_message)


@pytest.mark.parametrize(
    "exception_class",
    [
        ConfigurationError,
        GeometryError,
        ComputationError,
        ResourceError,
    ],
    ids=[
        "ConfigurationError",
        "GeometryError",
        "ComputationError",
        "ResourceError",
    ],
)
def test_exception_hierarchy(exception_class):
    """Test that all custom exceptions inherit from SimulationError."""
    assert issubclass(exception_class, SimulationError)


def test_exception_message_attribute():
    """Test that SimulationError stores the message attribute."""
    error = ConfigurationError("Test message")
    assert error.message == "Test message"
    assert str(error) == "Test message"
