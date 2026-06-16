"""Test the SimulationMethod base class for misuka method."""
import pytest
from unittest.mock import patch
from pathlib import Path

from misuka_interface.definition import SimulationMethod


@patch.multiple(SimulationMethod, __abstractmethods__=set())
def test_simulation_method_with_valid_file(create_temporary_input_file):
    """Test SimulationMethod initialization with a valid file."""
    method = SimulationMethod(create_temporary_input_file)
    assert method.input_json_path == create_temporary_input_file


@pytest.mark.parametrize("empty_path", [None, ""])
@patch.multiple(SimulationMethod, __abstractmethods__=set())
def test_simulation_method_with_none_path(empty_path):
    """Test SimulationMethod initialization with None path."""
    with pytest.raises(FileNotFoundError, match="input_json_path cannot be None or empty"):
        SimulationMethod(empty_path)


@patch.multiple(SimulationMethod, __abstractmethods__=set())
def test_simulation_method_with_nonexistent_file():
    """Test SimulationMethod initialization with a non-existent file."""
    nonexistent_path = "/tmp/nonexistent_file_that_does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Input JSON file not found"):
        SimulationMethod(nonexistent_path)


@patch.multiple(SimulationMethod, __abstractmethods__=set())
def test_simulation_method_with_path_object(create_temporary_input_file):
    """Test SimulationMethod initialization with a Path object."""
    path_obj = Path(create_temporary_input_file)
    method = SimulationMethod(path_obj)
    assert method.input_json_path == path_obj
