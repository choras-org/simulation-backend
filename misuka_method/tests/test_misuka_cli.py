"""Test the misuka method CLI."""
import os
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
import pyfar

from misuka_interface import main, misukaMethod


class TestMisukaIntegration:
    """Test suite for misuka integration."""

    def test_misuka_method_initialization(self, create_temporary_input_file):
        """Test misukaMethod initialization."""
        method = misukaMethod(input_json_path=create_temporary_input_file)
        
        assert method is not None
        assert method.input_json_path == create_temporary_input_file
        assert hasattr(method, 'run_simulation')
        assert hasattr(method, '_misuka_method')

    def test_json_loading_with_absorption_coefficients(self, create_temporary_input_file):
        """Test loading JSON with absorption coefficients."""
        with open(create_temporary_input_file, 'r') as f:
            data = json.load(f)
        
        # Verify required fields exist
        assert "simulationSettings" in data
        assert "results" in data
        assert "msh_path" in data
        
        # Verify absorption coefficients can be accessed
        result = data["results"][0]
        assert "sourceX" in result
        assert "sourceY" in result
        assert "sourceZ" in result
        assert "frequencies" in result


class TestMisukaOutput:
    """Test output generation from misuka simulations."""

    def test_receiver_results_structure(self, create_temporary_input_file):
        """Test that receiver results have correct structure."""
        with open(create_temporary_input_file, 'r') as f:
            data = json.load(f)
        
        response = data["results"][0]["responses"][0]
        
        # Should have receiverResults key
        assert "receiverResults" in response

class TestMisukaRunthrough:
    """Test the full run-through of the misuka method."""

    def test_full_simulation_run(self, create_temporary_input_file):

        print("pyfar file:", pyfar.__file__)

        print("pyfar version:", pyfar.__version__)

        print("has constants:", hasattr(pyfar, "constants"))

        print("dir contains constants:", "constants" in dir(pyfar))
        """Test running the full simulation."""
        method = misukaMethod(input_json_path=create_temporary_input_file)
        
        # Run the simulation
        method.run_simulation()
        
        # Check that results are populated
        with open(create_temporary_input_file, 'r') as f:
            data = json.load(f)
        
        assert "results" in data
        assert len(data["results"]) > 0
        assert "responses" in data["results"][0]
