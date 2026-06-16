"""Test the misuka method CLI."""
import os
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from misuka_interface import main, misukaMethod


class TestMisukaIntegration:
    """Test suite for misuka integration."""

    def test_parse_absorption_from_json(self):
        """Test parsing absorption coefficients from JSON."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 500, 20000]
        absorption_coefficients = {
            "floor": "0.1, 0.5, 0.1",
            "wall": "0.2, 0.4, 0.3",
            "ceiling": "0.15, 0.25, 0.35"
        }
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        assert len(surfaces) == 3
        assert surfaces == ["floor", "wall", "ceiling"]
        assert len(absorption_values) == 3
        
        # Check floor absorption values
        assert absorption_values[0] == [(100, 0.1), (500, 0.5), (20000, 0.1)]
        
        # Check wall absorption values
        assert absorption_values[1] == [(100, 0.2), (500, 0.4), (20000, 0.3)]

    def test_convert_msh_to_obj(self, tmp_path):
        """Test MSH to OBJ conversion method exists."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        assert hasattr(method, 'convert_msh_to_obj')
        assert callable(method.convert_msh_to_obj)

    def test_misuka_method_initialization(self, create_temporary_input_file):
        """Test misukaMethod initialization."""
        method = misukaMethod(input_json_path=create_temporary_input_file)
        
        assert method is not None
        assert method.input_json_path == create_temporary_input_file
        assert hasattr(method, 'run_simulation')
        assert hasattr(method, '_misuka_method')
        assert hasattr(method, 'parse_absorption_from_json')

    def test_absorption_parsing_with_whitespace(self):
        """Test absorption parsing handles whitespace correctly."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 500, 1000]
        absorption_coefficients = {
            "room": "0.1 , 0.5 , 0.3"  # Extra spaces
        }
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        assert absorption_values[0] == [(100, 0.1), (500, 0.5), (1000, 0.3)]

    def test_absorption_parsing_empty_dict(self):
        """Test absorption parsing with empty dictionary."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 500]
        absorption_coefficients = {}
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        assert len(surfaces) == 0
        assert len(absorption_values) == 0

    def test_absorption_values_all_numeric(self):
        """Test that parsed absorption values are numeric."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 250, 500, 1000, 2000]
        absorption_coefficients = {
            "floor": "0.6, 0.69, 0.71, 0.7, 0.63",
            "wall": "0.5, 0.55, 0.60, 0.62, 0.61"
        }
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        # Verify all values are floats
        for surface_absorptions in absorption_values:
            for freq, absorption in surface_absorptions:
                assert isinstance(freq, (int, float))
                assert isinstance(absorption, float)
                assert 0 <= absorption <= 1.0

    def test_misuka_method_has_required_methods(self):
        """Test that misukaMethod has all required methods."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        required_methods = [
            'run_simulation',
            '_misuka_method',
            'parse_absorption_from_json',
            'convert_msh_to_obj'
        ]
        
        for method_name in required_methods:
            assert hasattr(method, method_name), f"Missing method: {method_name}"
            assert callable(getattr(method, method_name))

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

    def test_absorption_coefficient_format_conversion(self):
        """Test conversion of absorption format matches Mitsuba spec."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 500, 20000]
        absorption_coefficients = {
            "floor": "0.1, 0.5, 0.1"
        }
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        # Should match Mitsuba format: [(freq, value), ...]
        expected_format = [(100, 0.1), (500, 0.5), (20000, 0.1)]
        assert absorption_values[0] == expected_format

    def test_multiple_surfaces_parsing(self):
        """Test parsing multiple surfaces with different absorption profiles."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 500, 1000, 2000, 5000, 10000]
        absorption_coefficients = {
            "highly_absorbent": "0.9, 0.85, 0.8, 0.75, 0.7, 0.65",
            "slightly_absorbent": "0.1, 0.12, 0.15, 0.18, 0.2, 0.22",
            "neutral": "0.5, 0.5, 0.5, 0.5, 0.5, 0.5"
        }
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        assert len(surfaces) == 3
        assert len(absorption_values[0]) == 6
        
        # Verify highly absorbent has high values
        for freq, absorption in absorption_values[0]:
            assert absorption >= 0.65
        
        # Verify slightly absorbent has low values
        for freq, absorption in absorption_values[1]:
            assert absorption <= 0.22

    def test_frequency_absorption_tuple_structure(self):
        """Test that output tuples have correct structure."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [125, 250, 500]
        absorption_coefficients = {"wall": "0.2, 0.3, 0.4"}
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        absorption_array = absorption_values[0]
        
        # Each element should be a tuple of (freq, absorption)
        assert len(absorption_array) == 3
        for item in absorption_array:
            assert isinstance(item, tuple)
            assert len(item) == 2
            freq, absorption = item
            assert freq in frequencies
            assert 0 <= absorption <= 1.0

    def test_absorption_value_range(self):
        """Test that absorption values are within valid range [0, 1]."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 500, 20000]
        absorption_coefficients = {
            "normal": "0.0, 0.5, 1.0",
            "mid_range": "0.25, 0.5, 0.75"
        }
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        for surface_absorptions in absorption_values:
            for freq, absorption in surface_absorptions:
                assert 0.0 <= absorption <= 1.0, \
                    f"Absorption value {absorption} outside valid range [0, 1]"

    def test_surface_name_preservation(self):
        """Test that surface names are preserved correctly."""
        with patch.object(misukaMethod.__bases__[0], '__init__', lambda x, y: None):
            method = misukaMethod()
        
        frequencies = [100, 500]
        surface_names = ["floor", "wall1", "wall2", "ceiling", "back_wall"]
        absorption_coefficients = {
            name: "0.2, 0.3" for name in surface_names
        }
        
        surfaces, absorption_values = method.parse_absorption_from_json(
            frequencies, 
            absorption_coefficients
        )
        
        # Note: dict order depends on Python version, but all should be present
        assert set(surfaces) == set(surface_names)


class TestMisukaSceneGeneration:
    """Test scene generation for Mitsuba."""

    def test_scene_dict_structure(self, create_temporary_input_file):
        """Test that scene dictionary has correct structure."""
        with open(create_temporary_input_file, 'r') as f:
            data = json.load(f)
        
        # Should have required keys
        assert "simulationSettings" in data
        assert "results" in data
        assert data["simulationSettings"]["speed_of_sound"] == 343
        assert data["simulationSettings"]["max_time"] == 0.1

    def test_source_receiver_coordinates(self, create_temporary_input_file):
        """Test source and receiver coordinate extraction."""
        with open(create_temporary_input_file, 'r') as f:
            data = json.load(f)
        
        result = data["results"][0]
        
        source_coords = [result["sourceX"], result["sourceY"], result["sourceZ"]]
        assert len(source_coords) == 3
        assert all(isinstance(c, (int, float)) for c in source_coords)
        
        response = result["responses"][0]
        receiver_coords = [response["x"], response["y"], response["z"]]
        assert len(receiver_coords) == 3
        assert all(isinstance(c, (int, float)) for c in receiver_coords)


class TestMisukaOutput:
    """Test output generation from misuka simulations."""

    def test_receiver_results_structure(self, create_temporary_input_file):
        """Test that receiver results have correct structure."""
        with open(create_temporary_input_file, 'r') as f:
            data = json.load(f)
        
        response = data["results"][0]["responses"][0]
        
        # Should have receiverResults key
        assert "receiverResults" in response


def test_misuka_method_cli(mock_requests_post, create_temporary_input_file):
    """Test the misuka method CLI."""
    # Set JSON_PATH environment variable and call main() directly
    os.environ["JSON_PATH"] = create_temporary_input_file
    main()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)

    # Verify results structure
    assert "results" in data
    assert len(data["results"]) > 0
    
    # Verify that requests.post was called (save_results was executed)
    mock_requests_post.assert_called_once()


def test_misuka_method_cli_missing_json_path(mock_requests_post):
    """Test the misuka method CLI with missing JSON_PATH."""
    # Clear JSON_PATH environment variable
    if "JSON_PATH" in os.environ:
        del os.environ["JSON_PATH"]

    # Expect FileNotFoundError from SimulationMethod.__init__
    with pytest.raises(FileNotFoundError, match="input_json_path cannot be None or empty"):
        main()
