"""Test fixtures for MoDART method tests."""
import pytest


def test_default_input_data_structure(default_input_data):
    """Test that the default input data has the expected structure."""
    assert "results" in default_input_data
    assert len(default_input_data["results"]) > 0
    for i in range(len(default_input_data["results"])):
        assert "sourceX" in default_input_data["results"][i]
        assert "sourceY" in default_input_data["results"][i]
        assert "sourceZ" in default_input_data["results"][i]
        assert "responses" in default_input_data["results"][i]
        assert len(default_input_data["results"][i]["responses"]) > 0
    assert "geo_path" in default_input_data
    assert "absorption_coefficients" in default_input_data


def test_create_temporary_input_file_fixture(create_temporary_input_file):
    """Test that the temporary input file fixture works correctly."""
    import os
    assert os.path.exists(create_temporary_input_file)
    assert create_temporary_input_file.endswith(".json")
