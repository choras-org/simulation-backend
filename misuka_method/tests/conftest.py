"""Pytest configuration and fixtures."""
import sys
import json
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


def default_data_path():
    """Get the path to the default data folder."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)))


def load_default_input_data():
    """Load the example input data."""
    with open(os.path.join(
            default_data_path(),
            "test_input_misuka.json"), 'r') as f:
        data = json.load(f)

    return data


@pytest.fixture
def default_input_data():
    """Fixture to load the example input data."""
    return load_default_input_data()


@pytest.fixture
def create_temporary_input_file():
    """Fixture to create a temporary input JSON file which can be reused to
    write results to."""
    input_tmp = load_default_input_data()
    geo_file = os.path.join(
        default_data_path(), "test_room_misuka.geo")
    msh_file = os.path.join(
        default_data_path(), "test_room_misuka.msh")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname) / "temp_input.json"
        shutil.copy(geo_file, Path(tmpdirname))
        shutil.copy(msh_file, Path(tmpdirname))
        input_tmp['geo_path'] = os.path.join(
            tmpdirname, "test_room_misuka.geo")
        input_tmp['msh_path'] = os.path.join(
            tmpdirname, "test_room_misuka.msh")
        with open(tmp_path, 'w') as f:
            json.dump(input_tmp, f)

        yield str(tmp_path)

    return str(tmp_path)


@pytest.fixture
def mock_requests_post():
    """Fixture to mock requests.post for CLI tests.

    Returns the mock object so tests can make assertions on it.
    """
    with patch("misuka_interface.definition.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post
