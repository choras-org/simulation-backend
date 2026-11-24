"""Fixtures for testing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_json_file():
    """Fixture to create a temporary JSON file for testing.

    Yields the path to the temporary file and cleans it up after the test.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump({"test": "data"}, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_requests_post():
    """Fixture to mock requests.post for testing save_results method.

    Returns the mock object so tests can make assertions on it.
    """
    with patch("choras_simulation_core.base.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_time_sleep():
    """Fixture to mock time.sleep to speed up tests.

    Returns the mock object so tests can make assertions on it.
    """
    with patch("choras_simulation_core.base.time.sleep") as mock_sleep:
        yield mock_sleep
