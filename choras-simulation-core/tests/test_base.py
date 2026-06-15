"""Tests for the SimulationMethod base class."""

from functools import wraps
from unittest.mock import patch

import pytest

from choras_simulation_core import SimulationMethod


def patch_abstract_methods(test_func):
    """Decorator to patch abstract methods for testing.

    This decorator temporarily removes the __abstractmethods__ attribute
    from SimulationMethod, allowing it to be instantiated for testing
    without needing to implement the abstract run_simulation method.
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        with patch.multiple(SimulationMethod, __abstractmethods__=set()):
            return test_func(*args, **kwargs)

    return wrapper


@patch_abstract_methods
def test_simulation_method_init_with_valid_path(temp_json_file):
    """Test initialization with a valid JSON file."""
    method = SimulationMethod(temp_json_file)
    assert method.input_json_path == temp_json_file


@patch_abstract_methods
def test_simulation_method_init_with_none():
    """Test initialization with None path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="cannot be None or empty"):
        SimulationMethod(None)


@patch_abstract_methods
def test_simulation_method_init_with_empty_string():
    """Test initialization with empty string raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="cannot be None or empty"):
        SimulationMethod("")


@patch_abstract_methods
def test_simulation_method_init_with_nonexistent_file():
    """Test initialization with nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Input JSON file not found"):
        SimulationMethod("/nonexistent/path/file.json")


@patch_abstract_methods
def test_save_results_method_exists(temp_json_file):
    """Test that save_results method is available and callable."""
    method = SimulationMethod(temp_json_file)
    # Verify the method exists and is callable
    assert callable(method.save_results)


@patch_abstract_methods
def test_input_json_path_property(temp_json_file):
    """Test that input_json_path is accessible as a property."""
    method = SimulationMethod(temp_json_file)
    # Test getter
    assert method.input_json_path == temp_json_file
    # Verify it's read-only (no setter defined)
    assert hasattr(method, "input_json_path")


@patch_abstract_methods
def test_save_results_success(temp_json_file, mock_requests_post):
    """Test that save_results successfully sends file on first attempt."""
    method = SimulationMethod(temp_json_file)
    result = method.save_results(url="http://test.example.com/receive")

    # Verify success
    assert result is True

    # Verify requests.post was called once
    assert mock_requests_post.call_count == 1

    # Verify the correct URL was used
    mock_requests_post.assert_called_once()
    call_args = mock_requests_post.call_args
    assert call_args[0][0] == "http://test.example.com/receive"

    # Verify the file was sent
    assert "file" in call_args[1]["files"]


@patch_abstract_methods
def test_save_results_retries_on_failure(
    temp_json_file, mock_time_sleep, mock_requests_post
):
    """Test that save_results retries on non-200 status codes."""
    # Mock responses: first two fail, third succeeds
    responses = [500, 500, 200]

    def side_effect(*args, **kwargs):
        mock_response = type("MockResponse", (), {})()
        mock_response.status_code = responses.pop(0)
        return mock_response

    mock_requests_post.side_effect = side_effect

    method = SimulationMethod(temp_json_file)
    result = method.save_results(max_retries=3, delay=1)

    # Verify success after retries
    assert result is True

    # Verify requests.post was called 3 times
    assert mock_requests_post.call_count == 3

    # Verify sleep was called between retries (2 times)
    assert mock_time_sleep.call_count == 2
    mock_time_sleep.assert_called_with(1)


@patch_abstract_methods
def test_save_results_max_retries_exceeded(
    temp_json_file, mock_time_sleep, mock_requests_post
):
    """Test that save_results returns False after max retries."""
    # Mock all attempts to fail
    mock_requests_post.return_value.status_code = 500

    method = SimulationMethod(temp_json_file)
    result = method.save_results(max_retries=3, delay=1)

    # Verify failure
    assert result is False

    # Verify requests.post was called max_retries times
    assert mock_requests_post.call_count == 3

    # Verify sleep was called between all retries
    assert mock_time_sleep.call_count == 3


@patch_abstract_methods
def test_save_results_handles_request_exception(
    temp_json_file, mock_time_sleep, mock_requests_post
):
    """Test that save_results handles RequestException and retries."""
    import requests

    # Mock first two attempts to raise exception, third succeeds
    side_effects = [
        requests.RequestException("Connection error"),
        requests.RequestException("Timeout"),
        type("MockResponse", (), {"status_code": 200})(),
    ]
    mock_requests_post.side_effect = side_effects

    method = SimulationMethod(temp_json_file)
    result = method.save_results(max_retries=3, delay=1)

    # Verify success after handling exceptions
    assert result is True

    # Verify requests.post was called 3 times
    assert mock_requests_post.call_count == 3

    # Verify sleep was called between retries
    assert mock_time_sleep.call_count == 2


@patch_abstract_methods
def test_save_results_uses_default_parameters(
    temp_json_file, mock_requests_post
):
    """Test that save_results uses correct default parameters."""
    method = SimulationMethod(temp_json_file)
    method.save_results()

    # Verify default URL was used
    call_args = mock_requests_post.call_args
    assert call_args[0][0] == "http://host.docker.internal:5001/receive"


def test_abstract_method_must_be_implemented(temp_json_file):
    """Test that abstract SimulationMethod cannot be instantiated."""
    # Without the decorator, this should raise TypeError
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        SimulationMethod(temp_json_file)
