"""Test the sparrowpy method CLI."""
import os
import json
import pytest

from sparrowpy_interface import main, sparrowpyMethod


def test_sparrowpy_method_cli(mock_requests_post, create_temporary_input_file):
    """Test the sparrowpy method CLI."""
    # Set JSON_PATH environment variable and call main() directly
    os.environ["JSON_PATH"] = create_temporary_input_file
    main()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)
    
    # check that results were written to the JSON file
    assert "receiverResults" in data['results'][0]['responses'][0]
    results = data['results'][0]['responses'][0]['receiverResults']
    assert results is not None
    assert len(results) > 0

    # Verify that requests.post was called (save_results was executed)
    mock_requests_post.assert_called_once()


def test_sparrowpy_method_cli_missing_json_path(mock_requests_post):
    """Test the sparrowpy method CLI with missing JSON_PATH."""
    # Clear JSON_PATH environment variable
    if "JSON_PATH" in os.environ:
        del os.environ["JSON_PATH"]

    # Expect FileNotFoundError from SimulationMethod.__init__
    with pytest.raises(FileNotFoundError, match="input_json_path cannot be None or empty"):
        main()
