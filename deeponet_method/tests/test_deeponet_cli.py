"""Test the deeponet method CLI."""
import os
from pathlib import Path

import pytest

from deeponet_interface import main


def test_deeponet_method_cli(
    mock_requests_post, mock_simulation_stack, create_temporary_input_file
):
    """CLI plumbing: env var → DeepONetMethod → pipeline → results post."""
    os.environ["JSON_PATH"] = create_temporary_input_file
    main()

    mock_simulation_stack.assert_called_once()
    mock_requests_post.assert_called_once()


def test_deeponet_method_cli_does_not_mutate_user_json(
    mock_requests_post, mock_simulation_stack, create_temporary_input_file
):
    """The user-facing JSON must be treated as read-only by the CLI flow."""
    before = Path(create_temporary_input_file).read_bytes()

    os.environ["JSON_PATH"] = create_temporary_input_file
    main()

    after = Path(create_temporary_input_file).read_bytes()
    assert before == after, "main() mutated the user-facing JSON"


def test_deeponet_method_cli_missing_json_path(mock_requests_post):
    """Missing JSON_PATH must surface a FileNotFoundError before any work."""
    if "JSON_PATH" in os.environ:
        del os.environ["JSON_PATH"]

    with pytest.raises(FileNotFoundError, match="input_json_path cannot be None or empty"):
        main()
