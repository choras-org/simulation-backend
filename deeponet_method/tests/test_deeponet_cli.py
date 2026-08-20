"""Test the deeponet method CLI."""
import json
import os
from pathlib import Path

from deeponet_interface import main
from deeponet_interface.deeponet_interface import _write_results_json


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
    """In headless mode the user-facing JSON is read-only."""
    before = Path(create_temporary_input_file).read_bytes()

    os.environ["JSON_PATH"] = create_temporary_input_file
    os.environ["HEADLESS"] = "true"
    try:
        main()
    finally:
        del os.environ["HEADLESS"]

    after = Path(create_temporary_input_file).read_bytes()
    assert before == after, "main() mutated the user-facing JSON in headless mode"


def test_write_results_json_overwrites_source_in_place(
    create_temporary_input_file, tmp_path, monkeypatch
):
    """CHORAS flow: when source == output path, results are merged into the user JSON."""
    train_cfg = {"id": "test_model", "output_dir": str(tmp_path)}
    receivers_dir = tmp_path / "test_model" / "figs" / "receivers"
    receivers_dir.mkdir(parents=True)

    # Empty placeholder; filename carries the positions, contents are mocked away.
    (receivers_dir / "0_x0=['1.00', '2.00', '1.50']_r0=['5.00', '3.50', '1.50']_pred.wav").touch()
    monkeypatch.setattr(
        "deeponet_interface.deeponet_interface._read_wav_impulse_response",
        lambda _path: [0.0],
    )

    with open(create_temporary_input_file) as f:
        before = json.load(f)

    _write_results_json(create_temporary_input_file, train_cfg, create_temporary_input_file)

    with open(create_temporary_input_file) as f:
        after = json.load(f)

    assert after["results"][0]["sourceX"] == 1.0
    assert after["results"][0]["sourceY"] == 2.0
    assert after["results"][0]["sourceZ"] == 1.5
    assert after["results"][0]["resultType"] == "DON"
    assert after["results"][0]["responses"][0]["x"] == 5.0

    # Non-results fields are preserved
    assert after["geo_path"] == before["geo_path"]
    assert after["msh_path"] == before["msh_path"]
    assert after["simulationSettings"] == before["simulationSettings"]
