"""A relative ``msh_path`` is resolved against the JSON file's directory.

CHORAS writes absolute paths, but a hand-written JSON next to its mesh must
work from any working directory even though the wrapper ``chdir``s into the
package directory before running DEISM.
"""

import json
import os
import shutil
from pathlib import Path

from deism_interface.DEISMinterface import DeismMethod

FIXTURE_DIR = Path(__file__).parent


def test_relative_msh_path_resolves_against_json_dir(
    tmp_path, monkeypatch, default_input_data
):
    shutil.copy(FIXTURE_DIR / "test_room_Deism.msh", tmp_path)
    data = dict(default_input_data)
    data["msh_path"] = "test_room_Deism.msh"
    data["geo_path"] = "test_room_Deism.geo"
    json_path = tmp_path / "input.json"
    json_path.write_text(json.dumps(data))

    # Run from an unrelated working directory with an absolute JSON path.
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    DeismMethod(str(json_path)).run_simulation()

    result = json.loads(json_path.read_text())
    assert result["results"][0]["percentage"] == 100
    assert len(result["results"][0]["responses"][0]["receiverResults"]) > 0
    assert os.getcwd() == str(other_cwd)
