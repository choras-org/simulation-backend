"""Fill numba's compilation cache by running one small simulation.

Run this once after installation (e.g. in the Dockerfile) so later
simulations skip the ~30 s JIT compilation:

    python -m sparrowpy_interface.warmup tests/test_input_sparrowpy.json

The input JSON's ``geo_path`` is resolved relative to the JSON file.
Everything runs on a temporary copy, so the input files are not modified
and no results are uploaded.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .sparrowpy_interface import sparrowpyMethod


def warmup(input_json_path: str | Path) -> None:
    """Run the simulation for ``input_json_path`` on a temporary copy."""
    input_json_path = Path(input_json_path).resolve()
    with open(input_json_path) as f:
        data = json.load(f)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # sparrowpy meshes the geo file itself, msh_path is not read
        geo = input_json_path.parent / data["geo_path"]
        shutil.copy(geo, tmp)
        data["geo_path"] = str(tmp / geo.name)
        data["msh_path"] = str(tmp / (geo.stem + ".msh"))
        tmp_json = tmp / "input.json"
        with open(tmp_json, "w") as f:
            json.dump(data, f)

        sparrowpyMethod(tmp_json).run_simulation()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m sparrowpy_interface.warmup INPUT_JSON")
    warmup(sys.argv[1])
    print("numba cache warmed up.")
