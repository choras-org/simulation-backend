"""Regenerate test_room_tilted.geo and test_input_tilted.json.

The room has an L-shaped floor (6 x 3 m plus 3 x 2 m) and a ceiling tilted
from 2.5 m at x=0 to 4.0 m at x=6 (volume 75.75 m^3, surface 118.74 m^2).
The .geo is produced by the backend's geometry pipeline, so run this with the
backend environment from this directory:

    ../../../backend/.venv/bin/python make_test_room_tilted.py
"""
import json
import re
import shutil
import tempfile
from pathlib import Path

from geometry_pipeline import process_geometry

here = Path(__file__).parent

floor = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
n = len(floor)
bot = [(x, y, 0.0) for x, y in floor]
top = [(x, y, 2.5 + 0.25 * x) for x, y in floor]
triangles = [
    # floor and ceiling as a fan from the reflex corner (index 3)
    (3, 2, 1), (3, 1, 0), (3, 0, 5), (3, 5, 4),
    (n+3, n+4, n+5), (n+3, n+5, n), (n+3, n, n+1), (n+3, n+1, n+2),
]
for i in range(n):
    j = (i + 1) % n
    triangles += [(i, j, n+j), (i, n+j, n+i)]

with tempfile.TemporaryDirectory() as tmp:
    obj = Path(tmp) / "test_room_tilted.obj"
    with open(obj, "w") as f:
        # OBJ is Y-up like a Fusion 360 export; the pipeline maps (x, y, z) -> (x, -z, y)
        for x, y, z in bot + top:
            f.write(f"v {x} {z} {-y}\n")
        for t in triangles:
            f.write("f %d %d %d\n" % tuple(k + 1 for k in t))
    process_geometry(obj, Path(tmp), volume_name="RoomVolume")
    shutil.copyfile(Path(tmp) / "test_room_tilted_repaired.geo", here / "test_room_tilted.geo")

names = re.findall(
    r'Physical Surface\("([^"]+)"\)', (here / "test_room_tilted.geo").read_text())
data = json.loads((here / "test_input_sparrowpy.json").read_text())
data["geo_path"] = "test_room_tilted.geo"
data["msh_path"] = "test_room_tilted.msh"
data["absorption_coefficients"] = {
    name: "0.05, 0.05, 0.08, 0.1, 0.12, 0.15" for name in names}
# the default receiver (5, 3.5) is outside the L
data["results"][0]["responses"][0]["x"] = 2.0
data["results"][0]["responses"][0]["y"] = 4.0
(here / "test_input_tilted.json").write_text(json.dumps(data, indent=4))
