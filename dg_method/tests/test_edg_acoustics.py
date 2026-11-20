import os
import re
import gmsh
import json
import numpy as np
from pathlib import Path

import edg_acoustics
from acousticDE.FiniteVolumeMethod.CreateMeshFVM import generate_mesh

from dg_interface import DGMethod
from dg_interface.DGinterface import surface_materials


def test_edg_acoustics(create_temporary_input_file):
    """
    Test the DG acoustic simulation method.
    """
    gmsh.initialize()
    interface = DGMethod(create_temporary_input_file)
    interface.run_simulation()
    gmsh.finalize()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)

    rir = np.array(data['results'][0]['responses'][0]['receiverResults'])

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)


def test_surface_materials_maps_physical_groups_by_name(tmp_path):
    """Boundary labels use the physical group tags, not the surface order."""
    geo = (Path(__file__).parent / "test_room_edg_acoustics.geo").read_text()
    geo = re.sub(r'Physical Surface\("[^"]+"\) = \{ \d+ \};\n', "", geo)
    geo = geo.replace(
        "Physical Volume",
        'Physical Surface("floor", 11) = { 1 };\n'
        'Physical Surface("walls", 12) = { 2, 4, 5, 6 };\n'
        'Physical Surface("ceiling", 13) = { 3 };\n'
        "Physical Volume",
    )
    geo_path = tmp_path / "room.geo"
    geo_path.write_text(geo)
    msh_path = tmp_path / "room.msh"
    result_container = {
        "absorption_coefficients": {
            "floor": "0.1, 0.1",
            "walls": "0.2, 0.2",
            "ceiling": "0.3, 0.3",
        },
        "simulationSettings": {},
    }

    gmsh.initialize()
    generate_mesh(str(geo_path), str(msh_path), 1)
    BC_labels, absorption, _, _ = surface_materials(result_container, 343)
    gmsh.finalize()

    assert BC_labels == {"floor": 11, "walls": 12, "ceiling": 13}
    assert absorption == {"floor": [0.1, 0.1], "walls": [0.2, 0.2], "ceiling": [0.3, 0.3]}
    edg_acoustics.Mesh(str(msh_path), BC_labels)
