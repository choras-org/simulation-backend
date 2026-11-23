"""Regression tests for the receiver interpolation in edg-acoustics.

Up to edg-acoustics 08cac98, ``AcousticsSimulation.sample3D`` evaluated the
orthonormal simplex basis at physical coordinates. Its collapsed coordinates
are singular on z = 1 and y + z = 0, so receivers at exactly z = 1 m returned
wrong pressures and receivers with y + z = 0 raised "Singular matrix".
"""
from pathlib import Path

import edg_acoustics
import gmsh
import numpy as np
import pytest

MESH_FILE = Path(__file__).parent / "test_room_edg_acoustics.msh"


def _smooth_field(x):
    return np.cos(2.1 * x[0]) * np.sin(1.3 * x[1] + 0.4) * np.cos(0.9 * x[2])


@pytest.fixture(scope="module")
def simulation():
    gmsh.initialize()
    gmsh.open(str(MESH_FILE))
    bc_labels = {
        gmsh.model.getPhysicalName(dim, tag) or f"surface_{tag}": tag
        for dim, tag in gmsh.model.getPhysicalGroups(2)
    }
    gmsh.finalize()
    mesh = edg_acoustics.Mesh(str(MESH_FILE), bc_labels)
    return edg_acoustics.AcousticsSimulation(1.21, 343.0, 4, mesh, bc_labels)


@pytest.mark.parametrize(
    "receiver",
    [
        (2.0, 1.0, 1.0),  # z = 1 exactly: used to be off by orders of magnitude
        (3.0, 2.0, 1.0),
        (2.0, 1.5, 1.0),
        (1.0, 1.0, 1.5),
        (2.03, 1.07, 1.11),
    ],
)
def test_receiver_interpolation_matches_smooth_field(simulation, receiver):
    rec = np.array(receiver, dtype=float)[:, np.newaxis]
    simulation.init_rec(rec, "brute_force")

    nodes = simulation.xyz[:, :, simulation.nodeindex[0]]
    interpolated = simulation.sampleWeight[0] @ _smooth_field(nodes)

    assert simulation.nodeindex[0] >= 0
    assert np.max(np.abs(simulation.sampleWeight)) < 10
    # The test mesh is coarse (elements > 1 m), so allow for the regular
    # degree-4 interpolation error; the bug gave errors of order 1-100.
    assert interpolated == pytest.approx(_smooth_field(rec)[0], abs=1e-2)


def test_receiver_at_y_plus_z_zero_does_not_crash(simulation):
    rec = np.array([[1.0], [0.0], [0.0]])
    simulation.init_rec(rec, "brute_force")
    assert np.all(np.isfinite(simulation.sampleWeight))


def test_receiver_outside_mesh_raises(simulation):
    rec = np.array([[10.0], [10.0], [10.0]])
    with pytest.raises(ValueError, match="not found inside the mesh"):
        simulation.init_rec(rec, "brute_force")
