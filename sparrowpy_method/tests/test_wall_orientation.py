"""Test the wall normals and up vectors returned by _import_room_geometry."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import trimesh

from sparrowpy_interface import sparrowpyMethod
from sparrowpy_interface.sparrowpy_interface import _import_room_geometry

from conftest import default_data_path

ROOMS = [
    ("test_input_sparrowpy.json", "test_room_sparrowpy.geo"),
    ("test_input_tilted.json", "test_room_tilted.geo"),
    ("MeasurementRoom_repaired.json", "MeasurementRoom_repaired.geo"),
    ("Room2215_simple_repaired.json", "Room2215_simple_repaired.geo"),
]


@pytest.fixture(params=ROOMS, ids=[geo[:-4] for _, geo in ROOMS])
def room_input_file(request, tmp_path):
    """Copy a room's input JSON and geo file into a temporary directory."""
    json_name, geo_name = request.param
    with open(os.path.join(default_data_path(), json_name)) as f:
        input_data = json.load(f)
    shutil.copy(os.path.join(default_data_path(), geo_name), tmp_path)
    input_data['geo_path'] = str(tmp_path / geo_name)
    input_data['msh_path'] = str(tmp_path / (geo_name[:-4] + ".msh"))
    json_path = Path(tmp_path) / "input.json"
    with open(json_path, 'w') as f:
        json.dump(input_data, f)
    return str(json_path)


def _geometry(json_path, patch_length=1):
    (walls_points, walls_normal, walls_up_vector, *_) = \
        _import_room_geometry(json_path, patch_length)
    return (np.asarray(walls_points, dtype=float),
            np.asarray(walls_normal, dtype=float),
            np.asarray(walls_up_vector, dtype=float))


@pytest.mark.parametrize("patch_length", [1, 3])
def test_normals_point_into_room(room_input_file, patch_length):
    walls_points, walls_normal, _ = _geometry(room_input_file, patch_length)
    n_walls = walls_points.shape[0]
    room = trimesh.Trimesh(
        walls_points.reshape(-1, 3), np.arange(3 * n_walls).reshape(-1, 3))
    room.merge_vertices()
    assert room.is_watertight

    npt.assert_allclose(np.linalg.norm(walls_normal, axis=1), 1, atol=1e-9)
    # perpendicular to the wall
    face_normal = np.cross(
        walls_points[:, 1] - walls_points[:, 0],
        walls_points[:, 2] - walls_points[:, 0])
    face_normal /= np.linalg.norm(face_normal, axis=1, keepdims=True)
    npt.assert_allclose(
        np.abs(np.einsum("ij,ij->i", face_normal, walls_normal)), 1,
        atol=1e-6)
    # a point slightly in front of each wall lies inside the room
    step = 1e-3 * np.linalg.norm(room.extents)
    inside = room.contains(walls_points.mean(axis=1) + step * walls_normal)
    assert np.all(inside), \
        f"{np.sum(~inside)} of {n_walls} normals point out of the room"


@pytest.mark.parametrize("patch_length", [1, 3])
def test_up_vectors_perpendicular_to_normals(room_input_file, patch_length):
    _, walls_normal, walls_up_vector = _geometry(
        room_input_file, patch_length)
    npt.assert_allclose(
        np.linalg.norm(walls_up_vector, axis=1), 1, atol=1e-9)
    npt.assert_allclose(
        np.einsum("ij,ij->i", walls_normal, walls_up_vector), 0, atol=1e-9)


def test_simulation_runs_in_tilted_room(create_temporary_input_file_tilted):
    sparrowpyMethod(create_temporary_input_file_tilted).run_simulation()

    with open(create_temporary_input_file_tilted) as f:
        data = json.load(f)
    rir = data['results'][0]['responses'][0]['receiverResults']
    assert np.all(np.isfinite(rir))
    assert np.any(np.asarray(rir) != 0)


def test_inward_normals_open_room_warns_and_points_to_center():
    from sparrowpy_interface.sparrowpy_interface import _inward_normals
    box = trimesh.creation.box(extents=[2, 2, 2])
    # remove the two faces of the top (z = 1), so the room is not closed
    keep = box.triangles_center[:, 2] < 0.99
    room = trimesh.Trimesh(box.vertices, box.faces[keep], process=False)
    surfaces = [[i] for i in range(len(room.faces))]
    with pytest.warns(UserWarning, match="not closed"):
        normals = _inward_normals(room, surfaces)
    to_center = -room.triangles_center
    assert np.all(np.einsum("ij,ij->i", normals, to_center) > 0)
