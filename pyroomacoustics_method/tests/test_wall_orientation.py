import numpy as np
import pytest
from pyroomacoustics_interface.pyroomacoustics_interface import orient_triangles_outward


def _cube_triangles():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([v[[a, b, c]] for a, b, c, d in quads] + [v[[a, c, d]] for a, b, c, d in quads])


def _volume(tris):
    return np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0


@pytest.mark.parametrize("flip", [[], [0, 3, 7], list(range(12))])
def test_orientation_is_made_consistent_and_outward(flip):
    tris = _cube_triangles()
    tris[flip] = tris[flip][:, ::-1]
    oriented = orient_triangles_outward(tris)
    assert _volume(oriented) == pytest.approx(1.0)
    centre = np.array([0.5, 0.5, 0.5])
    normals = np.cross(oriented[:, 1] - oriented[:, 0], oriented[:, 2] - oriented[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, oriented.mean(axis=1) - centre) > 0)


def test_open_mesh_is_left_unchanged():
    tris = _cube_triangles()[:-1]
    with pytest.warns(UserWarning):
        assert np.array_equal(orient_triangles_outward(tris), tris)
