"""Test the trimesh visibility against sparrowpy's own visibility check."""
import json

import numpy as np
import numpy.testing as npt
import pytest
import sparrowpy
import trimesh

from sparrowpy_interface.sparrowpy_interface import _import_room_geometry
from sparrowpy_interface.visibility import patch_visibility, point_visibility


@pytest.fixture(autouse=True, params=["embree", "rtree"])
def ray_backend(request, monkeypatch):
    """Run every test with embree and with trimesh's rtree fallback, which
    is used on platforms without embreex wheels (e.g. linux/arm64)."""
    if request.param == "embree":
        if not trimesh.ray.has_embree:
            pytest.skip("embreex is not installed")
    else:
        monkeypatch.setattr(trimesh.ray, "has_embree", False)
    return request.param


# sparrowpy's own check is slow, so compute it once for both ray backends
_SPARROWPY_REFERENCE = {}


def _sparrowpy_reference(key, compute):
    if key not in _SPARROWPY_REFERENCE:
        _SPARROWPY_REFERENCE[key] = compute()
    return _SPARROWPY_REFERENCE[key]


def _room_key(json_path, patch_length):
    with open(json_path) as f:
        geo_path = json.load(f)['geo_path']
    with open(geo_path) as f:
        return f.read(), patch_length


def _patches(json_path, patch_length):
    (_, walls_normal, _, patches_points, _, patch_to_wall_ids,
     *_) = _import_room_geometry(json_path, patch_length)
    return (np.asarray(patches_points, dtype=float),
            walls_normal[patch_to_wall_ids])


def _passes_through_edge(start, end, patches_points, tol=1e-9):
    """Whether the segment from ``start`` to ``end`` passes through an edge
    of any patch (closest distance between segments below ``tol``)."""
    p0 = patches_points.reshape(-1, 3)
    v = np.roll(patches_points, -1, axis=1).reshape(-1, 3) - p0
    u = end - start
    w = start - p0
    a, b, c = u @ u, v @ u, np.einsum("ij,ij->i", v, v)
    d, e = w @ u, np.einsum("ij,ij->i", v, w)
    denom = a * c - b**2
    s = np.clip(np.where(
        denom > 1e-12, (b * e - c * d) / np.where(denom > 1e-12, denom, 1),
        0), 0, 1)
    t = np.clip((b * s + e) / c, 0, 1)
    s = np.clip((b * t - d) / a, 0, 1)
    distance = np.linalg.norm(
        start + s[:, np.newaxis] * u - (p0 + t[:, np.newaxis] * v), axis=1)
    return np.min(distance) < tol


def _assert_matches(actual, expected, segment, patches_points, ray_backend):
    """Assert equal visibility. trimesh's rtree fallback misses rays passing
    exactly through an edge (e.g. the inner corner edge of the L-shaped
    room), which embree and sparrowpy count as blocked. Only these
    mismatches are allowed with the fallback."""
    mismatch = actual != expected
    if ray_backend == "rtree":
        for idx in zip(*np.nonzero(mismatch)):
            if actual[idx] and _passes_through_edge(
                    *segment(*idx), patches_points):
                mismatch[idx] = False
    assert not np.any(mismatch), \
        f"visibility differs from sparrowpy at {np.argwhere(mismatch)}"


def _compare_patch_visibility(json_path, patch_length, ray_backend):
    """Compare patch_visibility with sparrowpy and return it."""
    patches_points, patches_normal = _patches(json_path, patch_length)
    centers = patches_points.mean(axis=1)

    actual = patch_visibility(patches_points, patches_normal)
    expected = _sparrowpy_reference(
        ("patch", *_room_key(json_path, patch_length)),
        lambda: sparrowpy.geometry._check_patch2patch_visibility(
            centers, patches_normal, patches_points))
    _assert_matches(
        actual, expected, lambda i, j: (centers[i], centers[j]),
        patches_points, ray_backend)
    return actual


@pytest.mark.parametrize("patch_length", [3, 1])
def test_visibility_matches_sparrowpy_box(
        create_temporary_input_file, patch_length, ray_backend):
    _compare_patch_visibility(
        create_temporary_input_file, patch_length, ray_backend)


@pytest.mark.parametrize("patch_length", [3, 1])
def test_visibility_matches_sparrowpy_l_shaped(
        create_temporary_input_file_tilted, patch_length, ray_backend):
    """The L-shaped room has patches that are hidden behind a corner.

    With ``patch_length=3`` sparrowpy before #168 wrongly reported one pair
    as blocked: its point-in-polygon test counted a polygon vertex lying
    exactly on its test ray twice. See
    ``test_visibility_segment_level_with_vertex``.
    """
    actual = _compare_patch_visibility(
        create_temporary_input_file_tilted, patch_length, ray_backend)
    # facing each other but blocked by the corner
    assert np.any(~actual & _facing(create_temporary_input_file_tilted,
                                    patch_length))


def _facing(json_path, patch_length):
    (_, walls_normal, _, patches_points, n_patches, patch_to_wall_ids,
     *_) = _import_room_geometry(json_path, patch_length)
    normals = walls_normal[patch_to_wall_ids]
    centers = np.asarray(patches_points).mean(axis=1)
    d = centers[np.newaxis, :] - centers[:, np.newaxis]
    facing = (np.einsum("ik,ijk->ij", normals, d) > 1e-6) & \
        (np.einsum("jk,ijk->ij", normals, -d) > 1e-6)
    return np.triu(facing, 1)


def test_visibility_upper_triangle_only():
    # two parallel triangles facing each other
    points = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 1], [1, 0, 1]],
    ], dtype=float)
    normals = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    npt.assert_array_equal(
        patch_visibility(points, normals), [[False, True], [False, False]])


def test_visibility_blocked_by_triangle_in_between():
    points = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 2], [0, 1, 2], [1, 0, 2]],
        # large blocker at z=1 covering the segment between both patches
        [[-5, -5, 1], [10, -5, 1], [-5, 10, 1]],
    ], dtype=float)
    normals = np.array([[0, 0, 1], [0, 0, -1], [0, 0, -1]], dtype=float)
    visibility = patch_visibility(points, normals)
    assert not visibility[0, 1]
    assert visibility[0, 2]


def test_visibility_segment_level_with_vertex():
    """The segment crosses the blocker's plane level with one of its
    vertices, but outside of it. This is the case from the L-shaped room
    which sparrowpy gets wrong."""
    points = np.array([
        # source patch in the plane y=4/3, center (0.5, 4/3, 2.625)
        [[0, 4/3, 2.125], [1, 4/3, 2.875], [0.5, 4/3, 2.875]],
        # target patch on the wall y=5, center (0.5, 5, 1.25)
        [[0.5, 5, 0.75], [0.0, 5, 1.5], [1.0, 5, 1.5]],
        # blocker in the plane y=3 at x=4.5..6, vertex (6, 3, 2) is level
        # with the crossing point (0.5, 3, 2)
        [[4.5, 3, 1.625], [5.047, 3, 2.625], [6, 3, 2]],
    ], dtype=float)
    # the third normal does not matter for the pair (0, 1)
    normals = np.array([[0, 1, 0], [0, -1, 0], [0, -1, 0]], dtype=float)
    assert patch_visibility(points, normals)[0, 1]


def _compare_point_visibility(json_path, patch_length, points, ray_backend):
    """Compare point_visibility with sparrowpy and return it."""
    patches_points, patches_normal = _patches(json_path, patch_length)
    centers = patches_points.mean(axis=1)

    actual = point_visibility(points, patches_points, patches_normal)
    expected = _sparrowpy_reference(
        ("point", *_room_key(json_path, patch_length), points.tobytes()),
        lambda: np.array([
            sparrowpy.geometry._check_point2patch_visibility(
                p, centers, patches_normal, patches_points)
            for p in points]))
    _assert_matches(
        actual, expected, lambda p, j: (points[p], centers[j]),
        patches_points, ray_backend)
    return actual


def _source_and_receivers(json_path):
    with open(json_path) as f:
        result = json.load(f)['results'][0]
    points = [[result['sourceX'], result['sourceY'], result['sourceZ']]]
    points += [[r['x'], r['y'], r['z']] for r in result['responses']]
    return np.array(points, dtype=float)


@pytest.mark.parametrize("patch_length", [3, 1])
def test_point_visibility_matches_sparrowpy_box(
        create_temporary_input_file, patch_length, ray_backend):
    points = _source_and_receivers(create_temporary_input_file)
    _compare_point_visibility(
        create_temporary_input_file, patch_length, points, ray_backend)


@pytest.mark.parametrize("patch_length", [3, 1])
def test_point_visibility_matches_sparrowpy_l_shaped(
        create_temporary_input_file_tilted, patch_length, ray_backend):
    # source, receiver and two points in each leg of the L-shaped room
    points = np.vstack([
        _source_and_receivers(create_temporary_input_file_tilted),
        [[1.5, 4.0, 1.2], [4.5, 1.0, 1.7], [1.0, 1.0, 1.0]]])
    actual = _compare_point_visibility(
        create_temporary_input_file_tilted, patch_length, points, ray_backend)
    # some patches are hidden behind the corner of the L
    assert not np.all(actual)


def test_point_visibility_blocked_and_behind():
    points = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        # blocker at z=1 between the point and the patch
        [[-5, -5, 1], [10, -5, 1], [-5, 10, 1]],
    ], dtype=float)
    normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=float)
    visibility = point_visibility(
        np.array([[.3, .3, .5], [.3, .3, 2], [.3, .3, -1]]),
        points, normals)
    npt.assert_array_equal(visibility, [
        [True, False],   # in front of both, nothing in between
        [False, True],   # patch 0 hidden behind the blocker
        [False, False],  # behind both patches
    ])
