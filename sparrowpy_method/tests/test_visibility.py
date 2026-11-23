"""Test the trimesh visibility against sparrowpy's own visibility check."""
import json

import numpy as np
import numpy.testing as npt
import pytest
import sparrowpy

from sparrowpy_interface.sparrowpy_interface import _import_room_geometry
from sparrowpy_interface.visibility import patch_visibility, point_visibility


def _visibility_both_ways(json_path, patch_length):
    (walls_points, walls_normal, _, patches_points, _, patch_to_wall_ids,
     *_) = _import_room_geometry(json_path, patch_length)
    patches_normal = walls_normal[patch_to_wall_ids]
    patches_points = np.asarray(patches_points, dtype=float)

    actual = patch_visibility(patches_points, patches_normal)
    expected = sparrowpy.geometry._check_patch2patch_visibility(
        patches_points.mean(axis=1), patches_normal, patches_points)
    return actual, expected


@pytest.mark.parametrize("patch_length", [3, 1])
def test_visibility_matches_sparrowpy_box(
        create_temporary_input_file, patch_length):
    actual, expected = _visibility_both_ways(
        create_temporary_input_file, patch_length)
    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize("patch_length", [3, 1])
def test_visibility_matches_sparrowpy_l_shaped(
        create_temporary_input_file_tilted, patch_length):
    """The L-shaped room has patches that are hidden behind a corner.

    With ``patch_length=3`` sparrowpy before #168 wrongly reported one pair
    as blocked: its point-in-polygon test counted a polygon vertex lying
    exactly on its test ray twice. See
    ``test_visibility_segment_level_with_vertex``.
    """
    actual, expected = _visibility_both_ways(
        create_temporary_input_file_tilted, patch_length)
    # facing each other but blocked by the corner
    assert np.any(~actual & _facing(create_temporary_input_file_tilted,
                                    patch_length))
    npt.assert_array_equal(actual, expected)


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


def _point_visibility_both_ways(json_path, patch_length, points):
    (_, walls_normal, _, patches_points, _, patch_to_wall_ids,
     *_) = _import_room_geometry(json_path, patch_length)
    patches_normal = walls_normal[patch_to_wall_ids]
    patches_points = np.asarray(patches_points, dtype=float)
    centers = patches_points.mean(axis=1)

    actual = point_visibility(points, patches_points, patches_normal)
    expected = np.array([
        sparrowpy.geometry._check_point2patch_visibility(
            p, centers, patches_normal, patches_points) for p in points])
    return actual, expected


def _source_and_receivers(json_path):
    with open(json_path) as f:
        result = json.load(f)['results'][0]
    points = [[result['sourceX'], result['sourceY'], result['sourceZ']]]
    points += [[r['x'], r['y'], r['z']] for r in result['responses']]
    return np.array(points, dtype=float)


@pytest.mark.parametrize("patch_length", [3, 1])
def test_point_visibility_matches_sparrowpy_box(
        create_temporary_input_file, patch_length):
    points = _source_and_receivers(create_temporary_input_file)
    actual, expected = _point_visibility_both_ways(
        create_temporary_input_file, patch_length, points)
    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize("patch_length", [3, 1])
def test_point_visibility_matches_sparrowpy_l_shaped(
        create_temporary_input_file_tilted, patch_length):
    # source, receiver and two points in each leg of the L-shaped room
    points = np.vstack([
        _source_and_receivers(create_temporary_input_file_tilted),
        [[1.5, 4.0, 1.2], [4.5, 1.0, 1.7], [1.0, 1.0, 1.0]]])
    actual, expected = _point_visibility_both_ways(
        create_temporary_input_file_tilted, patch_length, points)
    # some patches are hidden behind the corner of the L
    assert not np.all(actual)
    npt.assert_array_equal(actual, expected)


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
