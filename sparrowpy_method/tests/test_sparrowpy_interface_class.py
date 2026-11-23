
from sparrowpy_interface import sparrowpyMethod
import json
import sparrowpy
import numpy.testing as npt
import numpy as np

from sparrowpy_interface.sparrowpy_interface import _import_room_geometry


def test_simple_method(create_temporary_input_file):
    
    sparrowpy_method_object = sparrowpyMethod(create_temporary_input_file)
    sparrowpy_method_object.run_simulation()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)
    
    assert "receiverResults" in data['results'][0]['responses'][0]
    results = data['results'][0]['responses'][0]['receiverResults']
    assert results is not None
    assert len(results) > 0

    # receiverResults is the broadband RIR as a flat list of samples
    assert np.all(np.isfinite(results))



def test_import_room_geometry(create_temporary_input_file):
    (
        walls_points, walls_normal, walls_up_vector,
        patches_points, n_patches, patch_to_wall_ids,
        material_to_walls, alphas, scattering, room_volume,
        ) = _import_room_geometry(
            create_temporary_input_file, patch_length=5)

    assert room_volume > 88
    
    # create radiosity object
    radiosity = sparrowpy.DirectionalRadiosityFast(
        walls_points,
        walls_normal,
        walls_up_vector,
        patches_points,
        n_patches,
        patch_to_wall_ids,
        )

    # check geometry stuff
    radiosity.check()

    # check material stuff 
    # one wall per patch, each in exactly one material
    npt.assert_equal(len(walls_points), n_patches)
    npt.assert_equal(np.sort(np.concatenate(material_to_walls)), np.arange(n_patches))
    npt.assert_equal(np.array(alphas).shape, (6, 6))
    npt.assert_equal(np.array(scattering).shape, (6, 6))
    npt.assert_array_less(np.array(alphas), 1)
    npt.assert_allclose(np.array(scattering), 1)


def test_import_room_geometry_normals(create_temporary_input_file):
    (
        walls_points, walls_normal, walls_up_vector,
        patches_points, n_patches, patch_to_wall_ids,
        material_to_walls, alphas, scattering, room_volume,
        ) = _import_room_geometry(
            create_temporary_input_file, patch_length=5)
    
    # first wall (triangle) of each surface
    npt.assert_almost_equal(walls_normal[material_to_walls[0][0]], [0, 0, 1])  # floor
    npt.assert_almost_equal(walls_normal[material_to_walls[2][0]], [0, 0, -1])  # ceiling
    npt.assert_almost_equal(walls_normal[material_to_walls[3][0]], [0, 1, 0])  # wall2
    npt.assert_almost_equal(walls_normal[material_to_walls[4][0]], [1, 0, 0])  # wall3


def test_import_room_geometry_tilted_area(create_temporary_input_file_tilted):
    (
        walls_points, walls_normal, walls_up_vector,
        patches_points, n_patches, patch_to_wall_ids,
        material_to_walls, alphas, scattering, room_volume,
        ) = _import_room_geometry(
            create_temporary_input_file_tilted, patch_length=1)

    # walls must cover the whole room surface, including the tilted
    # ceiling and the non-rectangular floor
    area = 0.5 * np.linalg.norm(np.cross(
        walls_points[:, 1] - walls_points[:, 0],
        walls_points[:, 2] - walls_points[:, 0]), axis=1)
    npt.assert_almost_equal(np.sum(area), 118.7386, decimal=3)
    npt.assert_almost_equal(room_volume, 75.75, decimal=3)
