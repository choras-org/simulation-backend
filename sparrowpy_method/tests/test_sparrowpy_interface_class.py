
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

    # test no inf
    for i in range(6):
        assert np.min(results[i]['data']) > -np.inf



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
    npt.assert_equal(np.squeeze(material_to_walls), np.arange(6))
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
    
    npt.assert_almost_equal(walls_normal[0], [0, 0, 1])  # floor
    npt.assert_almost_equal(walls_normal[2], [0, 0, -1])  # ceiling
    npt.assert_almost_equal(walls_normal[3], [0, 1, 0])  # wall2
    npt.assert_almost_equal(walls_normal[4], [1, 0, 0])  # wall3
