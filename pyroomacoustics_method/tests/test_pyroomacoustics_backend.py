"""Test the pyroomacoustics simulation backend.
"""
import json
import os

import numpy as np
import numpy.testing as npt
import pytest

from pyroomacoustics_interface import PyroomacousticsMethod


def test_get_receiver(create_temporary_input_file):
    """Test the get_receiver function."""
    interface = PyroomacousticsMethod(create_temporary_input_file)
    receiver = interface.get_receiver_positions()

    assert receiver is not None
    npt.assert_array_equal(receiver, np.array([[1.0, 1.0, 1.5]]))


def test_get_source_positions(create_temporary_input_file):
    """Test the get_source_positions function."""
    interface = PyroomacousticsMethod(create_temporary_input_file)
    sources = interface.get_source_positions()

    assert sources is not None
    npt.assert_array_equal(sources, np.array([2.0, 2.0, 1.5]))


def test_get_simulation_settings(create_temporary_input_file):
    """Test the get_simulation_settings function."""
    interface = PyroomacousticsMethod(create_temporary_input_file)
    settings = interface._get_simulation_settings()

    assert settings is not None
    assert settings['sampling_rate'] == 5000
    assert settings['image_source_order'] == 2


def test_get_results(create_temporary_input_file):
    """Test the get_results function."""
    interface = PyroomacousticsMethod(create_temporary_input_file)
    idx = 0
    results = interface._get_result_data(idx)

    assert results is not None
    assert 'responses' in results
    for source_pos in ['sourceX', 'sourceY', 'sourceZ']:
        assert source_pos in results


def test_get_response_data(create_temporary_input_file):
    """Test the get_response_data function."""
    interface = PyroomacousticsMethod(create_temporary_input_file)
    idx_receiver = 0
    idx_source = 0

    response = interface._get_response_data(
        response_idx=idx_receiver, result_idx=idx_source)

    assert response is not None
    for key in ['x', 'y', 'z']:
        assert key in response


def test_export_rir_to_input(create_temporary_input_file):
    """Test the export_rir_to_input function."""

    input_file = create_temporary_input_file
    rir = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    PyroomacousticsMethod(create_temporary_input_file)._export_rir_to_input(rir)

    with open(input_file, 'r') as f:
        data = json.load(f)

    npt.assert_array_equal(
        data['results'][0]['responses'][0]['receiverResults'], rir)

def test_export_pressure_csv(create_temporary_input_file):
    """Test the export_pressure_csv function."""

    input_file = create_temporary_input_file
    rir = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    interface = PyroomacousticsMethod(input_file)
    interface._export_rir_to_input(rir)

    with open(input_file, 'r') as f:
        config = json.load(f)

    interface.export_rir_to_csv()

    output_file = str(input_file).replace('.json', '_pressure.csv')
    assert os.path.exists(output_file)

    data_csv = np.loadtxt(output_file, delimiter=',', skiprows=1)
    times = data_csv[:, 0]
    pressures = data_csv[:, 1]

    npt.assert_array_equal(pressures, rir)
    npt.assert_array_equal(
        times,
        np.arange(len(rir)) / config['simulationSettings']['sampling_rate'],
    )


# Input JSON files of simulation runs, each referencing its geometry via
# geo_path
simulation_runs = [
    "test_input_pyroomacoustics.json",
    "MeasurementRoom_092dd9ff70824eb7801bc5ed5aac07d2_13.json",
]


@pytest.mark.parametrize(
    "create_modified_input_file",
    [
        {"input_file": simulation_run, **settings}
        for simulation_run in simulation_runs
        for settings in [
            {"sampling_rate": 8000, "image_source_order": 1},
            {"sampling_rate": 16000, "image_source_order": 1},
            {"sampling_rate": 48000, "image_source_order": 1},
            {"ray_tracer_number_of_rays": 1000, "image_source_order": 1},
            {"ray_tracer_number_of_rays": 5000, "image_source_order": 1},
            {"ray_tracer_number_of_rays": 5000, "image_source_order": -1},
            {
                "ray_tracer_number_of_rays": 1000,
                "image_source_order": 1,
                "ray_tracer_detector_radius": 0.1,
            },
        ]
    ],
    indirect=True,
)
def test_run_simulation(create_modified_input_file):
    """Run the full simulation pipeline."""

    input_file_path = create_modified_input_file
    interface = PyroomacousticsMethod(input_file_path)
    interface.run_simulation()

    with open(input_file_path, 'r') as f:
        input_config = json.load(f)

    result = input_config["results"][0]
    response = result["responses"][0]
    rir = np.array(response["receiverResults"])

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)

    for name, values in response["parameters"].items():
        assert len(values) == len(result["frequencies"]), name
        assert np.all(np.isfinite(values)), name


@pytest.mark.parametrize('create_modified_input_file', [
    {
        'ray_tracer_number_of_rays': 1000,
        'ray_tracer_detector_radius': 0.1,
        'ray_tracer_energy_threshold': -45,
        'ray_tracer_time_threshold': 1,
    },
    {'ray_tracer_number_of_rays': 5000},
    {'speed_of_sound': 343.0},
    {'speed_of_sound': 200.0},
    {'global_scattering_coefficient': 0.0},
    {'global_scattering_coefficient': 1.0},
], indirect=True)
def test_run_simulation_ray_tracer_settings(
        create_modified_input_file,
    ):
    """Run the full simulation pipeline with ray tracing settings."""
    input_file_path = create_modified_input_file

    interface = PyroomacousticsMethod(input_file_path)
    interface.run_simulation()

    with open(input_file_path, 'r') as f:
        input_config = json.load(f)

    rir = np.array(
        input_config['results'][0]['responses'][0]['receiverResults'])

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)


@pytest.mark.parametrize('create_modified_input_file', [
    {'ray_tracer_number_of_rays': -1},
], indirect=True)
def test_invalid_number_rays(create_modified_input_file):
    """Run the full simulation pipeline with invalid input data."""
    interface = PyroomacousticsMethod(create_modified_input_file)
    with pytest.raises(ValueError, match="not allowed"):
        interface.run_simulation()


def test_import_room_geometry_multi_surface_group(create_temporary_input_file):
    """All surfaces of a physical group become walls, not just the first."""
    n_walls = len(
        PyroomacousticsMethod(
            create_temporary_input_file
        ).import_room_geometry()
    )

    # merge walls sharing a material (surfaces 2, 4, 5, 6) into one
    # physical group
    with open(create_temporary_input_file) as f:
        config = json.load(f)
    with open(config["geo_path"]) as f:
        geo = [
            line
            for line in f
            if not line.startswith(
                tuple(f'Physical Surface("wall{i}")' for i in range(1, 5))
            )
        ]
    with open(config["geo_path"], "w") as f:
        f.writelines(geo + ['Physical Surface("walls") = { 2, 4, 5, 6 };\n'])
    coefficients = config["absorption_coefficients"]
    coefficients["walls"] = coefficients["wall1"]
    for i in range(1, 5):
        del coefficients[f"wall{i}"]
    with open(create_temporary_input_file, "w") as f:
        json.dump(config, f)

    walls = PyroomacousticsMethod(
        create_temporary_input_file
    ).import_room_geometry()
    assert len(walls) == n_walls
