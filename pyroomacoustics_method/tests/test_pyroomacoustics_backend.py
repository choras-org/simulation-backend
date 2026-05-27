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


def test_get_reponse_data(create_temporary_input_file):
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


@pytest.mark.parametrize('config_file', [
    {'sampling_rate': 8000, 'image_source_order': 1},
    {'sampling_rate': 16000, 'image_source_order': 1},
    {'sampling_rate': 48000, 'image_source_order': 1},
    {'number_of_rays': 1000, 'image_source_order': 1},
    {'number_of_rays': 5000, 'image_source_order': 1},
    {'number_of_rays': 5000, 'image_source_order': -1},
    {'number_of_rays': 1000, 'image_source_order': 1, "detector_radius": 0.1},
], indirect=True)
def test_run_simulation(config_file):
    """Run the full simulation pipeline."""
    interface = PyroomacousticsMethod(config_file)
    interface.run_simulation()

    with open(config_file, 'r') as f:
        data = json.load(f)

    rir = np.array(data['results'][0]['responses'][0]['receiverResults'])

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)


@pytest.mark.parametrize('config_file', [
    {'number_of_rays': -1},
], indirect=True)
def test_invalid_number_rays(config_file):
    """Run the full simulation pipeline with invalid input data."""
    interface = PyroomacousticsMethod(config_file)
    with pytest.raises(ValueError, match="not allowed"):
        interface.run_simulation()
