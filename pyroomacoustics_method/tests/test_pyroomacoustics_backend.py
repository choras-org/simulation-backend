"""Test the pyroomacoustics simulation backend.
"""
import json
import os

import numpy as np
import numpy.testing as npt

from pyroomacoustics_interface import PyroomacousticsMethod
from pyroomacoustics_interface.pyroomacoustics_interface import (
    export_rir_to_csv,
    export_rir_to_input,
    get_receiver_positions,
    get_source_positions,
)


def test_get_receiver(default_input_data):
    """Test the get_receiver function."""
    receiver = get_receiver_positions(default_input_data)

    assert receiver is not None
    npt.assert_array_equal(receiver, np.array([[1.0, 1.0, 1.5]]))


def test_get_source_positions(default_input_data):
    """Test the get_source_positions function."""
    sources = get_source_positions(default_input_data)

    assert sources is not None
    npt.assert_array_equal(sources, np.array([2.0, 2.0, 1.5]))


def test_export_rir_to_input(create_temporary_input_file):
    """Test the export_rir_to_input function."""
    rir = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    export_rir_to_input(create_temporary_input_file, rir)

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)

    npt.assert_array_equal(
        data['results'][0]['responses'][0]['receiverResults'], rir)

def test_export_pressure_csv(create_temporary_input_file):
    """Test the export_pressure_csv function."""

    input_file = create_temporary_input_file
    rir = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    export_rir_to_input(input_file, rir)

    with open(input_file, 'r') as f:
        config = json.load(f)

    export_rir_to_csv(input_file)

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

def test_run_simulation(create_temporary_input_file):
    """Run the full simulation pipeline."""
    interface = PyroomacousticsMethod(create_temporary_input_file)
    interface.run_simulation()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)

    rir = np.array(data['results'][0]['responses'][0]['receiverResults'])

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)
