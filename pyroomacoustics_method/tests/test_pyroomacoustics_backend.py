"""Test the pyroomacoustics simulation backend.
"""
import json
import os

import numpy as np
import numpy.testing as npt

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


def test_export_rir_to_input(create_temporary_input_file):
    """Test the export_rir_to_input function."""
    rir = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    PyroomacousticsMethod(create_temporary_input_file)._export_rir_to_input(rir)

    with open(create_temporary_input_file, 'r') as f:
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
