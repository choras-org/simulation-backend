import json

from pyroomacoustics_interface import PyroomacousticsMethod

from pyroomacoustics_interface.pyroomacoustics_interface import (
    calculate_room_acoustic_parameters,
)


def test_rap_estimation(example_rir):
    """Test the estimation of room acoustic parameters."""

    bands = [125, 250, 500, 1000, 2000, 4000]

    parameters = calculate_room_acoustic_parameters(
        example_rir, bands=bands)

    assert parameters is not None

    for param in ['t20', 't30', 'edt', 'd50', 'c80', 'ts', 'spl_t0_freq']:
        assert isinstance(parameters[param], list)
        assert len(parameters[param]) == len(bands)


def test_export_parameters_to_dict(create_temporary_input_file):
    """Test exporting the estimated parameters to a JSON file."""

    parameters = {
        't20': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        't30': [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        'edt': [0.4, 0.3, 0.2, 0.1, 0.05, 0.02],
        'd50': [50, 60, 70, 80, 90, 95],
        'c80': [-5, -3, -1, 1, 3, 5],
        'ts': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        'spl_t0_freq': [60, 65, 70, 75, 80, 85],
    }

    input_file = create_temporary_input_file

    PyroomacousticsMethod(input_file)._export_room_acoustic_parameters_to_json(
        parameters)

    with open(input_file, 'r') as f:
        output_file = json.load(f)

    param_names = list(parameters.keys())
    print(param_names)

    for param_name in param_names:
        assert output_file['results'][0]['responses'][0][
            'parameters'
        ][param_name] == parameters[param_name]
