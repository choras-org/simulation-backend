"""Test the sparrowpy pipeline with repaired room geometries from the frontend."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sparrowpy_interface import sparrowpyMethod
from sparrowpy_interface.sparrowpy_interface import _import_room_geometry

from conftest import default_data_path

# T20 is fitted between -5 and -25 dB of the energy decay curve. Only bands
# whose expected decay reaches -35 dB within the simulated duration are
# checked, otherwise the truncated decay curve biases the fit.
_DECAY_DB = 35
# Bounds of the simulated T20 relative to Eyring's estimate. Radiosity with
# diffuse walls stays below Eyring in non-diffuse (e.g. flat) rooms, so the
# bounds are wide; they catch broken geometry such as normals pointing out
# of the room or wrong visibility, which give no decay or none at all.
_EYRING_RATIO = (0.3, 1.2)


@pytest.fixture(scope="module", params=[
    "MeasurementRoom_repaired",
    "Room2215_simple_repaired",
])
def simulated_room(request, tmp_path_factory):
    """Run the simulation for a repaired room once and return the input
    path and the written results."""
    room = request.param
    tmp_path = tmp_path_factory.mktemp(room)
    data_path = Path(default_data_path())
    with open(data_path / f"{room}.json") as f:
        input_data = json.load(f)

    shutil.copy(data_path / f"{room}.geo", tmp_path)
    input_data['geo_path'] = str(tmp_path / f"{room}.geo")
    input_data['msh_path'] = str(tmp_path / f"{room}.msh")
    json_path = tmp_path / "input.json"
    with open(json_path, 'w') as f:
        json.dump(input_data, f)

    sparrowpyMethod(json_path).run_simulation()
    with open(json_path) as f:
        return json_path, json.load(f)


def _eyring(json_path, data):
    """Return Eyring's reverberation time and the mean absorption per band."""
    patch_length = data['simulationSettings']['patch_length']
    (walls_points, *_, material_to_walls, alphas, _, volume) = \
        _import_room_geometry(json_path, patch_length)
    walls_points = np.asarray(walls_points, dtype=float)
    area = 0.5 * np.linalg.norm(np.cross(
        walls_points[:, 1] - walls_points[:, 0],
        walls_points[:, 2] - walls_points[:, 0]), axis=1)
    absorption_area = sum(
        area[walls].sum() * np.asarray(alpha)
        for walls, alpha in zip(material_to_walls, alphas))
    alpha_mean = absorption_area / area.sum()
    t_eyring = 0.163 * volume / (
        -area.sum() * np.log(1 - alpha_mean))
    return t_eyring, alpha_mean


def _responses(data):
    return data['results'][0]['responses']


def test_results_are_finite(simulated_room):
    _, data = simulated_room
    settings = data['simulationSettings']
    n_samples = int(settings['sampling_rate'] * settings['etc_duration_s'])
    for response in _responses(data):
        rir = np.asarray(response['receiverResults'])
        assert rir.shape == (n_samples, )
        assert np.all(np.isfinite(rir))
        assert np.any(rir != 0)
        for name in ('t20', 't30', 'c80', 'd50', 'ts', 'spl_t0_freq'):
            assert np.all(np.isfinite(response['parameters'][name])), name
        # -1 means the regression failed, e.g. on an unnormalized EDC
        assert np.all(np.asarray(response['parameters']['edt']) > 0)


def test_reverberation_time_matches_eyring(simulated_room):
    json_path, data = simulated_room
    t_eyring, _ = _eyring(json_path, data)
    checked = t_eyring * _DECAY_DB / 60 < \
        data['simulationSettings']['etc_duration_s']
    assert np.any(checked), "simulation too short to check any band"

    for response in _responses(data):
        ratio = np.asarray(response['parameters']['t20']) / t_eyring
        assert np.all(ratio[checked] > _EYRING_RATIO[0]), ratio
        assert np.all(ratio[checked] < _EYRING_RATIO[1]), ratio


def test_reverberation_time_follows_absorption(simulated_room):
    """More absorption in a band gives a shorter T20, equal absorption a
    similar one."""
    json_path, data = simulated_room
    t_eyring, alpha_mean = _eyring(json_path, data)
    checked = t_eyring * _DECAY_DB / 60 < \
        data['simulationSettings']['etc_duration_s']

    for response in _responses(data):
        t20 = np.asarray(response['parameters']['t20'])[checked]
        alpha = alpha_mean[checked]
        for i in range(len(t20) - 1):
            if np.isclose(alpha[i + 1], alpha[i]):
                assert np.isclose(t20[i + 1], t20[i], rtol=0.15), t20
            elif alpha[i + 1] > alpha[i]:
                assert t20[i + 1] < t20[i], t20
            else:
                assert t20[i + 1] > t20[i], t20
