"""Test the sparrowpy pipeline with repaired room geometries from the frontend."""
import json
import os
import shutil
from pathlib import Path
import pyfar as pf

import pytest

from sparrowpy_interface import sparrowpyMethod

from conftest import default_data_path


@pytest.fixture(params=[
    "MeasurementRoom_repaired",
    "Room2215_simple_repaired",
])
def repaired_room_input_file(request, tmp_path):
    """Copy the input JSON, geo and msh file of a repaired room into a
    temporary directory and return the path to the JSON file."""
    room = request.param
    with open(os.path.join(default_data_path(), f"{room}.json"), 'r') as f:
        input_data = json.load(f)

    for ext in ("geo", "msh"):
        shutil.copy(
            os.path.join(default_data_path(), f"{room}.{ext}"), tmp_path)
    input_data['geo_path'] = str(tmp_path / f"{room}.geo")
    input_data['msh_path'] = str(tmp_path / f"{room}.msh")
    # drop error entry from the frontend export
    input_data.pop('error', None)

    json_path = Path(tmp_path) / "temp_input.json"
    with open(json_path, 'w') as f:
        json.dump(input_data, f)

    return str(json_path)


def test_pipeline_runs_to_results(repaired_room_input_file):
    """The simulation runs through and writes receiver results."""
    sparrowpyMethod(repaired_room_input_file).run_simulation()

    with open(repaired_room_input_file, 'r') as f:
        data = json.load(f)

    rir = data['results'][0]['responses'][0]['receiverResults']
    fs = data['simulationSettings']['sampling_rate']
    rir = pf.Signal(rir, sampling_rate=fs)
    f = data['results'][0]['frequencies']
    t20 = data['results'][0]['responses'][0]['parameters']['t20']
    assert np.all(t20>0)
    assert results is not None
    assert len(results) > 0
