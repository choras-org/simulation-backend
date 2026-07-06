import json
import numpy as np

from deism_interface import DeismMethod


def test_deism_method(create_temporary_input_file):
    """Test the DEISM acoustic simulation method."""
    interface = DeismMethod(create_temporary_input_file)
    interface.run_simulation()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)

    rir = np.array(
        data['results'][0]['responses'][0]['receiverResults']
    )

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)
    assert data["fs_auralization"] == data["simulationSettings"]["samplingRate"]
