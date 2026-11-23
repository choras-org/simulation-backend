import json
import numpy as np

from de_interface import DEMethod
from de_interface import DEinterface


def test_acoustic_de(create_temporary_input_file, monkeypatch):
    """
    Test the DE acoustic simulation method.
    """
    # keep the results of acousticDE to check where they are written to
    fvm_results = {}
    run_fvm_sim = DEinterface.run_fvm_sim
    monkeypatch.setattr(
        DEinterface, "run_fvm_sim",
        lambda *args: fvm_results.setdefault("r", run_fvm_sim(*args)))

    interface = DEMethod(create_temporary_input_file)
    interface.run_simulation()

    with open(create_temporary_input_file, 'r') as f:
         data = json.load(f)

    rir = np.array(data['results'][0]['responses'][0]['receiverResults'][0][
        "data"
    ])

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)

    # choras-org/simulation-backend#44: EDT held T20 and T20 held T30
    parameters = data['results'][0]['responses'][0]['parameters']
    np.testing.assert_allclose(parameters['edt'], fvm_results["r"]["edt_band"])
    np.testing.assert_allclose(parameters['t20'], fvm_results["r"]["t20_band"])
    np.testing.assert_allclose(parameters['t30'], fvm_results["r"]["t30_band"])
