"""Test the MoDART method CLI."""
import os
import json
import pytest
import numpy as np

from modart_interface import main


"""
DEFAULT VALUES:
{'absorption_coefficients': {'walls': '0.02, 0.03, 0.04, 0.08, 0.15',
                             'floor': '0.11, 0.22, 0.42, 0.57, 0.63',
                             'ceiling': '0.01, 0.01, 0.01, 0.01, 0.01'},
 'simulationSettings': {'durat': 1,
                        'f_e': 10000,
                        'T60': 0.1,
                        'slopes': 2,
                        'humi': 50,
                        'temp': 20,
                        'pres': 100,
                        'ppsm': 30,
                        'rays': 1000,
                        'pool': 4}},
"""
@pytest.mark.parametrize('create_modified_input_file', [
    {'simulationSettings': {'pool': 1}},
    {'simulationSettings': {'pool': 2}},
    {'simulationSettings': {'slopes': 1}},
    {'simulationSettings': {'slopes': 3}},
    {'absorption_coefficients': {'walls': '0, 0, 0, 0, 0',
                                 'floor': '0, 0, 0, 0, 0',
                                 'ceiling': '0, 0, 0, 0, 0'},
     'simulationSettings': {'slopes': 1, 'humi': 0, 'temp': 0}},
    {'absorption_coefficients': {'walls': '1, 1, 1, 1, 1',
                                 'floor': '1, 1, 1, 1, 1',
                                 'ceiling': '1, 1, 1, 1, 1'},
     'simulationSettings': {'slopes': 1}},
], indirect=True)
def test_modart_method_cli(mock_requests_post, create_modified_input_file):
    """Test the MoDART method CLI."""
    # Set JSON_PATH environment variable and call main() directly
    os.environ["JSON_PATH"] = create_modified_input_file

    with open(create_modified_input_file, 'r') as f:
        input_data = json.load(f)
    
    assert 'simulationSettings' in input_data
    settings = input_data['simulationSettings']
    assert len(settings) > 0
    
    assert 'absorption_coefficients' in input_data
    coeffs = input_data['absorption_coefficients']
    assert len(coeffs) > 0
    
    if all(coeff == '1, 1, 1, 1, 1'
           for coeff in coeffs.values()):
        # When all material absorptions are exactly 1, reverberation time is 0.
        # The state transition matrix becomes singular, and decomposition is impossible.
        # with pytest.raises(RuntimeError, match='Failed to run the modal analysis'):
        #     main()

        # https://stackoverflow.com/a/73478360
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

        with open(create_modified_input_file, 'r') as f:
            output_data = json.load(f)
        
        assert 'error' in output_data
        error = output_data['error']
        assert 'type' in error
        assert 'message' in error
        assert error['type'] == 'RuntimeError'
        assert 'Failed to run' in error['message']
    else:
        # With other settings, the decomposition should not have any issues.
        # N.B.: All remaining tests are in this scope, where "main()" is successful.
        main()

        with open(create_modified_input_file, 'r') as f:
            output_data = json.load(f)
        
        #### ASSERTIONS FOR ALL TEST CASES ####

        assert 'results' in output_data
        results = output_data['results']
        assert len(results) > 0
        for res in results:
            assert 'responses' in res
            responses = res['responses']
            assert len(responses) > 0
            for resp in responses:
                assert 'receiverResults' in resp
                rec_res = resp['receiverResults']
                assert rec_res is not None
                assert len(rec_res) > 0

                assert np.any(np.abs(rec_res) > 0)

                # for rir in rec_res:
                #     assert len(r) == 4
                #     assert 'data' in r
                #     assert 't' in r
                #     assert 'frequency' in r
                #     assert 'type' in r
                #     assert r['type'] == 'edc'
                #     assert len(r['data']) == len(r['t'])

                #     assert np.all(np.isfinite(r['data']))

        # The function returns a few MoD-ART parameters for debugging/testing.
        assert 'MoDART_data' in output_data
        MoDART_data = output_data['MoDART_data']
        assert len(MoDART_data) > 0

        # The simulation settings ask for a number of slopes in 5 frequency bands,
        #  so there should be this many in total.
        expected_modes = settings['slopes'] * 5

        assert len(MoDART_data['T60']) == expected_modes
        assert len(MoDART_data['Band idx']) == expected_modes

        # The band indices of the detected slopes should range from 0 to 4,
        #  and there should be a set number of each.
        unique, unique_counts = np.unique(MoDART_data['Band idx'], return_counts=True)
        assert np.all(unique == np.arange(5))
        assert np.all(unique_counts == settings['slopes'])

        # The test room has 6 patches with full visibility, so there should be 6*(6-1)=30 paths.
        # If the second dimension is != 30, that means something went wrong in the mesh decoding.
        assert MoDART_data['Eigenvector shape'] == [expected_modes, 30]
        
        #### ASSERTIONS SPECIFIC TO EACH TEST CASE ####

        if all(coeff == '0, 0, 0, 0, 0'
            for coeff in coeffs.values()):
            if settings['slopes'] != 1:
                raise NotImplementedError('I did not prepare that much reference data.')
            
            # When all material absorptions are 0, reverberation time is only governed by air absorption.
            assert np.allclose(MoDART_data['T60'],
                               [194.175, 184.681, 179.758, 169.130, 137.950],
                               rtol=0.05, atol=0.05)
        
        else:
            # Confirm that the T60 values are reasonably close to a reference run.
            if settings['slopes'] == 1:
                assert np.allclose(MoDART_data['T60'],
                                   [0.56, 0.32, 0.19, 0.13, 0.1],
                                   rtol=0.05, atol=0.05)
            elif settings['slopes'] == 2:
                assert np.allclose(MoDART_data['T60'],
                                   [0.56, 0.03, 0.32, 0.03, 0.19, 0.03, 0.13, 0.03, 0.1, 0.02],
                                   rtol=0.05, atol=0.05)
            elif settings['slopes'] == 3:
                assert np.allclose(MoDART_data['T60'],
                                   [0.56, 0.03, 0.03,
                                    0.32, 0.03, 0.03,
                                    0.19, 0.03, 0.03,
                                    0.13, 0.03, 0.03,
                                    0.1, 0.02, 0.02],
                                    rtol=0.05, atol=0.05)
            else:
                raise NotImplementedError('I did not prepare that much reference data.')

        # Verify that requests.post was called (save_results was executed)
        mock_requests_post.assert_called_once()


def test_modart_method_cli_missing_json_path(mock_requests_post):
    """Test the MoDART method CLI with missing JSON_PATH."""
    # Clear JSON_PATH environment variable
    if "JSON_PATH" in os.environ:
        del os.environ["JSON_PATH"]

    # Expect FileNotFoundError from SimulationMethod.__init__
    with pytest.raises(FileNotFoundError, match="input_json_path cannot be None or empty"):
        main()
