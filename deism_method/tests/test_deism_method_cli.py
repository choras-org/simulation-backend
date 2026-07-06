import os
import json
import numpy as np

from deism_interface import main


def test_deism_method_cli(mock_requests_post, create_temporary_input_file):
    """Test the DEISM method CLI."""
    # Set JSON_PATH environment variable and call main() directly
    os.environ["JSON_PATH"] = create_temporary_input_file
    main()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)

    rir = np.array(
        data['results'][0]['responses'][0]['receiverResults']
    )

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)
    assert data['results'][0]['percentage'] == 100

    # Verify that requests.post was called (save_results was executed)
    mock_requests_post.assert_called_once()
