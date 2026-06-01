import json
import os

import numpy as np
import pytest

from pyroomacoustics_interface import main


@pytest.mark.parametrize('create_modified_input_file', [
    {'sampling_rate': 8000, 'image_source_order': 1},
    {'sampling_rate': 16000, 'image_source_order': 1},
    {'sampling_rate': 48000, 'image_source_order': 1},
    {'number_of_rays': 1000, 'image_source_order': 1},
    {'number_of_rays': 5000, 'image_source_order': 1},
    {'number_of_rays': 5000, 'image_source_order': -1},
    {'number_of_rays': 1000, 'image_source_order': 1, "detector_radius": 0.1},
], indirect=True)
def test_pyroomacoustics_method_cli(
        mock_requests_post,
        create_modified_input_file
    ):
    """Test the Pyroomacoustics method CLI."""
    # Set JSON_PATH environment variable and call main() directly
    os.environ["JSON_PATH"] = create_modified_input_file
    main()

    with open(create_modified_input_file, 'r') as f:
        data = json.load(f)

    rir = np.array(data['results'][0]['responses'][0]['receiverResults'])

    assert rir is not None
    assert len(rir) > 0
    assert isinstance(rir, np.ndarray)
    assert np.any(np.abs(rir) >= 1e-6)

    # Verify that requests.post was called (save_results was executed)
    mock_requests_post.assert_called_once()
