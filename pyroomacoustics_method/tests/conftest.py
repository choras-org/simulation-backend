import os
import shutil
import tempfile
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
import pyfar as pf

import pytest


def default_data_path():
    """Get the path to the default data folder."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)))


def load_default_input_data():
    """Load the example input data."""
    with open(os.path.join(
            default_data_path(),
            "test_input_pyroomacoustics.json"), 'r') as f:
        data = json.load(f)

    return data


@pytest.fixture
def default_input_data():
    """Fixture to load the example input data."""
    return load_default_input_data()


@pytest.fixture
def create_temporary_input_file():
    """Fixture to create a temporary input JSON file for testing.

    Can be reused to write results to.
    """

    input_tmp = load_default_input_data()
    geo_file = os.path.join(
        default_data_path(), "test_room_pyroomacoustics.geo")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname) / "temp_input.json"
        shutil.copy(geo_file, Path(tmpdirname))
        input_tmp['geo_path'] = os.path.join(
            tmpdirname, "test_room_pyroomacoustics.geo")
        with open(tmp_path, 'w') as f:
            json.dump(input_tmp, f)

        yield str(tmp_path)

    return str(tmp_path)


def create_modified_settings_input_file_factory(key, value):
    """Factory to create a modified input file with a specific setting changed.

    Parameters
    ----------
    key : str
        The key in simulationSettings to modify.
    value : any
        The value to set for the key.

    Returns
    -------
    callable
        A generator function that yields the path to the temporary file.
    """

    def _create_modified_settings_input_file():
        input_tmp = load_default_input_data()

        # Ensure simulationSettings exists
        if 'simulationSettings' not in input_tmp:
            input_tmp['simulationSettings'] = {}

        input_tmp['simulationSettings'][key] = value
        geo_file = os.path.join(
            default_data_path(), "test_room_pyroomacoustics.geo")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname) / "temp_input.json"
            shutil.copy(geo_file, Path(tmpdirname))
            input_tmp['geo_path'] = os.path.join(
                tmpdirname, "test_room_pyroomacoustics.geo")
            with open(tmp_path, 'w') as f:
                json.dump(input_tmp, f)

            yield str(tmp_path)

    return _create_modified_settings_input_file


def create_modified_settings_input_file_multi_factory(**settings):
    """Factory to create a modified input file with multiple settings changed.

    Parameters
    ----------
    **settings : dict
        Key-value pairs to update in simulationSettings.

    Returns
    -------
    callable
        A generator function that yields the path to the temporary file.

    Examples
    --------
    >>> factory = create_modified_settings_input_file_multi_factory(
    ...     sampling_rate=48000,
    ...     image_source_order=3
    ... )
    >>> gen = factory()
    >>> json_path = next(gen)
    """

    def _create_modified_settings_input_file():
        input_tmp = load_default_input_data()

        # Ensure simulationSettings exists
        if 'simulationSettings' not in input_tmp:
            input_tmp['simulationSettings'] = {}

        # Update all provided settings
        input_tmp['simulationSettings'].update(settings)

        geo_file = os.path.join(
            default_data_path(), "test_room_pyroomacoustics.geo")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname) / "temp_input.json"
            shutil.copy(geo_file, Path(tmpdirname))
            input_tmp['geo_path'] = os.path.join(
                tmpdirname, "test_room_pyroomacoustics.geo")
            with open(tmp_path, 'w') as f:
                json.dump(input_tmp, f)

            yield str(tmp_path)

    return _create_modified_settings_input_file


@pytest.fixture
def json_file_factory():
    """Factory fixture to create JSON files with custom simulation settings.

    This fixture returns a callable that creates temporary JSON files with
    modified simulation settings. Use this in your test files with
    @pytest.mark.parametrize for flexible parametrization.

    Returns
    -------
    callable
        A function that takes **settings and returns a generator that yields
        the JSON file path.

    Examples
    --------
    In your test file:

    >>> @pytest.fixture
    >>> def config_file(request, json_file_factory):
    ...     gen = json_file_factory(**request.param)
    ...     return next(gen)
    >>>
    >>> @pytest.mark.parametrize('config_file', [
    ...     {'sampling_rate': 5000, 'image_source_order': 2},
    ...     {'sampling_rate': 10000, 'image_source_order': 3},
    ... ], indirect=True)
    >>> def test_something(config_file):
    ...     interface = PyroomacousticsMethod(config_file)
    ...     # test logic...
    """
    def _factory(**settings):
        return create_modified_settings_input_file_multi_factory(**settings)()

    return _factory


@pytest.fixture
def create_modified_input_file(request, json_file_factory):
    """Fixture that creates a JSON file based on test parameters."""
    gen = json_file_factory(**request.param)
    json_path = next(gen)
    # Make sure that the generator is properly closed after the test function
    # finalizes to ensure cleanup of temporary files
    try:
        yield json_path
    finally:
        gen.close()



@pytest.fixture
def mock_requests_post():
    """Fixture to mock requests.post for CLI tests.

    Returns the mock object so tests can make assertions on it.
    """
    with patch("pyroomacoustics_interface.definition.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def example_rir() -> pf.Signal:
    """Fixture to provide an example RIR for testing."""
    return pf.signals.files.room_impulse_response()
