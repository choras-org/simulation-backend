"""Module implementing a CHORAS interface for pyroomacoustics.
"""
import warnings
from pathlib import Path

import gmsh
import numpy as np
import pandas as pd
import pyfar as pf
import pyrato
import pyroomacoustics as pra

# Support both package and script execution.
from .definition import SimulationMethod


class PyroomacousticsMethod(SimulationMethod):
    """Interface class to run simulations using pyroomacoustics.

    This interface class sets configures the simulation parameters,
    source and receiver positions, as well as room geometry and boundary data.

    On a successful simulation, the computed RIRs are exported to the input
    data file.

    Parameters
    ----------
    input_json_path : str or Path, optional
        Path to the simulation configuration file.

    """

    def __init__(self, input_json_path: str | Path | None):
        """Initialize from configuration file.

        Parameters
        ----------
        input_json_path : str | Path | None, optional
            The input configuration file path. Note that if ``None`` is
            provided, the simulation will return an error. This input is only
            allowed to support the case where the environment variable is not
            set.
        """
        super().__init__(input_json_path)

    def run_simulation(self) -> None:
        """Execute the simulation and export results to the configuration file.
        """

        print("pyroomacoustics_method: starting simulation")


        walls = import_room_geometry(self.input_json_path)

        simulation_setup = setup_simulation(self.input_json_path, walls)

        # Compute the RIRs
        simulation_setup.compute_rir()

        # Get the RIRs for the first source and first microphone
        rir = simulation_setup.rir[0][0]

        # Export the RIRs to the input data structure
        export_rir_to_input(self.input_json_path, rir)

        print("pyroomacoustics_method: simulation done!")


def read_json_input(json_file_path: str | Path) -> dict:
    """Read the input JSON file and return the content as a dictionary.

    Parameters
    ----------
    json_file_path : str | Path
        Path of the input JSON file

    Returns
    -------
    input_data : dict
        Parsed JSON data as a dictionary

    """
    with open(json_file_path, 'r') as f:
        import json
        input_data = json.load(f)

    return input_data


def import_room_geometry(json_file_path: str | Path) -> list[pra.Wall]:
    """Import room geometry and absorption coefficients.

    The geometry is read from a .geo file specified in the JSON input file.
    The absorption coefficients are directly read from the JSON file.

    Parameters
    ----------
    json_file_path : str | Path
        Path to the JSON file containing room geometry and absorption
        coefficients.

    Returns
    -------
    walls : list of pyroomacoustics.Wall
        List of walls defining the room geometry and boundary conditions for
        all frequency bands.

    Raises
    ------
    ValueError
        If absorption coefficients for any surface are not found in the
        input JSON file.
    """

    with open(json_file_path, 'r') as f:
        import json
        input_data = json.load(f)


    # initialize gmsh and load the geometry file
    gmsh.initialize()
    try:
        walls = _import_room_geometry(input_data)
    finally:
        gmsh.finalize()

    return walls


def _import_room_geometry(input_data: dict) -> list[pra.Wall]:
    """Private import class for geometry and boundary conditions.

    This private function ensures that gmsh is properly finalized after
    geometry import, even if an error occurs during the import process.

    The public function `import_room_geometry` is responsible for initializing
    and finalizing gmsh.

    Parameters
    ----------
    input_data : dict
        The input configuration data.

    Returns
    -------
    list[pra.Wall]
        List of walls defining the room geometry and boundary conditions for 
        all frequency bands.

    Raises
    ------
    ValueError
        If absorption coefficients for any surface are not found in the
        input JSON file.
    """


    frequencies = input_data['results'][0]['frequencies']
    geometry_file = input_data['geo_path']
    gmsh.open(geometry_file)

    # generate 2d surface mesh
    dim = 2 # 2D surfaces
    gmsh.model.mesh.generate(dim)

    # get all named surfaces in the geometry
    surface_group_tags = gmsh.model.getPhysicalGroups(dim=dim)
    surface_group_names = [
        gmsh.model.getPhysicalName(dim, tag)
        for (dim, tag) in surface_group_tags
    ]

    # get all nodes of the surface mesh
    node_tags_all, coords_all, _ = gmsh.model.mesh.getNodes()
    coords = coords_all.reshape((len(node_tags_all), 3))

    # get the material names from absorption coefficient input
    absorption_names = list(input_data['absorption_coefficients'].keys())

    # check if absorption coefficient data are available for all surfaces
    for name in surface_group_names:
        if name not in absorption_names:
            raise ValueError(
                "Absorption coefficients for surface "
                f"'{name}' not found in input JSON file.")


    # get the element type for surface mesh
    element_type = gmsh.model.mesh.getElementType("triangle", 1)

    # create pyroomacoustics.walls for all surface elements (triangles)
    walls = []

    # loop through all named surfaces and create walls with corresponding
    # absorption and scattering coefficients

    for surface_name in surface_group_names:
        dim_tags = gmsh.model.getEntitiesForPhysicalName(surface_name)
        dim, tag = dim_tags[0]

        face_nodes = gmsh.model.mesh.getElementFaceNodes(
            element_type, 3, tag=tag)
        faces = np.reshape(face_nodes, (len(face_nodes) // 3, 3))

        absorption_coeffs_config = input_data[
            'absorption_coefficients'][surface_name]
        if isinstance(absorption_coeffs_config, str):
            absorption_coeffs = np.array(
                [
                    float(x.strip())
                    for x in absorption_coeffs_config.split(",")
                ],
                dtype=float)
        elif isinstance(absorption_coeffs_config, list):
            absorption_coeffs = np.array(
                absorption_coeffs_config, dtype=float)
        elif isinstance(absorption_coeffs_config, np.ndarray):
            absorption_coeffs = absorption_coeffs_config.astype(float)
        else:
            raise ValueError(
                "Invalid format for the absorption coefficient. ",
                f"Got type {type(absorption_coeffs_config)}."
            )

        material = pra.Material(
            energy_absorption={
                'description': surface_name,
                'center_freqs': frequencies,
                'coeffs': absorption_coeffs,
            },
            scattering={
                'description': surface_name,
                'center_freqs': frequencies,
                'coeffs': np.ones_like(absorption_coeffs)*0.1,
            }
        )

        walls.extend(
            pra.wall_factory(
                coords[face - 1, :].reshape(-1, 3).T,
                absorption=material.energy_absorption['coeffs'],
                scattering=material.scattering['coeffs'],
            )
            for face in faces
        )

    return walls


def get_source_positions(input_data: dict) -> np.ndarray:
    """Extract source positions from input data.

    Parameters
    ----------
    input_data : dict
        Input data as a dictionary.

    Returns
    -------
    source_positions : np.ndarray
        Array of source positions with shape (3,).
    """
    return np.array([
        input_data['results'][0]['sourceX'],
        input_data['results'][0]['sourceY'],
        input_data['results'][0]['sourceZ'],
    ])


def get_receiver_positions(input_data: dict) -> np.ndarray:
    """Extract receiver positions from input data.

    Parameters
    ----------
    input_data : dict
        Input data as a dictionary.

    Returns
    -------
    receiver_positions : np.ndarray
        Array of receiver positions with shape (n_receivers, 3).
    """

    num_receivers = len(input_data['results'][0]['responses'])

    response_section = input_data['results'][0]['responses']

    receiver_pos = np.zeros((num_receivers, 3), dtype=float)

    for i in range(num_receivers):
        receiver_pos[i] = np.array([
            response_section[i]['x'],
            response_section[i]['y'],
            response_section[i]['z']
        ])

    return receiver_pos


def set_default_simulation_settings(input_data: dict) -> dict:
    """Set default simulation settings if not provided in input data.

    Parameters
    ----------
    input_data : dict
        Input data as a dictionary.

    """
    if 'simulationSettings' not in input_data:
        input_data['simulationSettings'] = {}

    settings = input_data['simulationSettings']

    if 'image_source_order' not in settings:
        settings['image_source_order'] = 2
        warnings.warn(
            "Image source order not specified. "
            "Defaulting to 2nd order reflections.",
            stacklevel=2
        )

    if 'sampling_rate' not in settings:
        settings['sampling_rate'] = 20000
        warnings.warn(
            "Sampling rate not specified. "
            "Defaulting to 20000 Hz.",
            stacklevel=2
        )

    if 'ray_tracing' not in settings:
        settings['ray_tracing'] = True
        warnings.warn(
            "Ray tracing setting not specified. "
            "Defaulting to True.",
            stacklevel=2
        )

    if 'air_absorption' not in settings:
        settings['air_absorption'] = True
        warnings.warn(
            "Air absorption setting not specified. "
            "Defaulting to True.",
            stacklevel=2
        )

    return input_data


def setup_simulation(
        json_file_path: str | Path,
        walls: list[pra.Wall]
    ) -> pra.Room:
    """Set up the pyroomacoustics simulation based on the JSON file.

    Parameters
    ----------
    json_file_path : str
        Path of the input JSON file
    walls : list of pyroomacoustics.Wall
        List of walls defining the room geometry and boundary conditions

    Returns
    -------
    room : pyroomacoustics.Room
        The configured pyroomacoustics Room object
    """

    print("setup_simulation: setting up simulation")


    input_data = read_json_input(json_file_path)
    extended_input_data = set_default_simulation_settings(input_data)

    sampling_rate = extended_input_data["simulationSettings"].get(
        "sampling_rate"
    )
    image_source_order = extended_input_data["simulationSettings"].get(
        "image_source_order"
    )
    ray_tracing = bool(
        extended_input_data["simulationSettings"].get("ray_tracing")
    )
    air_absorption = bool(
        extended_input_data["simulationSettings"].get("air_absorption")
    )

    room = pra.Room(
        walls,
        fs=sampling_rate,
        max_order=image_source_order,
        ray_tracing=ray_tracing,
        air_absorption=air_absorption,
    )

    frequencies = input_data["results"][0]["frequencies"]

    room.octave_bands.base_freq = frequencies[0]
    room.n_octave_bands = len(frequencies)

    alpha, m_pyfar, _  = pf.constants.air_attenuation(
        20,
        frequencies,
        relative_humidity=50/1e2)
    m = np.squeeze(m_pyfar.freq)

    room.air_absorption = m

    # Add sources
    source_pos = get_source_positions(input_data)
    if source_pos.shape != (3,):
        raise ValueError("Source position must be a 3D coordinate.")
    room.add_source(source_pos)

    receiver_pos = get_receiver_positions(input_data)
    room.add_microphone(receiver_pos.T)

    print("setup_simulation: setup done!")

    return room


def export_rir_to_input(
        json_file_path: str | Path,
        rir: list[list[np.ndarray]]
    ) -> None:
    """Export the computed RIRs to the input data structure.

    Parameters
    ----------
    json_file_path : str | Path
        Path to the input JSON file.
    rir : list of list of np.ndarray
        Computed RIRs from pyroomacoustics.

    Returns
    -------
    None
    """

    with open(json_file_path, 'r') as f:
        import json
        input_data = json.load(f)

    input_data['results'][0]['responses'][0]['receiverResults'] = rir.tolist()
    input_data["results"][0]["percentage"] = 100

    with open(json_file_path, 'w') as f:
        json.dump(input_data, f, indent=4)


def export_rir_to_csv(
        json_file_path: str | Path,
    ) -> None:
    """Export the computed RIRs to a CSV file.

    Parameters
    ----------
    json_file_path : str | Path
        Path to the input JSON file.
    rir : list of list of np.ndarray
        Computed RIRs from pyroomacoustics.
    output_csv_path : str | Path
        Path to the output CSV file where the RIRs will be saved.

    Returns
    -------
    None
    """


    input_data = read_json_input(json_file_path)
    output_csv_path = str(json_file_path).replace('.json', '_pressure.csv')

    sampling_rate = input_data['simulationSettings']['sampling_rate']
    rir = input_data['results'][0]['responses'][0]['receiverResults']
    times = np.arange(len(rir)) / sampling_rate


    df = pd.DataFrame({'t': times, 'pressure': rir})
    df.to_csv(output_csv_path, index=False)


def calculate_room_acoustic_parameters(
        room_impulse_response: pf.Signal,
        bands: np.ndarray | list[float],
    ) -> dict[str, np.ndarray]:
    """Calculate room acoustic parameters from the RIR.

    Parameters
    ----------
    room_impulse_response : pf.TimeData
        The room impulse response.
    bands : np.ndarray or list of float
        The frequency bands for which to calculate acoustic parameters.
        This assumes octave bands.

    Returns
    -------
    parameters : dict
        A dictionary containing the calculated room acoustic parameters for
        each frequency band. The keys are the parameter names (e.g., 'RT60',
        'C50', 'D50'), and the values are lists containing the
        parameter values for each frequency band.
    """

    bands = np.asarray(bands, dtype=float)

    start_sample = pf.dsp.find_impulse_response_start(room_impulse_response)
    room_impulse_response_shifted = pf.dsp.time_shift(
        room_impulse_response,
        -start_sample,
        unit='samples',
    )

    rir_bands = pf.dsp.filter.fractional_octave_bands(
        room_impulse_response_shifted,
        num_fractions=1,
        frequency_range=[np.min(bands), np.max(bands)],
        order=6,
    )

    bands = pf.constants.fractional_octave_frequencies_nominal(
        num_fractions=1,
        frequency_range=(np.min(bands), np.max(bands)),
    )

    n_bands = rir_bands.cshape[0]
    EDT = np.zeros(n_bands, dtype=float)
    T_20 = np.zeros(n_bands, dtype=float)
    T_30 = np.zeros(n_bands, dtype=float)
    D_50 = np.zeros(n_bands, dtype=float)
    C_80 = np.zeros(n_bands, dtype=float)

    edc_bands = pyrato.edc.schroeder_integration(rir_bands, is_energy=False)
    EDT = pyrato.parameters.reverberation_time_linear_regression(
        edc_bands, T='EDT')
    T_20 = pyrato.parameters.reverberation_time_linear_regression(
        edc_bands, T='T20')
    T_30 = pyrato.parameters.reverberation_time_linear_regression(
        edc_bands, T='T30')

    # pyrato returns a range [0, 1] for D50,
    # CHORAS expects a percentage value in [0, 100]
    D_50 = pyrato.parameters.definition(edc_bands, early_time_limit=50) * 1e2

    C_80 = pyrato.parameters.clarity(edc_bands, early_time_limit=80)

    spl_t0_freq = 20*np.log10(pf.dsp.rms(rir_bands)/20e-6)

    T_s = center_time(edc_bands) * 1e3 # convert to ms

    def _normalize_nan_and_inf(arr: np.ndarray) -> np.ndarray:
        """Replace NaN and Inf values in the array with 0.0.

        The CHORAS frontend does not handle NaN and Inf values at the moment.

        Parameters
        ----------
        arr : np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The array with NaN and Inf values replaced by 0.0.
        """
        arr = np.where(np.isnan(arr), 0.0, arr)
        arr = np.where(np.isinf(arr), 0.0, arr)
        return arr

    parameters = {
        'bands': bands.tolist(),
        'edt': np.squeeze(_normalize_nan_and_inf(EDT)).tolist(),
        't20': np.squeeze(_normalize_nan_and_inf(T_20)).tolist(),
        't30': np.squeeze(_normalize_nan_and_inf(T_30)).tolist(),
        'd50': np.squeeze(_normalize_nan_and_inf(D_50)).tolist(),
        'c80': np.squeeze(_normalize_nan_and_inf(C_80)).tolist(),
        'ts': np.squeeze(_normalize_nan_and_inf(T_s)).tolist(),
        'spl_t0_freq': np.squeeze(
            _normalize_nan_and_inf(spl_t0_freq)).tolist(),
    }

    return parameters


def export_room_acoustic_parameters_to_json(
        input_data: dict,
        parameters: dict[str, list],
    ) -> dict:
    """Export the calculated room acoustic parameters to the input config.

    Parameters
    ----------
    input_data : dict
        The original input configuration data as a dictionary.
    parameters : dict
        A dictionary containing the calculated room acoustic parameters for
        each frequency band. The keys are the parameter names (e.g., 'T20',
        'C50', 'D50'), and the values are lists containing the
        parameter values for each frequency band.

    Returns
    -------
    updated_config : dict
        The updated input configuration dictionary with the room acoustic
        parameters added to the results section.
    """


    response_params = input_data['results'][0]['responses'][0]['parameters']
    for key in ['t20', 't30', 'edt', 'd50', 'c80', 'ts', 'spl_t0_freq']:
        response_params[key] = parameters[key]


    return input_data


# copy pasted from pyrato
def center_time(energy_decay_curve):
    r"""
    Calculate the room-acoustic center time (:math:`T_s`).

    The center time :math:`T_s` is the time of the centroid of the squared
    impulse response. It quantifies the balance between early and late
    sound energy [#isoTs]_.

    The parameter is defined as

    .. math::

        T_s =
        \frac{
            \displaystyle \int_{0}^{\infty} t \cdot p^2(t)\,\mathrm{d}t
        }{
            \displaystyle \int_{0}^{\infty} p^2(t)\,\mathrm{d}t
        }

    where :math:`p(t)` is the room impulse response sound pressure.

    Using the energy decay curve :math:`e(t)`, the parameter can be
    computed efficiently via the EDC identity as

    .. math::

        T_s =
        \frac{
            \displaystyle \int_{0}^{\infty} e(t)\,\mathrm{d}t
        }{
            e(0)
        }.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve of the room impulse response. The EDC must
        start at time zero and must have equal time spacing.

    Returns
    -------
    center_time : numpy.ndarray
        Center time (:math:`T_s`) in seconds,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#isoTs] ISO 3382, Acoustics — Measurement of the reverberation
        time of rooms with reference to other acoustical parameters.
    """

    if not isinstance(energy_decay_curve, pf.TimeData):
        raise TypeError(
            "energy_decay_curve must be a pyfar.TimeData or derived object.")

    if not np.isclose(energy_decay_curve.times[0], 0.0):
        raise ValueError("energy_decay_curve must start at time zero.")

    if np.any(energy_decay_curve.time[..., 0] == 0):
        raise ValueError(
            "Initial energy of energy_decay_curve must not be zero.")

    dt = np.diff(energy_decay_curve.times)
    if not np.allclose(dt, dt[0]):
        raise ValueError(
            "energy_decay_curve must have equal time spacing.")

    sampling_interval = dt[0]
    initial_energy = energy_decay_curve.time[..., 0]
    center_time = (
        np.nansum(energy_decay_curve.time, axis=-1)
        * sampling_interval
        / initial_energy
    )

    return center_time
