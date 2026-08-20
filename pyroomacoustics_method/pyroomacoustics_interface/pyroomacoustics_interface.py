"""Module implementing a CHORAS interface for pyroomacoustics.
"""
import json
import warnings
from pathlib import Path
from typing import Any

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

    @property
    def configuration(self) -> dict:
        """Return the current configuration as a dictionary.

        Returns
        -------
        config : dict
            The current configuration read from the input JSON file.
        """
        return self._read_config()

    @configuration.setter
    def configuration(self, config: dict) -> None:
        """Set the configuration by writing to the input JSON file.

        Parameters
        ----------
        config : dict
            The configuration to be written to the input JSON file.
        """
        self._write_config(config)

    def _get_result_data(self, result_idx: int = 0) -> dict:
        """Get the first result section from configuration.

        Returns
        -------
        dict
            The first result dictionary from the configuration.
        """
        return self.configuration['results'][result_idx]

    def _get_response_data(
            self,
            response_idx: int = 0,
            result_idx: int = 0,
        ) -> dict:
        """Get a specific response section from the first result.

        Parameters
        ----------
        response_idx : int, optional
            The index of the response to retrieve, by default 0.
        result_idx : int, optional
            The index of the result to retrieve the response from,
            by default 0.

        Returns
        -------
        dict
            The response dictionary at the specified index.
        """
        return self._get_result_data(
            result_idx=result_idx)['responses'][response_idx]

    def _get_response_parameters(
            self,
            response_idx: int = 0,
            result_idx: int = 0,
        ) -> dict:
        """Get the parameters section of a specific response.

        Parameters
        ----------
        response_idx : int, optional
            The index of the response, by default 0.
        result_idx : int, optional
            The index of the result to retrieve the response from,
            by default 0.

        Returns
        -------
        dict
            The parameters dictionary for the specified response.
        """
        return self._get_response_data(
            response_idx, result_idx)['parameters']

    def _get_simulation_settings(self) -> dict:
        """Get the simulation settings from configuration.

        Returns
        -------
        dict
            The simulation settings dictionary.
        """
        return self.configuration.get('simulationSettings', {})

    def _update_response_data(
            self,
            response_idx: int,
            key: str,
            value: Any
        ) -> None:
        """Update a specific field in a response and write to config.

        Parameters
        ----------
        response_idx : int
            The index of the response to update.
        key : str
            The key to update in the response.
        value : any
            The value to set.
        """
        config = self.configuration
        config['results'][0]['responses'][response_idx][key] = value
        self.configuration = config

    def _update_response_parameters(
            self,
            response_idx: int,
            parameters: dict
        ) -> None:
        """Update multiple parameters in a response and write to config.

        Parameters
        ----------
        response_idx : int
            The index of the response to update.
        parameters : dict
            Dictionary of parameter key-value pairs to update.
        """
        config = self.configuration
        response_params = config["results"][0]["responses"][response_idx][
            "parameters"
        ]
        for key, value in parameters.items():
            response_params[key] = value
        self.configuration = config

    def _update_result_data(self, key: str, value: Any) -> None:
        """Update a field in the result data and write to config.

        Parameters
        ----------
        key : str
            The key to update in the result data.
        value : any
            The value to set.
        """
        config = self.configuration
        config['results'][0][key] = value
        self.configuration = config

    def _write_progress(self, progress: int) -> None:
        self._update_result_data('percentage', progress)

    def _read_config(self) -> dict:
        with open(self.input_json_path, 'r') as f:
            input_data = json.load(f)

        return input_data

    def _write_config(self, config: dict) -> None:
        with open(self.input_json_path, 'w') as f:
            json.dump(config, f, indent=4)

    def run_simulation(self) -> None:
        """Execute the simulation and export results to the configuration file.
        """

        print("pyroomacoustics_method: starting simulation")

        pra.constants.set(
            'c',
            float(self._get_simulation_settings().get(
                "speed_of_sound", 343.0)),
        )

        walls = self.import_room_geometry()
        self._write_progress(20)

        simulation_setup = self.setup_simulation(walls)
        self._write_progress(30)

        # Compute the RIRs
        simulation_setup.compute_rir()
        self._write_progress(80)

        # Get the RIRs for the first source and first microphone
        rir = simulation_setup.rir[0][0]

        # Export the RIRs to the input data structure
        self._export_rir_to_input(rir)
        self._write_progress(90)

        self.export_rir_to_csv()
        self._write_progress(95)

        bands = self._get_result_data()['frequencies']

        rir_signal = pf.Signal(rir, simulation_setup.fs)

        # pyroomacoustics uses a fractional delay filter to create
        # sub-sample accurate RIRs. The rir is always delayed by
        # half the length of the fractional delay filter.
        # To compensate for this and ensure physically plausible
        # propagation delays, the RIR is shifted back.
        pra_frac_delay = pra.constants.get("frac_delay_length") // 2

        # Cyclic time shift. May introduce non-causal components
        # if source and receiver are very close.
        rir_signal = pf.dsp.time_shift(
            rir_signal, -pra_frac_delay, unit="samples",
        )

        rap = calculate_room_acoustic_parameters(
            rir_signal,
            bands=bands,
        )

        self._export_room_acoustic_parameters_to_json(rap)
        self._write_progress(100)

        print("pyroomacoustics_method: simulation done!")

    def import_room_geometry(self) -> list[pra.Wall]:
        """Import room geometry and absorption coefficients.

        The geometry is read from a .geo file specified in the JSON input file.
        The absorption coefficients are directly read from the JSON file.

        This method reads geometry and absorption coefficients from the
        configuration referenced by ``self.input_json_path``.

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

        # initialize gmsh and load the geometry file
        gmsh.initialize()
        try:
            walls = self._import_room_geometry()
        finally:
            gmsh.finalize()

        return walls

    def _import_room_geometry(self) -> list[pra.Wall]:
        """Private import class for geometry and boundary conditions.

        This private function ensures that gmsh is properly finalized after
        geometry import, even if an error occurs during the import process.

        The public function `import_room_geometry` is responsible for
        initializing and finalizing gmsh.

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
        input_data = self.configuration

        frequencies = self._get_result_data()['frequencies']
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

            # This is way to set scattering coefficients in pyroomacoustics
            # until the setting becomes available through the materials
            # interface in the frontend.
            scattering_coefficient = self._get_simulation_settings().get(
                'global_scattering_coefficient', 0.1)
            scattering_coefficients = np.ones_like(
                absorption_coeffs) * scattering_coefficient

            material = pra.Material(
                energy_absorption={
                    'description': surface_name,
                    'center_freqs': frequencies,
                    'coeffs': absorption_coeffs,
                },
                scattering={
                    'description': surface_name,
                    'center_freqs': frequencies,
                    'coeffs': scattering_coefficients,
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


    def get_source_positions(self) -> np.ndarray:
        """Extract source positions from input data.

        Returns
        -------
        source_positions : np.ndarray
            Array of source positions with shape (3,).
        """
        result_data = self._get_result_data()
        return np.array([
            result_data['sourceX'],
            result_data['sourceY'],
            result_data['sourceZ'],
        ])


    def get_receiver_positions(self) -> np.ndarray:
        """Extract receiver positions from input data.

        Returns
        -------
        receiver_positions : np.ndarray
            Array of receiver positions with shape (n_receivers, 3).
        """
        result_data = self._get_result_data()

        num_receivers = len(result_data['responses'])
        response_section = result_data['responses']
        receiver_pos = np.zeros((num_receivers, 3), dtype=float)

        for i in range(num_receivers):
            receiver_pos[i] = np.array([
                response_section[i]['x'],
                response_section[i]['y'],
                response_section[i]['z']
            ])

        return receiver_pos


    def set_default_simulation_settings(self) -> dict:
        """Set default simulation settings if not provided in input data.

        Returns
        -------
        input_data : dict
            The input data dictionary with default simulation settings added if
            they were not already present.
        """
        input_data = self.configuration

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

        if 'air_absorption' not in settings:
            settings['air_absorption'] = True
            warnings.warn(
                "Air absorption setting not specified. "
                "Defaulting to True.",
                stacklevel=2
            )

        if 'ray_tracer_number_of_rays' not in settings:
            settings['ray_tracer_number_of_rays'] = 10_000
            warnings.warn(
                "Number of rays not specified. "
                "Defaulting to 10000.",
                stacklevel=2,
            )

        if 'ray_tracer_detector_radius' not in settings:
            settings['ray_tracer_detector_radius'] = 0.5
            warnings.warn(
                "Detector radius not specified. "
                "Defaulting to 0.5 m.",
                stacklevel=2,
            )

        if 'ray_tracer_time_threshold' not in settings:
            settings['ray_tracer_time_threshold'] = 10.
            warnings.warn(
                "Ray tracer time threshold not specified. "
                "Defaulting to 10 seconds",
                stacklevel=2,
            )

        if 'ray_tracer_energy_threshold' not in settings:
            settings['ray_tracer_energy_threshold'] = -70
            warnings.warn(
                "Ray tracer energy threshold not specified. "
                "Defaulting to -70 dB re 1, i.e. 1e-7 on a linear scale.",
                stacklevel=2,
            )

        return input_data


    def setup_simulation(
            self,
            walls: list[pra.Wall]
        ) -> pra.Room:
        """Set up the pyroomacoustics simulation.

        Parameters
        ----------
        walls : list of pyroomacoustics.Wall
            List of walls defining the room geometry and boundary conditions

        Returns
        -------
        room : pyroomacoustics.Room
            The configured pyroomacoustics Room object
        """

        print("setup_simulation: setting up simulation")

        # Ensure default settings are set
        self.set_default_simulation_settings()
        settings = self._get_simulation_settings()

        sampling_rate = settings.get("sampling_rate")
        image_source_order = settings.get("image_source_order")
        num_rays = int(settings.get('ray_tracer_number_of_rays'))
        detector_radius = float(settings.get('ray_tracer_detector_radius'))
        ray_tracer_energy_threshold_dB = float(
            settings.get('ray_tracer_energy_threshold'))
        ray_tracer_energy_threshold = 10**(ray_tracer_energy_threshold_dB/10)
        ray_tracer_time_threshold = float(
            settings.get('ray_tracer_time_threshold'))

        air_absorption = bool(settings.get("air_absorption"))

        room = pra.Room(
            walls,
            fs=sampling_rate,
            max_order=image_source_order,
            ray_tracing=True,
            air_absorption=air_absorption,
        )

        frequencies = self._get_result_data()["frequencies"]

        room.octave_bands.base_freq = frequencies[0]
        room.n_octave_bands = len(frequencies)

        # This avoids a bug in pyroomacoustics which prevents setting the
        # air attenuation for frequency dependent data
        _, m_pyfar, _  = pf.constants.air_attenuation(
            20,
            frequencies,
            relative_humidity=50/1e2)
        room.air_absorption = np.squeeze(m_pyfar.freq)

        # Add sources
        source_pos = self.get_source_positions()
        if source_pos.shape != (3,):
            raise ValueError("Source position must be a 3D coordinate.")
        room.add_source(source_pos)

        receiver_pos = self.get_receiver_positions()
        room.add_microphone(receiver_pos.T)

        room.set_ray_tracing(
            n_rays=num_rays,
            receiver_radius=detector_radius,
            energy_thres=ray_tracer_energy_threshold,
            time_thres=ray_tracer_time_threshold,
        )

        print("setup_simulation: setup done!")

        return room


    def _export_rir_to_input(
            self,
            rir: np.ndarray,
        ) -> None:
        """Export the computed RIRs to the input data structure.

        Parameters
        ----------
        rir : np.ndarray
            Computed RIRs from pyroomacoustics.

        """

        self._update_response_data(0, 'receiverResults', rir.tolist())

    def _export_room_acoustic_parameters_to_json(
            self,
            parameters: dict[str, list],
        ) -> None:
        """Export the calculated room acoustic parameters to the input config.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the calculated room acoustic parameters for
            each frequency band. The keys are the parameter names (e.g., 'T20',
            'C50', 'D50'), and the values are lists containing the
            parameter values for each frequency band.

        """

        # Prepare parameters dictionary with only the keys we want to update
        params_to_update = {
            key: parameters[key]
            for key in ['t20', 't30', 'edt', 'd50', 'c80', 'ts', 'spl_t0_freq']
        }
        self._update_response_parameters(0, params_to_update)

    def export_rir_to_csv(self) -> None:
        """Export the computed RIRs to a CSV file.

        This is a legacy helper function which will be obsolete in the future.

        The CSV is written next to the input JSON file using the same basename
        and the suffix ``_pressure.csv``.

        Returns
        -------
        None

        """
        json_file_path = self.input_json_path
        output_csv_path = str(json_file_path).replace('.json', '_pressure.csv')

        settings = self._get_simulation_settings()
        sampling_rate = settings['sampling_rate']

        response_data = self._get_response_data(0)
        rir = response_data['receiverResults']
        times = np.arange(len(rir)) / sampling_rate

        df = pd.DataFrame({'t': times, 'pressure': rir})
        df.to_csv(output_csv_path, index=False)

        warnings.warn(
            "export_rir_to_csv is a legacy helper function and will be "
            "obsolete in the future. Consider using the RIRs from the "
            "input JSON file directly.",
            stacklevel=2)


def calculate_room_acoustic_parameters(
        room_impulse_response: pf.Signal,
        bands: np.ndarray | list[float],
    ) -> dict[str, list[float]]:
    """Calculate room acoustic parameters from the RIR.

    Parameters
    ----------
    room_impulse_response : pf.Signal
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

    rir_bands = pf.dsp.filter.fractional_octave_bands(
        room_impulse_response,
        num_fractions=1,
        frequency_range=[np.min(bands), np.max(bands)],
        order=6,
    )

    rir_bands_shifted = pf.dsp.time_shift(
        rir_bands,
        -start_sample,
        unit='samples',
        mode='linear',
        pad_value=np.nan,
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

    edc_bands = pyrato.edc.schroeder_integration(
        rir_bands_shifted,
        is_energy=False
    )

    edc_bands = pf.dsp.normalize(edc_bands, nan_policy='omit')

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
