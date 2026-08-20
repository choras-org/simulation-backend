"""Module implementing a CHORAS interface for misuka."""

import json
from pathlib import Path
import gmsh
import meshio
import mitsuba as mi
import numpy as np
import pyfar as pf
import pyrato as pr
import pandas as pd
import os
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

mi.set_variant("llvm_ad_acoustic")

from .definition import SimulationMethod
from . import parametric as pra

class misukaMethod(SimulationMethod):
    """Interface class to run the misuka method.

    The class implements method to run the calculations for the
    misuka simulation method. All required configuration parameters
    are expected to be provided in the input JSON file passed during
    initialization.

    """

    def __init__(self, input_json_path: str | Path | None = None):
        """Initialize the misuka method interface for the given JSON file."""
        super().__init__(input_json_path)

    def run_simulation(self) -> None:
        """Run the simulation using the JSON file provided at initialization."""
        self._misuka_method()


    def _misuka_method(self) -> None:
        """
        Run the misuka simulation method using the input JSON file.

        Reads the simulation settings from the user configuration JSON file
        (self.input_json_path), sets up the scene, and performs the simulation
        for each source-receiver pair. Results are written back to the input
        JSON file, including the EDC and acoustic parameters for each
        source-receiver pair.

        Raises
        ------
        KeyError
        - If required keys are missing in the input JSON file.
        
        ValueError
        - If a receiver is at the same position as the source.
        - If the position of a source is the same as a receiver.
        """

        try:
            # Load the input JSON file
            with open(self.input_json_path, "r") as json_file:
                result_container = json.load(json_file)
        except KeyError as e:
            raise KeyError(f"Could not fetch the simulation settings for misuka T.T: {e}")
        
        try:
            simulation_settings = result_container["simulationSettings"]
            speed_of_sound = simulation_settings["speed_of_sound"]
            max_time = simulation_settings["max_time"]
            time_bins = simulation_settings["sampling_rate"]
            rpf = simulation_settings["rays_per_frequency"]
            n_receivers = len(result_container["results"][0]["responses"])
            frequencies = result_container["results"][0]["frequencies"]
            msh_path = result_container["msh_path"]
            absorption_map = result_container["absorption_coefficients"]
            scattering_str = simulation_settings["scattering_coefficients"]
        except KeyError as e:
            raise KeyError(f"Missing required key in the input JSON file: {e}")

        target_sampling_rate = time_bins

        logger.info(f'Using variant: {mi.variant()}')
        logger.info(f'Rendered frequencies: {frequencies}')
        logger.info(f'Input value for rays per frequency: {rpf}')
        logger.info(f'Input value for maximum response length: {max_time}')
        logger.info(f'Input value for time_bins: {time_bins}')
        logger.info(f'Input value for speed of sound: {speed_of_sound}')
        logger.info(f'Input values for absorption: {absorption_map}')
        
        frequency_range = (float(np.min(frequencies)), float(np.max(frequencies)))
        f_center, f_lower, f_upper = pf.constants.fractional_octave_frequencies_exact(1, frequency_range)
        assert np.all(np.abs(f_center-frequencies)/frequencies < 1e-2), "Frequency mismatch between input frequencies and pyrato's fractional octave frequencies"
        frequencies_misuka = ", ".join(str(f) for f in frequencies)

        source_coords = [ #TODO: when multiple soundsources are implemented adjust like receiver_coords
            [
                result_container["results"][0]["sourceX"],
                result_container["results"][0]["sourceY"],
                result_container["results"][0]["sourceZ"],
            ]
        ]
        receiver_coords = [
            [
                result_container["results"][0]["responses"][i]["x"],
                result_container["results"][0]["responses"][i]["y"],
                result_container["results"][0]["responses"][i]["z"],
            ]
            for i in range(n_receivers)
        ]
        logger.info(f'Source coordinates: {source_coords}\n Reciever coordinates: {receiver_coords}')

        # updating percentage of progress
        set_progress_and_save(25, result_container, self.input_json_path)

        # set up the integrator
        integrator_acoustic = mi.load_dict(
            {
                "type": "acoustic_path",
                "speed_of_sound": speed_of_sound,
                "max_depth": -1,  # maximum path depth (-1 = no limit, 1 = direct sound only, 2 = up to 1 reflection, etc.)
                "max_time": max_time,  # maximum propagation time in seconds
            }
        )

        logger.info(f'Integrator Setup: {integrator_acoustic}')

        # reconstructing filter bank for RIR synthesis from the per-band ETC envelope;
        # only depends on frequencies/target_sampling_rate, so build it once
        reflection_filter_bank, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
            signal=None,
            sampling_rate=target_sampling_rate,
            num_fractions=1,
            frequency_range=[float(frequencies[0] * 2 ** (-1 / 2)), float(frequencies[-1] * 2 ** (1 / 2))],
        )

        # Extract physical group information from gmsh; also estimates the room
        # volume from the same open gmsh session, avoiding a separate
        # initialize/finalize cycle just for that.
        scene_dict, room_volume = build_scene_from_gmsh(
            msh_path,
            absorption_map,
            frequencies,
            scattering_str,
        )
        logger.info(f'Estimated room volume: {room_volume} m^3')

        for i_src, source_coord in enumerate(source_coords):
            for i_rec, receiver_coord in enumerate(receiver_coords):

                microphone_direction = (np.asarray(source_coord) - np.asarray(receiver_coord))
                distance_to_source = np.linalg.norm(microphone_direction.astype(float))
                if distance_to_source == 0:
                    raise ValueError(f"Receiver {i_rec} is at the same position as the source.")
                
                microphone_direction = (microphone_direction / distance_to_source).tolist()

                microphone = mi.load_dict({
                    "type": "microphone",
                    "origin": receiver_coord,
                    "direction": microphone_direction,
                    "film": {
                        "type": "tape",
                        "frequencies": frequencies_misuka,  # rendered frequencies
                        "time_bins": time_bins,  # number of time bins
                        "component_format": "float32",
                    },
                })

                logger.info(f'Loaded Microphone for reciever at {receiver_coord} and source at {source_coord}: {microphone}')

                scene_dict["emitter"] = {
                    "type": "sphere",
                    "radius": 0.1,
                    "center": source_coord,
                    "emitter": {
                        'type': 'area',
                        'radiance': {
                            'type': 'uniform',
                            'value': 30.0 # not scaled yet but needs some value to successfully load scene
                        }
                    }
                }

                scene = mi.load_dict(scene_dict)

                logger.info(f'Loaded Scene for reciever at {receiver_coord} and source at {source_coord}: {scene}')

                #TODO: use sound_power_W for scaling the etc
                etc = mi.render(scene, sensor=microphone, integrator=integrator_acoustic,spp=rpf)
                etc_signal = pf.Signal(etc.numpy().T, sampling_rate=time_bins / max_time, domain='time')
                etc_signal.time /= np.max(np.abs(etc_signal.time))

                edc = pr.edc.schroeder_integration(etc_signal, is_energy=True)
                edc_normalized = pf.dsp.normalize(edc)
                with np.errstate(divide='ignore', invalid='ignore'):
                    edc_normalized_db = 10*np.log10(edc_normalized.time/1e-12)
                edc_normalized_db = np.squeeze(edc_normalized_db, axis=0)
                edc_normalized_db = finite_array(edc_normalized_db, nan=0.0, neginf=-100, posinf=100)

                # Synthesize a broadband RIR from the per-band ETC envelope, following
                # the reflection-sequence method: reflections arrive as a Poisson
                # process (rate set by the room volume), get random phase, are
                # split into octave bands and shaped by the (resampled) ETC envelope.
                n_samples = int(np.floor(etc_signal.times[-1] * target_sampling_rate))
                times_target = np.arange(n_samples) / target_sampling_rate
                etc_time = np.squeeze(etc_signal.time, axis=0)
                envelope_target = _interpolate_envelope(
                    etc_signal.times, np.clip(etc_time, 0, None), times_target,
                )

                times_of_arrival = pra.time_of_arrival_poisson_process(
                    room_volume, etc_signal.times,
                    speed_of_sound=speed_of_sound,
                    reflection_rate_limit=20e3,
                )
                reflection_sequence = pra.random_reflection_sequence(
                    times_of_arrival,
                    n_samples=n_samples,
                    sampling_rate=target_sampling_rate,
                    distribution='binary',  # misuka only needs random phase, not random amplitude
                )
                reflection_sequence_bands = reflection_filter_bank.process(reflection_sequence)[1:-1]
                imp_tot = np.sum(np.squeeze(reflection_sequence_bands.time) * np.sqrt(envelope_target), axis=0)

                # store RIR in results
                result_container["results"][0]["responses"][i_rec]["receiverResults"] = imp_tot.tolist()

                # store EDC in results
                result_container["results"][0]["responses"][i_rec]["receiverResultsEDC"] = [
                    {
                        "data": (edc_normalized_db[i_frq]).tolist(),
                        "t": etc_signal.times.tolist(),
                        "frequency": frequencies[i_frq],
                        "type": "edc",
                    }
                    for i_frq in range(len(frequencies))
                ]

                with np.errstate(divide='ignore', invalid='ignore'):
                    t20 = np.squeeze(pr.parameters.reverberation_time_linear_regression(edc, 'T20'))
                    t20 = finite_array(t20, nan=0.0, neginf=0.0, posinf=0.0)
                    result_container["results"][0]["responses"][i_rec]["parameters"]['t20'] = t20.tolist()

                    t30 = np.squeeze(pr.parameters.reverberation_time_linear_regression(edc, 'T30'))
                    t30 = finite_array(t30, nan=0.0, neginf=0.0, posinf=0.0)
                    result_container["results"][0]["responses"][i_rec]["parameters"]['t30'] = t30.tolist()

                    c80 = np.squeeze(pr.parameters.clarity(edc, 80))
                    c80 = finite_array(c80, nan=0.0, neginf=0.0, posinf=0.0)
                    result_container["results"][0]["responses"][i_rec]["parameters"]['c80'] = c80.tolist()

                    d50 = np.squeeze(pr.parameters.definition(edc, 50)) * 100
                    d50 = finite_array(d50, nan=0.0, neginf=0.0, posinf=0.0)
                    result_container["results"][0]["responses"][i_rec]["parameters"]['d50'] = d50.tolist()

                    ts = center_time(edc)*1000 # in ms TODO replace by pyrato 1.1.0 version
                    result_container["results"][0]["responses"][i_rec]["parameters"]['ts'] = np.squeeze(ts).tolist()

                    spl = np.squeeze(10*np.log10(edc.time[..., 0]/1e-12))
                    spl = finite_array(spl, nan=0.0, neginf=0.0, posinf=0.0)
                    result_container["results"][0]["responses"][i_rec]["parameters"]['spl_t0_freq'] = spl.tolist()

                    edt = np.squeeze(pr.parameters.reverberation_time_linear_regression(edc, 'EDT'))
                    edt = finite_array(edt, nan=0.0, neginf=0.0, posinf=0.0)
                    result_container["results"][0]["responses"][i_rec]["parameters"]['edt'] = edt.tolist()

            # update progress for each source in even steps to ~90%
            set_progress_and_save(35 + int(55/(n_receivers+1) * (i_src + 1)), result_container, self.input_json_path)
        
        # set to 100%
        set_progress_and_save(100, result_container, self.input_json_path)


def build_scene_from_gmsh(
    msh_path: str | Path,
    absorption_map: dict,
    frequencies: list,
    scattering_str: str,
) -> tuple[dict, float]:
    """Initialize Gmsh, export physical surfaces to PLY and populate the Mitsuba scene.

    This function loads a mesh file using Gmsh, extracts all physical surface groups,
    exports each surface as a separate PLY file, and populates a fresh Mitsuba scene
    dictionary with the corresponding material properties (BSDF and geometry). It
    also estimates the room volume from the same open Gmsh session. Gmsh is
    guaranteed to be finalized even if an exception occurs during processing.

    Parameters
    ----------
    msh_path : str | Path
        Path to the Gmsh mesh file (.msh format) containing physical groups
        defining surfaces.
    absorption_map : dict
        Mapping of physical surface names to absorption coefficient strings.
        Each value should be a comma-separated string of 5 absorption coefficients
        (one per frequency band). Surfaces not in the map default to
        "0.1, 0.1, 0.1, 0.1, 0.1".
    frequencies : list
        List of frequency values (in Hz) for which to define the acoustic spectrum.
        Example: [125, 250, 500, 1000, 2000, 4000].
    scattering_str : str
        Comma-separated string of scattering coefficients (one per frequency band).
        Applied uniformly to all surfaces.

    Returns
    -------
    dict
        Populated Mitsuba scene dictionary containing BSDF definitions and PLY geometry
        references for all physical surfaces. The dictionary includes keys like:
        `bsdf_0`, `bsdf_1`, ... for material properties and
        `shape_0`, `shape_1`, ... for geometry references.
    float
        Volume of the room in the same units as the mesh coordinates, computed
        from the closed surface mesh (see `compute_mesh_volume`).

    Raises
    ------
    RuntimeError:
    - If Gmsh fails to initialize, open the mesh file, extract physical groups,
    or export surfaces. The original exception is chained for debugging.

    Notes
    -----
    - PLY files are exported to the same directory as the input mesh file with
      naming convention: `surface_{physical_group_name}.ply`.
    - All physical groups of dimension 2 (surfaces) are processed. Groups with
      other dimensions are ignored.
    - Gmsh is finalized in a finally block to ensure cleanup even on error.
    - Only triangular mesh faces (3 nodes per element) are exported to PLY.

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> msh_file = Path("room.msh")
    >>> absorption = {"Wall": "0.2, 0.2, 0.3, 0.3, 0.4, 0.4", "Floor": "0.1, 0.1, 0.15, 0.15, 0.2, 0.2"}
    >>> freq = [125, 250, 500, 1000, 2000, 4000]
    >>> scatter_spec = [(125, 0.05), (250, 0.05), (500, 0.1), (1000, 0.1), (2000, 0.15), (4000, 0.15)]
    >>> result, volume = build_scene_from_gmsh(msh_file, absorption, freq, scatter_spec)
    >>> "bsdf_0" in result
    True
    """
    scene_dict = {
        "type": "scene",
    }
    try:
        gmsh.initialize()
        gmsh.open(str(msh_path))
        room_volume = volume_from_msh(msh_path)
        surfaces = gmsh.model.getPhysicalGroups(2)

        ply_dir = Path(msh_path).parent

        #TODO: when scattering is surface material dependent move this into for loop
        scattering_spectrum = create_frequency_value_pairs(
            frequencies,
            scattering_str,
            num_tuples=len(frequencies),
        )

        for i, (dim, tag) in enumerate(surfaces):
            surface_name = gmsh.model.getPhysicalName(dim, tag)
            absorption_str = absorption_map.get(surface_name, "0.1, 0.1, 0.1, 0.1, 0.1, 0.1")
            entity_tags = [
                (dim, entity_tag)
                for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            ]

            ply_file = str(ply_dir / f"surface_{surface_name}.ply")
            export_physical_surface_to_ply(surface_name, entity_tags, ply_file)
            absorption_spectrum = create_frequency_value_pairs(
                frequencies,
                absorption_str,
                num_tuples=len(frequencies),
            )

            scene_dict[f"bsdf_{i}"] = {
                "type": "twosided",
                "id": f"bsdf_{i}",
                "bsdf": {
                    "type": "acousticbsdf",
                    "absorption": {
                        "type": "spectrum",
                        "value": absorption_spectrum,
                    },
                    "scattering": {
                        "type": "spectrum",
                        "value": scattering_spectrum,
                    },
                    "specular_lobe_width": 0.001,
                },
            }

            scene_dict[f"shape_{i}"] = {
                "type": "ply",
                "filename": ply_file,
                "face_normals": True,
                "bsdf": {"type": "ref", "id": f"bsdf_{i}"},
            }

        return scene_dict, room_volume
    except Exception as exc:
        logger.exception("Failed to build Mitsuba scene from geometry '%s'.", msh_path)
        raise RuntimeError(f"Failed to build Mitsuba scene from geometry '{msh_path}'.") from exc
    finally:
        try:
            gmsh.finalize()
        except Exception:
            logger.debug("Gmsh finalize skipped because it was not initialized.")

def set_progress_and_save(percentage: int, result_container: dict, json_file_path: str | Path):
    """Update the simulation progress state and persist it to the JSON result file.

    The function writes the current percentage into the first result entry,
    removes any outdated percentage list entries, and saves the updated
    result container back to the given JSON file.

    Parameters
    ----------
    percentage : int
        Current progress percentage (0-100).
    result_container : dict
        Dictionary containing the simulation results and settings.
    json_file_path : str | Path
        Path to the JSON file where the updated result container will be saved.

    Notes
    -----
    - The function updates the ``percentage`` field in the first result of the result container.
    - It can also be used to persist any other updates made to the container data.
    """
    result = result_container["results"][0]
    result["percentage"] = percentage
    result.pop("percentages", None)
    # Save the updated JSON
    with open(json_file_path, "w") as json_output:
        json_output.write(json.dumps(result_container,
                                     indent=4,
                                     allow_nan=False
                                     ))

def create_frequency_value_pairs(frequencies: list, val_str: str, num_tuples: int = 5) -> list:
    """Create a configurable number of frequency-value tuples for Mitsuba spectra.

    This function takes a list of frequencies and a comma-separated string of values,
    validates the input on the number of values, their range and same size as the frequency list, 
    and returns a list of (frequency, value) tuples suitable for Mitsuba's spectrum format. The function
    is used e.g. for scattering and absorption coefficients.

    Parameters
    ----------
    frequencies : list
        List of frequency values [125, 250, 500, ...]
    val_str : str
        A comma-separated string of values.
    num_tuples : int, optional
        Number of frequency-value tuples to create. The default is 5.

    Returns
    -------
    list
        List of (frequency, value) tuples for Mitsuba spectrum.

    Raises
    ------
    ValueError:
    - if ``num_tuples`` is not a positive integer
    - If the number of values in the string does not match ``num_tuples``
    - If any value is outside the range [0, 1]
    - If Value string is empty or not convertible to float
    - If the number of frequencies does not match ``num_tuples``

    Examples
    --------
    >>> create_frequency_value_pairs([125, 250, 500, 1000, 2000], "0.1, 0.2, 0.3, 0.4, 0.5")
    [(125, 0.1), (250, 0.2), (500, 0.3), (1000, 0.4), (2000, 0.5)]
    >>> create_frequency_value_pairs([125, 250, 500, 1000, 2000], "0.1, 0.2, 0.3, 0.4, 1.5")
    ValueError: Number of Values in String must match the requested number of tuples and each must be between 0 and 1.
    >>> create_frequency_value_pairs([125, 250, 500, 1000, 2000], "0.1, 0.2, 0.3, 0.4, 0.5, 0.5", num_tuples=6)
    [(125, 0.1), (250, 0.2), (500, 0.3), (1000, 0.4), (2000, 0.5)]
    """
    try:
        if not isinstance(num_tuples, int) or num_tuples <= 0:
            raise ValueError("Number of tuples must be a positive integer.")
        if len(frequencies) != num_tuples:
            raise ValueError("Number of frequencies must match the requested number of tuples.")
        if not val_str:
            raise ValueError("Value string is empty.")

        vals = [float(val.strip()) for val in val_str.split(",")]
        if len(vals) != num_tuples or any(val < 0 or val > 1 for val in vals):
            raise ValueError(
                "Number of Values in String must match the requested number of tuples and each must be between 0 and 1."
            )
    except ValueError as e:
        logger.error(f"Error parsing Values: {e}")
        raise ValueError(f"Invalid values: {val_str}") from e

    return list(zip(frequencies[:num_tuples], vals))

def export_physical_surface_to_ply(surface_name: str, entity_tags, ply_path: str | Path) -> Path:
    """Export only the triangular mesh faces of one physical surface to PLY.

    This function extracts the triangular mesh faces corresponding to a specific physical surface (specified by name and entity tags)
    from a Gmsh model and exports them to a PLY file. It ensures that only triangular elements
    are included, and raises an error if no triangular faces are found.

    Parameters
    ----------
    surface_name : str 
        Name of the physical surface to export
    entity_tags : list of tuples
        List of (dimension, entity_tag) tuples for the physical surface
    ply_path : str | Path
        Path to the output PLY file

    Returns
    -------
    Path
        Path to the exported PLY file
    
    Raises
    ------
    ValueError:
    - If the physical surface contains no triangular mesh faces.
    """
    ply_path = Path(ply_path)
    node_tags_all, coords_all, _ = gmsh.model.mesh.getNodes()
    coords_all = coords_all.reshape((len(node_tags_all), 3))
    node_tag_to_index = {
        int(node_tag): index for index, node_tag in enumerate(node_tags_all)
    }

    surface_faces = []
    for dim, entity_tag in entity_tags:
        element_types, _, element_node_tags = gmsh.model.mesh.getElements(
            dim, entity_tag
        )
        for element_type, node_tags in zip(element_types, element_node_tags):
            (
                element_name,
                element_dim,
                _order,
                num_nodes,
                _local_node_coord,
                _num_primary_nodes,
            ) = gmsh.model.mesh.getElementProperties(element_type)
            if element_dim != 2 or num_nodes != 3:
                continue

            faces = np.asarray(node_tags, dtype=np.int64).reshape((-1, num_nodes))
            surface_faces.append(faces)

    if not surface_faces:
        raise ValueError(
            f"Physical surface '{surface_name}' contains no triangular mesh faces."
        )

    surface_faces = np.concatenate(surface_faces, axis=0)
    used_node_tags = np.unique(surface_faces)
    used_coords = np.array(
        [coords_all[node_tag_to_index[int(node_tag)]] for node_tag in used_node_tags]
    )
    compact_node_index = {
        int(node_tag): index for index, node_tag in enumerate(used_node_tags)
    }
    compact_faces = np.vectorize(lambda node_tag: compact_node_index[int(node_tag)])(
        surface_faces
    )

    meshio.Mesh(
        points=used_coords,
        cells=[("triangle", compact_faces)],
    ).write(str(ply_path))
    return ply_path

# copy pasted from pyrato
def center_time(energy_decay_curve: pf.TimeData) -> np.ndarray:
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

def _interpolate_envelope(
    times_source: np.ndarray, envelope_source: np.ndarray, times_target: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a per-band, non-negative energy envelope to new time samples.

    Uses linear interpolation (rather than e.g. spline) to avoid negative values.

    Parameters
    ----------
    times_source : np.ndarray, shape (n_times_source,)
        Time samples at which `envelope_source` is defined.
    envelope_source : np.ndarray, shape (n_bands, n_times_source)
        Per-band energy envelope.
    times_target : np.ndarray, shape (n_times_target,)
        Time samples to interpolate to.

    Returns
    -------
    np.ndarray, shape (n_bands, n_times_target)
        Interpolated, non-negative envelope.
    """
    envelope_target = np.zeros((envelope_source.shape[0], times_target.shape[0]))
    for i in range(envelope_source.shape[0]):
        envelope_target[i] = np.interp(times_target, times_source, envelope_source[i])
    return np.clip(envelope_target, 0, None)

def finite_array(values: np.ndarray, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> np.ndarray:
    """
    Replace NaN, positive infinity, and negative infinity in an array with specified finite values.

    Parameters
    ----------
    values : array-like
        Input array.
    nan : float, optional
        Value to replace NaN with. Default is 0.0.
    posinf : float, optional
        Value to replace positive infinity with. Default is 0.0.
    neginf : float, optional
        Value to replace negative infinity with. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array with specified values replacing NaN and infinity.
    """
    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )

def compute_mesh_volume(vertices, faces):
    """
    Compute the volume of a closed triangulated mesh.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        XYZ coordinates of mesh vertices.
    faces : array-like, shape (M, 3)
        Triangle faces as 0-based indices into ``vertices``.

    Returns
    -------
    float
        Volume of the mesh in the same units as ``vertices``.

    Notes
    -----
    Uses the signed tetrahedron (divergence theorem) method. Each face
    contributes a signed tetrahedron volume from the origin::

        V = (1/6) * |sum_i  v0_i · (v1_i × v2_i)|

    The mesh must be closed (watertight) and faces must be consistently
    oriented (all outward- or all inward-facing normals) for the result
    to be correct.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.intp)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    # Scalar triple product: v0 · (v1 × v2), vectorized over all faces
    signed_volumes = np.einsum("ij,ij->i", v0, np.cross(v1, v2))

    return float(np.abs(signed_volumes.sum()) / 6.0)

def read_gmsh_mesh(path):
    """
    Read a Gmsh .msh file and return vertices and triangular faces.

    Parameters
    ----------
    path : str or path-like
        Path to a Gmsh .msh file.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3)
        XYZ coordinates of all mesh vertices.
    faces : np.ndarray, shape (M, 3)
        Triangle faces as 0-based indices into ``vertices``.
        Other element types (quads, tetrahedra) are ignored.

    Notes
    -----
    Requires gmsh to be initialized before calling. Leaves the opened
    model in gmsh's session; call ``gmsh.clear()`` afterwards if needed.
    """
    gmsh.open(str(path))

    _, coords, _ = gmsh.model.mesh.getNodes()
    vertices = coords.reshape(-1, 3)

    # careful: triangles only (element type 2); quads/tets ignored
    _, node_tags = gmsh.model.mesh.getElementsByType(2)
    faces = node_tags.reshape(-1, 3) - 1  # gmsh node tags are 1-based

    return vertices, faces.astype(np.intp)

def volume_from_msh(path):
    """
    Read a Gmsh .msh file and return the volume of the closed surface mesh.

    Parameters
    ----------
    path : str or path-like
        Path to a Gmsh .msh file containing a closed (watertight) surface mesh.

    Returns
    -------
    float
        Volume of the mesh in the same units as the mesh coordinates.

    Notes
    -----
    Convenience wrapper around `read_gmsh_mesh` and `compute_mesh_volume`.
    See `compute_mesh_volume` for correctness requirements on mesh orientation.
    """
    return compute_mesh_volume(*read_gmsh_mesh(path))