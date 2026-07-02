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

mi.set_variant("cuda_acoustic", "llvm_acoustic", "cuda_ad_acoustic", "llvm_ad_acoustic")

from .definition import SimulationMethod

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
        """Run the simulation.

        Parameters
        ----------
        json_file_path : str | Path | None, optional
            Path to the JSON file. If not provided, uses the path from initialization.
        """
        self._misuka_method(self.input_json_path)


    def _misuka_method(self, json_file_path: str | Path) -> None:
        """
        Run misuka simulation for acoustic wave propagation.

        Args:
            json_file_path: Path to the JSON configuration file
        """

        # ----------------------------------------------------------------
        # ------- fetching simulation settings and initialization -------
        # ----------------------------------------------------------------

        try:
            # Load the input JSON file
            with open(json_file_path, "r") as json_file:
                result_container = json.load(json_file)
        except KeyError as e:
            raise KeyError(f"Could not fetch the simulation settings for misuka T.T: {e}")
        
        simulation_settings = result_container["simulationSettings"]
        speed_of_sound = simulation_settings["speed_of_sound"]
        max_time = simulation_settings["max_time"]
        time_bins = simulation_settings["sampling_rate"]
        rpf = simulation_settings["rays_per_frequency"]
        sound_power_W = simulation_settings["sound_power_W"]
        n_receivers = len(result_container["results"][0]["responses"])
        frequencies = result_container["results"][0]["frequencies"]
        msh_path = result_container["msh_path"]
        absorption_map = result_container["absorption_coefficients"]
        dynamic_range_db = simulation_settings["dynamic_range_db"] == "yes"
        log_data = simulation_settings["log_data"] == "yes"
        
        if log_data: 
            print(f'Using variant: {mi.variant()}')
            print(f'Input value for speed of sound: {speed_of_sound}')
            print(f'Input value for maximum simulation time: {max_time}')
            print(f'Input value for time_bins: {time_bins}')
            print(f'Input value for rays per frequency: {rpf}')
            print(f'Input value for source sound power in W: {sound_power_W}')
            print(f'Rendered frequencies: {frequencies}')
            print(f'Input values for absorption: {absorption_map}')
            print(f'Use dynamic range limiter: {dynamic_range_db}')

        
        frequency_range = (float(np.min(frequencies)), float(np.max(frequencies)))
        f_center, f_lower, f_upper = pf.constants.fractional_octave_frequencies_exact(1, frequency_range)
        assert np.all(np.abs(f_center-frequencies)/frequencies < 1e-2), "Frequency mismatch between input frequencies and pyrato's fractional octave frequencies"
        frequencies_misuka = ", ".join(str(f) for f in frequencies)

        #TODO: currently using scattering specified by misuka_setting / User... for future use from msh file
        scattering_map = [
            simulation_settings[f"scattering_{int(f)}"]
            for f in frequencies
        ]
        # Create frequency-scattering tuples for spectrum
        scattering_spectrum = create_frequency_spectrum(frequencies, scattering_map) #TODO: when scattering is surface material dependent move this into for loop
        
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

        if log_data: print(f'Source coordinates: {source_coords}\n Reciever coordinates: {receiver_coords}')

        # updating percentage of progress
        set_progress_and_save(25, result_container, json_file_path)

        # set up the integrator
        integrator_acoustic = mi.load_dict(
            {
                "type": "acoustic_path",
                "speed_of_sound": speed_of_sound,
                "max_depth": -1,  # maximum path depth (-1 = no limit, 1 = direct sound only, 2 = up to 1 reflection, etc.)
                "max_time": max_time,  # maximum propagation time in seconds
            }
        )

        if log_data: print(f'Integrator: {integrator_acoustic}')

        # ----------------------------------------------------------------
        # --------------------- initializing scene ---------------------
        # ----------------------------------------------------------------
       
        # define scene as dictionary
        scene_dict = {
            "type": "scene",
        }

        # Extract physical group information from gmsh
        gmsh.initialize()
        gmsh.open(str(msh_path))
        surfaces = gmsh.model.getPhysicalGroups(2)

        # For each surface, create a separate PLY file
        # Nach gmsh.open(str(msh_file)):
        ply_dir = Path(msh_path).parent

        for i, (dim, tag) in enumerate(surfaces):
            surface_name = gmsh.model.getPhysicalName(dim, tag)
            absorption_str = absorption_map.get(surface_name, "0.1, 0.1, 0.1, 0.1, 0.1, 0.1")
            entity_tags = gmsh.model.getEntitiesForPhysicalName(surface_name)

            ply_file = str(ply_dir / f"surface_{surface_name}.ply")  # korrekt
            export_physical_surface_to_ply(surface_name, entity_tags, ply_file)
            absorption_spectrum = create_frequency_spectrum(frequencies, absorption_str)

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

        # ----------------------------------------------------------------
        # -------- rendering for every source to every reciever ----------
        # ----------------------------------------------------------------
        
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

                scene_dict["emitter"] = {
                    "type": "sphere",
                    "radius": 0.1,
                    "center": source_coord,
                    "emitter": {
                        'type': 'area',
                        'radiance': {
                        'type': 'uniform',
                        'value': sound_power_W,
                        }
                    }
                }

                scene = mi.load_dict(scene_dict)

                if log_data: print(f'Loaded Scene for reciever at {receiver_coord} and source at {source_coord}: {scene}')

                #TODO: use sound_power_W for scaling the etc
                etc = mi.render(scene, sensor=microphone, integrator=integrator_acoustic,spp=rpf)
                etc_signal = pf.Signal(etc.numpy().T, sampling_rate=time_bins / max_time, domain='time')
                etc_signal.time /= np.max(np.abs(etc_signal.time))
                edc = pr.edc.schroeder_integration(etc_signal, is_energy=True)
                edc_normalized = pf.dsp.normalize(edc)
                edc_normalized_db = 10*np.log10(edc_normalized.time/1e-12)
                edc_normalized_db = np.squeeze(edc_normalized_db, axis=0)
                edc_normalized_db = finite_array(edc_normalized_db, nan=0.0, neginf=-600.0, posinf=0.0)
                
                if dynamic_range_db:
                    limit = np.max(edc_normalized_db) - 100
                    edc_normalized_db[edc_normalized_db<limit] = limit

                result_container["results"][0]["responses"][i_rec]["receiverResults"] = [
                    {
                        "data": (edc_normalized_db[i_frq]).tolist(),
                        "t": etc_signal.times.tolist(),
                        "frequency": frequencies[i_frq],
                        "type": "edc",
                    }
                    for i_frq in range(len(frequencies))
                ]

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
            set_progress_and_save(35 + int(55/(n_receivers+1) * (i_src + 1)), result_container, json_file_path)
        

        # set to 100%
        set_progress_and_save(100, result_container, json_file_path)






def set_progress_and_save(percentage, result_container, json_file_path):
    result = result_container["results"][0]
    result["percentage"] = percentage
    result.pop("percentages", None)
    # Save the updated JSON
    with open(json_file_path, "w") as json_output:
        json_output.write(json.dumps(result_container, indent=4, allow_nan=False))

def create_frequency_spectrum(frequencies: list, values) -> list:
    """Create frequency-value tuples for Mitsuba spectrum format.
    
    Parameters
    ----------
    frequencies : list
        List of frequency values [125, 250, 500, ...]
    values : list or str
        Either a list of floats or a comma-separated string of values
    
    Returns
    -------
    list
        List of (frequency, value) tuples for Mitsuba spectrum
    """
    # Parse values if string
    if isinstance(values, str):
        vals = [float(val.strip()) for val in values.split(",")]
    else:
        vals = [float(val) for val in values]
    
    # Pad with default value (0.1) if not enough values provided
    while len(vals) < len(frequencies):
        vals.append(0.1)
    
    return list(zip(frequencies, vals[:len(frequencies)]))

def export_physical_surface_to_ply(surface_name: str, entity_tags, ply_path: str | Path) -> Path:
    """Export only the triangular mesh faces of one physical surface to PLY."""
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

def finite_array(values, nan=0.0, posinf=0.0, neginf=0.0) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )