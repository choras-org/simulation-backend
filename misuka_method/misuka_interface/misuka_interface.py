"""Module implementing a CHORAS interface for misuka."""

import json
from pathlib import Path
import gmsh
import meshio
import mitsuba as mi
import numpy as np
import pyfar as pf
import pyrato
import pandas as pd


mi.set_variant("cuda_acoustic", "llvm_acoustic", "cuda_ad_acoustic", "llvm_ad_acoustic")

from mitsuba import ScalarTransform4f as tf

from .definition import SimulationMethod



RESULT_PARAMETER_KEYS = (
    "edt",
    "t20",
    "t30",
    "c80",
    "d50",
    "ts",
    "spl_t0_freq",
)


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

        # Load the input JSON file
        with open(json_file_path, "r") as json_file:
            result_container = json.load(json_file)

        try:
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
            dynamic_range_db = simulation_settings["dynamic_range_db"]
        except KeyError as e:
            raise KeyError(f"Could not fetch the simulation settings for misuka T.T: {e}")
        
        source_intensity = sound_power_W / (4.0 * np.pi) # omnidirectional point source
        
        n_bands = len(frequencies)
        frequency_range = (float(np.min(frequencies)), float(np.max(frequencies)))
        f_center, f_lower, f_upper = pf.constants.fractional_octave_frequencies_exact(1, frequency_range)
        assert np.all(np.abs(f_center-frequencies)/frequencies < 1e-2), "Frequency mismatch between input frequencies and pyrato's fractional octave frequencies"
        frequencies_misuka = ", ".join(str(f) for f in frequencies)
        
        #TODO: currently using scattering specified by misuka_setting / User... for future use from msh file
        scattering_map = [
            simulation_settings[f"scattering_{int(f)}"]
            for f in frequencies
        ]
        source_coords = [
            result_container["results"][0]["sourceX"],
            result_container["results"][0]["sourceY"],
            result_container["results"][0]["sourceZ"],
        ]
        receiver_coords = [
            [
                result_container["results"][0]["responses"][i]["x"],
                result_container["results"][0]["responses"][i]["y"],
                result_container["results"][0]["responses"][i]["z"],
            ]
            for i in range(n_receivers)
        ]

        # updating percentage of progress
        set_progress_and_save(25, result_container, json_file_path)

        # ----------------------------------------------------------------
        # --- initializing scene parameters, microphone and integrator ---
        # ----------------------------------------------------------------
 
        scene = create_acoustic_scene(
            msh_file=msh_path,
            frequencies=frequencies,
            absorption_map=absorption_map,
            source_position=source_coords,
            source_intensity=source_intensity,
            scattering_map=scattering_map,
        )

        # set up the integrator
        integrator_acoustic = mi.load_dict(
            {
                "type": "acoustic_path",
                "speed_of_sound": speed_of_sound,
                "max_depth": -1,  # maximum path depth (-1 = no limit, 1 = direct sound only, 2 = up to 1 reflection, etc.)
                "max_time": max_time,  # maximum propagation time in seconds
            }
        )

        set_progress_and_save(35, result_container, json_file_path)

        # ----------------------------------------------------------------
        # ------------- calculating ETC for each receiver --------------
        # ----------------------------------------------------------------

        etc = np.empty((n_receivers, n_bands, time_bins), dtype=np.float64)

        for i_rec, receiver_coord in enumerate(receiver_coords):
            # set up the microphone sensor for this receiver
            microphone_direction = (
                np.asarray(source_coords) - np.asarray(receiver_coord)
            )
            distance_to_source = np.linalg.norm(
                microphone_direction.astype(float)
            )
            if distance_to_source == 0:
                raise ValueError(
                    f"Receiver {i_rec} is at the same position as the source."
                )
            microphone_direction = (
                microphone_direction / distance_to_source
            ).tolist()

            microphone = mi.load_dict(
                {
                    "type": "microphone",
                    "origin": receiver_coord,
                    "direction": microphone_direction,
                    "film": {
                        "type": "tape",
                        "frequencies": frequencies_misuka,  # rendered frequencies
                        "time_bins": time_bins,  # number of time bins
                    },
                }
            )

            # render the scene for this receiver and store the ETC
            render_raw = mi.render(
                scene,
                integrator=integrator_acoustic,
                sensor=microphone,
                spp=rpf,
            )
            # Convert Mitsuba tensor to float64 explicitly
            render_array = np.asarray(render_raw, dtype=np.float64) / float(rpf) #TODO: if misuka already averages over rays, this division is not needed. Check this and remove if so.
            render_array = np.squeeze(render_array)

            if render_array.shape != (time_bins, n_bands):
                raise ValueError(
                    "Unexpected Misuka render shape after squeeze: "
                    f"{render_array.shape}; expected {(time_bins, n_bands)}"
                )
            
            etc[i_rec] = render_array.T.astype(np.float64)

            # update progress for each reciever in even steps to ~90% 
            set_progress_and_save(35 + int(55/(n_receivers+1) * (i_rec + 1)), result_container, json_file_path)

        # ----------------------------------------------------------------
        # ------------- Converting ETC to EDC --------------
        # ----------------------------------------------------------------

        # Create time vector
        times_vector = np.linspace(0, max_time, time_bins)
        
        # Ensure etc is float64 before processing
        etc = np.asarray(etc, dtype=np.float64)
        times_vector = np.asarray(times_vector, dtype=np.float64)
        
        np.nan_to_num(etc, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(etc, 0.0, out=etc)

        # Create pf.TimeData with explicit float64 dtypes
        etc = pf.TimeData(etc.astype(np.float64), times_vector.astype(np.float64))
        times = etc.times.tolist()

        prepare_result_responses(result_container)
        for i_rec in range(n_receivers):
            edc = etc_to_edc(etc[i_rec, :], f_lower, f_upper)
            edc_db_normalized, edc_db_absolute = edc_to_db(edc, dynamic_range_db)
            # Ensure edc_db arrays are float64
            edc_db_normalized = np.asarray(edc_db_normalized, dtype=np.float64)
            edc_db_absolute = np.asarray(edc_db_absolute, dtype=np.float64)

            store_receiver_results(
                result_container["results"][0]["responses"][i_rec],
                edc_db_normalized,
                times,
                frequencies,
            )

            calculate_and_store_acoustic_parameters(
                edc, edc_db_absolute, result_container, i_rec
            )

        #print(result_container)

                # Export pressure data for legacy auralization pipeline
        pressure_csv_path = str(json_file_path).replace(".json", "_pressure.csv")

        pressure_data = []
        for time_idx, time_value in enumerate(times):
            row = {"t": time_value}
            for band_idx, frequency in enumerate(frequencies):
                row[f"{int(frequency)}Hz"] = float(
                    etc.time[0, band_idx, time_idx]
                )
            pressure_data.append(row)

        df = pd.DataFrame(pressure_data)
        df.to_csv(pressure_csv_path, index=False)
        print(f"Pressure CSV saved as {pressure_csv_path}")


        # Save and update
        set_progress_and_save(100, result_container, json_file_path)

        print(f'Files saved as {json_file_path}')


# ----------------------------------------------------------------
# ------- helper methods -------
# ----------------------------------------------------------------

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

def create_acoustic_scene(
    msh_file,
    frequencies,
    absorption_map,
    source_position,
    source_intensity,
    scattering_map,
):
    """Create an acoustic scene with surfaces from the MSH file.
    
    Parameters
    ----------
    msh_file : str | Path
        Path to the MSH file
    frequencies : list
        List of frequency values
    absorption_map : dict
        Mapping of surface names to comma-separated absorption values
    source_position : list
        [x, y, z] source position
    source_intensity : float
        Source intensity value
    scattering_map : list
        List of scattering coefficients per frequency band
    """
    
    scene_dict = {
        "type": "scene",
        "source": {
            "type": "point",
            "position": source_position,
            "intensity": {
                "type": "spectrum",
                "value": source_intensity,
            },
        },
    }

    # Create frequency-scattering tuples for spectrum
    scattering_spectrum = create_frequency_spectrum(frequencies, scattering_map)

    # Extract physical group information from gmsh
    gmsh.initialize()
    gmsh.open(str(msh_file))
    surfaces = gmsh.model.getPhysicalGroups(2)

    # For each surface, create a separate PLY file
    for i, (dim, tag) in enumerate(surfaces):
        surface_name = gmsh.model.getPhysicalName(dim, tag)
        absorption_str = absorption_map.get(surface_name, "0.1, 0.1, 0.1, 0.1, 0.1, 0.1")
        entity_tags = gmsh.model.getEntitiesForPhysicalName(surface_name)

        # Create output file for this surface
        ply_file = f"surface_{surface_name}.ply"

        export_physical_surface_to_ply(surface_name, entity_tags, ply_file)

        # Create frequency-absorption tuples for spectrum
        absorption_spectrum = create_frequency_spectrum(frequencies, absorption_str)

        scene_dict[f"shape_{i}"] = {
            "type": "ply",
            "filename": ply_file,
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
            },
        }

    gmsh.finalize()  # Finalize AFTER the loop

    return mi.load_dict(scene_dict)

def set_progress_and_save(percentage, result_container, json_file_path):
    result = result_container["results"][0]
    result["percentage"] = percentage
    result.pop("percentages", None)
    # Save the updated JSON
    with open(json_file_path, "w") as json_output:
        json_output.write(json.dumps(result_container, indent=4, allow_nan=False))



def prepare_result_responses(result_container: dict) -> None:
    """Reset response result containers to the CHORAS EDC/parameter format."""
    for response in result_container["results"][0]["responses"]:
        response["receiverResults"] = []
        response["parameters"] = {key: [] for key in RESULT_PARAMETER_KEYS}

def finite_array(values, nan=0.0, posinf=0.0, neginf=0.0) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )

def json_scalar(value):
    """Convert numpy scalar values to regular JSON scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    return value

def edc_to_db(edc: pf.TimeData, dynamic_range_db: float) -> tuple:
    """Convert an EDC to dB and clip it to the requested dynamic range.
    
    Returns (edc_db, edc_db_absolute) where edc_db is normalized to peak=0dB
    and edc_db_absolute uses absolute dB reference for SPL calculation.
    """
    edc_energy = np.nan_to_num(
        np.asarray(edc.time, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    
    # For pyrato compatibility: keep energy form but ensure minimum values
    # to avoid log(0) issues in pyrato functions
    edc_energy_safe = np.maximum(edc_energy, 1e-15)
    
    # Calculate absolute dB (referenced to 1e-12 W/m²)
    edc_db_absolute = 10 * np.log10(edc_energy_safe / 1e-12)
    
    # Normalize EDC to peak = 0 dB for display
    peak_energy = np.max(edc_energy_safe, axis=-1, keepdims=True)
    edc_normalized = edc_energy_safe / peak_energy
    edc_db_normalized = 10 * np.log10(edc_normalized)
    
    # Clip to dynamic range
    limit = -dynamic_range_db
    edc_db_normalized = np.maximum(edc_db_normalized, limit)
    
    return edc_db_normalized, edc_db_absolute

def store_receiver_results(
    response: dict,
    edc_db: np.ndarray,
    times: list,
    frequencies: np.ndarray,
) -> None:
    """Store one EDC result object per frequency band."""
    # Ensure all inputs are proper numeric types
    edc_db = np.asarray(edc_db, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    
    response["receiverResults"] = [
        {
            "data": finite_array(edc_db[i_frequency], nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "t": times if isinstance(times, list) else times.tolist(),
            "frequency": json_scalar(frequencies[i_frequency]),
            "type": "edc",
        }
        for i_frequency in range(len(frequencies))
    ]

def etc_to_edc(
    etc: pf.TimeData,
    lower_frequency_cutoffs: np.ndarray,
    upper_frequency_cutoffs: np.ndarray,
) -> pf.TimeData:
    """Convert energy time curve into energy decay curve.

    Parameters
    ----------
    etc : pf.TimeData
        energy time curve of cshape (..., n_bands).
    lower_frequency_cutoffs : np.ndarray
        lower cutoff frequencies from the frequency bands of shape (n_bands).
    upper_frequency_cutoffs : np.ndarray
        lower cutoff frequencies from the frequency bands of shape (n_bands).

    Results
    -------
    edc : pf.TimeData
        Resulting energy decay curve.
    """
    full_frequency_range = np.max(upper_frequency_cutoffs) - np.min(
        lower_frequency_cutoffs
    )
    bandwidth = upper_frequency_cutoffs - lower_frequency_cutoffs
    etc_eq = etc * (bandwidth / full_frequency_range)
    edc = pyrato.edc.schroeder_integration(etc_eq, is_energy=True)
    return edc

def center_time(edc: pf.TimeData) -> np.ndarray:
    edc_time = np.asarray(edc.time)
    energy = np.empty_like(edc_time)
    energy[..., :-1] = edc_time[..., :-1] - edc_time[..., 1:]
    energy[..., -1] = edc_time[..., -1]
    np.maximum(energy, 0, out=energy)

    denominator = np.sum(energy, axis=-1)
    numerator = np.sum(energy * np.asarray(edc.times), axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )

def calculate_reverberation_time_from_db(edc_db: np.ndarray, times: np.ndarray, db_start: float, db_end: float) -> np.ndarray:
    """Calculate reverberation time from dB decay curve.
    
    Fit a line between db_start and db_end points and extrapolate to -60 dB.
    
    Parameters
    ----------
    edc_db : np.ndarray
        EDC in dB (shape: n_bands x n_times)
    times : np.ndarray
        Time array (shape: n_times)
    db_start : float
        Starting dB level for fitting (e.g., -5)
    db_end : float
        Ending dB level for fitting (e.g., -25 for T20, -35 for T30, -10 for EDT)
    
    Returns
    -------
    np.ndarray
        Reverberation times for each band (shape: n_bands)
    """
    n_bands = edc_db.shape[0]
    rt = np.zeros(n_bands)
    
    for i_band in range(n_bands):
        decay = edc_db[i_band, :]
        
        # Find indices where decay is within [db_start, db_end]
        mask = (decay >= db_end) & (decay <= db_start)
        
        if np.sum(mask) < 2:
            # Not enough points to fit
            rt[i_band] = 0.0
            continue
        
        # Fit a line to the decay in this range
        valid_times = times[mask]
        valid_decay = decay[mask]
        
        # Linear regression: decay = a * time + b
        coeffs = np.polyfit(valid_times, valid_decay, 1)
        slope = coeffs[0]  # dB/second
        
        if slope >= 0:
            # Not decaying properly
            rt[i_band] = 0.0
            continue
        
        # Extrapolate to -60 dB: -60 = slope * t + b
        # t = (-60 - b) / slope
        intercept = coeffs[1]
        t_60 = (-60.0 - intercept) / slope
        
        # Ensure positive and reasonable value
        if t_60 > 0 and t_60 < 100:  # Max 100 seconds is reasonable for indoor rooms
            rt[i_band] = t_60
        else:
            rt[i_band] = 0.0
    
    return rt


def calculate_and_store_acoustic_parameters(
    edc: pf.TimeData,
    edc_db_absolute: np.ndarray,
    result_container: dict,
    i_rec: int) -> None:
    """Calculate and store acoustic parameters for a receiver.

    Parameters
    ----------
    edc : pf.TimeData
        Energy decay curve (in energy domain, not dB).
    edc_db_absolute : np.ndarray
        Energy decay curve in absolute dB (for SPL at t=0).
    result_container : dict
        Result container to store parameters.
    i_rec : int
        Receiver index.
    """
    response = result_container["results"][0]["responses"][i_rec]
    parameters = response.setdefault(
        "parameters", {key: [] for key in ["t20", "t30", "c80", "d50", "ts", "spl_t0_freq", "edt"]}
    )

    times = np.asarray(edc.times)
    
    try:
        t20 = calculate_reverberation_time_from_db(edc_db_absolute, times, db_start=-5.0, db_end=-25.0)
        parameters["t20"] = t20.tolist()
    except Exception as e:
        parameters["t20"] = [0.0] * len(edc.time)

    try:
        t30 = calculate_reverberation_time_from_db(edc_db_absolute, times, db_start=-5.0, db_end=-35.0)
        parameters["t30"] = t30.tolist()
    except Exception as e:
        parameters["t30"] = [0.0] * len(edc.time)

    try:
        c80 = pyrato.parameters.clarity(edc, 80)
        c80 = finite_array(c80, nan=0.0, neginf=0.0, posinf=0.0)
        parameters["c80"] = c80.tolist()
    except Exception as e:
        parameters["c80"] = [0.0] * len(edc.time)

    try:
        d50 = pyrato.parameters.definition(edc, 50) * 100
        d50 = finite_array(d50, nan=0.0, neginf=0.0, posinf=0.0)
        parameters["d50"] = d50.tolist()
    except Exception as e:
        parameters["d50"] = [0.0] * len(edc.time)

    try:
        ts = center_time(edc) * 1000
        ts = finite_array(ts, nan=0.0, neginf=0.0, posinf=0.0)
        parameters["ts"] = ts.tolist()
    except Exception as e:
        parameters["ts"] = [0.0] * len(edc.time)

    try:
        spl = edc_db_absolute[..., 0]
        parameters["spl_t0_freq"] = finite_array(spl, nan=0.0, neginf=0.0, posinf=0.0).tolist()
    except Exception as e:
        parameters["spl_t0_freq"] = [0.0] * len(edc.time)

    try:
        edt = calculate_reverberation_time_from_db(edc_db_absolute, times, db_start=0.0, db_end=-10.0)
        parameters["edt"] = edt.tolist()
    except Exception as e:
        parameters["edt"] = [0.0] * len(edc.time)
