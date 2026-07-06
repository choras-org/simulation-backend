import io
import json
import logging
import os
import sys
import traceback
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from deism.core_deism import DEISM
from deism.data_loader import ConflictChecks, compute_rest_params, detect_conflicts
from deism.room_check import get_room_geometry, sync_room_geometry

from .definition import SimulationMethod

logger = logging.getLogger(__name__)

# Maps the CHORAS `simulationSettings` JSON keys to DEISM's internal
# `deism.params` keys (and the caster to apply). Both the CHORAS-facing
# names (from `deism_setting.json`) and the raw DEISM parameter names are
# accepted so the settings file can use either convention.
DEISM_JSON_KEY_MAP = {
    "soundSpeed": ("soundSpeed", float),
    "airDensity": ("airDensity", float),
    "RIRLength": ("RIRLength", float),
    "samplingRate": ("sampleRate", int),
    "maxReflectionOrder": ("maxReflOrder", int),
    "sourceOrder": ("sourceOrder", int),
    "receiverOrder": ("receiverOrder", int),
    "sourceRadius": ("radiusSource", float),
    "receiverRadius": ("radiusReceiver", float),
    "sourceOrientation": ("orientSource", "array"),
    "receiverOrientation": ("orientReceiver", "array"),
    "sourceDirectivity": ("sourceType", str),
    "receiverDirectivity": ("receiverType", str),
    "ifRemoveDirect": ("ifRemoveDirectPath", int),
    "Method": ("DEISM_method", str),
    "mixEarlyOrder": ("mixEarlyOrder", int),
    "numParaImages": ("numParaImages", int),
    "ifReceiverNormalize": ("ifReceiverNormalize", int),
    "QFlowStrength": ("qFlowStrength", float),
    # Backward-compatible aliases matching the raw DEISM parameter names.
    "sampleRate": ("sampleRate", int),
    "maxReflOrder": ("maxReflOrder", int),
    "radiusSource": ("radiusSource", float),
    "radiusReceiver": ("radiusReceiver", float),
    "orientSource": ("orientSource", "array"),
    "orientReceiver": ("orientReceiver", "array"),
    "sourceType": ("sourceType", str),
    "receiverType": ("receiverType", str),
    "ifRemoveDirectPath": ("ifRemoveDirectPath", int),
    "DEISM_method": ("DEISM_method", str),
    "qFlowStrength": ("qFlowStrength", float),
}

def create_vgroups_names(file_path):
    """List the Gmsh physical-group tags and their material names.

    Parameters
    ----------
    file_path : str
        Full path to the mesh file.

    Returns
    -------
    list
        ``[dim, tag, name]`` entries for every physical group in the mesh
        (the name matches the material name assigned in the CHORAS
        geometry editor).
    """
    gmsh.initialize()
    try:
        gmsh.open(file_path)
        vgroups = gmsh.model.getPhysicalGroups(-1)
        vgroups_names = []
        for dim_group, tag_group in vgroups:
            name_group = gmsh.model.getPhysicalName(dim_group, tag_group)
            vgroups_names.append([dim_group, tag_group, name_group])
        return vgroups_names
    finally:
        gmsh.finalize()


def parse_value(val):
    """Parse a comma-separated string or scalar into a numpy array."""
    if isinstance(val, str):
        return np.array([float(x.strip()) for x in val.split(",") if x.strip()])
    if isinstance(val, (int, float)):
        return np.array([val])
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=float)
    raise ValueError(f"Unsupported type for parse_value: {type(val)}")


def parse_array_value(val):
    """Parse a JSON array-like setting into a numpy vector."""
    if isinstance(val, str):
        return np.array([float(x.strip()) for x in val.split(",") if x.strip()])
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.array(val, dtype=float)
    raise ValueError(f"Unsupported array setting type: {type(val)}")


def apply_simulation_settings_to_deism(deism, simulation_settings):
    """Override DEISM's yaml-loaded defaults with the runtime JSON settings.

    The JSON keys are expected to match `DEISM_JSON_KEY_MAP`, which maps
    them onto the final parameter names used internally by
    `deism.data_loader`.
    """
    if not simulation_settings:
        return
    if not isinstance(simulation_settings, dict):
        raise TypeError("simulationSettings must be a JSON object")

    for key, value in simulation_settings.items():
        if value is None:
            continue
        if key not in DEISM_JSON_KEY_MAP:
            logger.warning("Ignoring unsupported DEISM setting key: %s", key)
            continue

        target_key, caster = DEISM_JSON_KEY_MAP[key]
        if caster == "array":
            deism.params[target_key] = parse_array_value(value)
        else:
            deism.params[target_key] = caster(value)

    # Recompute dependent parameters after overriding the yaml defaults.
    deism.params = compute_rest_params(deism.params)


def apply_choras_runtime_overrides(deism, coord_source, coord_rec):
    """Apply the per-simulation source/receiver values from CHORAS.

    `freqs`/`waveNumbers`/`pointSrcStrength` are intentionally not set here:
    `deism.update_freqs()` (called later in `_deism_method`) unconditionally
    recomputes all three from `RIRLength`/`sampleRate`, so setting them here
    would just be overwritten.
    """
    deism.params["posSource"] = np.array(coord_source, dtype=float)
    deism.params["posReceiver"] = np.array(coord_rec, dtype=float)

    # CHORAS supplies the room geometry (via `update_room`) and the source/
    # receiver facing angles (`orientSource`/`orientReceiver`) already expressed
    # in the room's own coordinate frame. DEISM's bundled example config
    # (`configSingleParam_ARG_RIR.yml`) ships `ifRotateRoom: 1` with a
    # `[90, 90, 90]` example rotation; since neither the interface nor the
    # frontend overrides it, every CHORAS run would otherwise inherit that
    # example rotation and apply it on top of the CHORAS orientations, silently
    # rotating any directive source/receiver by 90/90/90 degrees. Disable it so
    # only the CHORAS-provided orientations are used.
    deism.params["ifRotateRoom"] = 0
    deism.params["roomRotation"] = np.zeros(3, dtype=float)


def create_deism_instance(mode, room_type):
    """Create a DEISM instance without leaking the container's own argv.

    CHORAS reports progress via the result JSON, not DEISM console output, so
    DEISM runs silently. Wrapper-level logs (start/done/errors/conflicts) are
    still emitted for container debugging.
    """
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] if original_argv else ["deism"]
        return DEISM(mode, room_type, silent=True)
    finally:
        sys.argv = original_argv


@contextmanager
def use_real_stdio():
    """Temporarily restore real stdio for DEISM calls needing `fileno()`."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    fallback_stream = None

    try:
        stdout_stream = (
            sys.__stdout__
            if getattr(sys.__stdout__, "fileno", None) is not None
            else None
        )
        stderr_stream = (
            sys.__stderr__
            if getattr(sys.__stderr__, "fileno", None) is not None
            else None
        )

        if stdout_stream is None or stderr_stream is None:
            fallback_stream = open(os.devnull, "w")
            if stdout_stream is None:
                stdout_stream = fallback_stream
            if stderr_stream is None:
                stderr_stream = fallback_stream

        sys.stdout = stdout_stream
        sys.stderr = stderr_stream
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if fallback_stream is not None:
            fallback_stream.close()


def get_deism_surface_order(vgroups_names, wall_centers_loaded):
    """Return the 6 wall surface names in a deterministic order.

    CHORAS rooms take DEISM's convex/ARG path, which re-matches each wall to
    its absorption value by centroid proximity (see ``deism.core_deism_arg``).
    The specific order is therefore irrelevant to the physics as long as each
    surface's absorption stays paired with its own centroid -- which the caller
    guarantees by keying both off the same surface name. We sort by centroid so
    the result no longer depends on Gmsh physical-tag declaration order.
    """
    surface_names = [name for dim, tag, name in vgroups_names if int(dim) == 2]
    if len(surface_names) != 6:
        raise ValueError(
            f"DEISM requires exactly 6 wall surfaces, found {len(surface_names)}"
        )
    centers = {
        name: np.asarray(wall_centers_loaded[name], dtype=float)
        for name in surface_names
    }
    return sorted(surface_names, key=lambda name: tuple(centers[name]))


def update_result_percentage(result_container, json_file_path, percentage):
    """Persist simulation progress for the first result entry."""
    if not result_container or not result_container.get("results"):
        return

    result_container["results"][0]["percentage"] = int(percentage)
    with open(json_file_path, "w") as json_file:
        json.dump(result_container, json_file, indent=4)


def run_parameter_conflict_checks(deism, result_container, json_file_path):
    """Run DEISM's built-in parameter conflict checks and surface the outcome.

    DEISM validates its parameters via ``ConflictChecks.check_all_conflicts``
    (which enforces some rules and raises ``ValueError`` on hard conflicts, e.g.
    conflicting wall materials) and ``detect_conflicts`` (print-only
    diagnostics). Both normally write to stdout, where their warnings are lost,
    and a raised error would kill the container with no explanation reaching
    CHORAS. Here we capture both, persist the warnings/error into the result
    JSON (which the backend reads back from the shared uploads volume), and
    re-emit them to the container log.

    On a hard conflict the parameters are invalid, so we stop before the
    expensive solve and raise -- but only after the message has been persisted
    for CHORAS to display.
    """
    buffer = io.StringIO()
    error_message = None
    try:
        with redirect_stdout(buffer):
            ConflictChecks.check_all_conflicts(deism.params)
            detect_conflicts(deism.params)
    except ValueError as exc:
        error_message = str(exc)

    captured = buffer.getvalue()
    # Re-emit captured diagnostics so they still appear in the container log.
    if captured.strip():
        print(captured, end="", flush=True)

    # ``check_all_conflicts`` and ``detect_conflicts`` overlap, so the same
    # warning can appear twice; dedupe while preserving first-seen order.
    warnings = list(
        dict.fromkeys(
            line.strip() for line in captured.splitlines() if "[Warning]" in line
        )
    )

    if warnings or error_message:
        diagnostics = {"warnings": warnings}
        if error_message:
            diagnostics["error"] = error_message
            print(f"[ParameterConflict] {error_message}", flush=True)
        result_container["parameterDiagnostics"] = diagnostics
        with open(json_file_path, "w") as json_file:
            json.dump(result_container, json_file, indent=4)

    if error_message:
        raise ValueError(error_message)


def check_should_cancel(json_file_path):
    """Check whether the user has requested cancellation via the JSON file."""
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
        return data.get("should_cancel", False)
    except Exception as exc:
        print(f"check_should_cancel returned: {exc}")
        print(traceback.format_exc())
        return False


class DeismMethod(SimulationMethod):
    """Interface class to run the DEISM (image-source) method.

    The class runs a full DEISM room-impulse-response calculation. All
    required configuration parameters are expected to be provided in the
    input JSON file passed during initialization.
    """

    def __init__(self, input_json_path: str | Path | None = None):
        """Initialize the DEISM method interface for the given JSON file."""
        super().__init__(input_json_path)

    def run_simulation(self, json_file_path: str | Path | None = None) -> None:
        """Run the simulation.

        Parameters
        ----------
        json_file_path : str | Path | None, optional
            Path to the JSON file. If not provided, uses the path from
            initialization.
        """
        if json_file_path is None:
            json_file_path = self.input_json_path
        self._deism_method(json_file_path)

    def _deism_method(self, json_file_path: str | Path) -> None:
        """Run the DEISM simulation described by `json_file_path`."""
        print("deism_method: starting simulation")

        # DEISM writes plots/temp files relative to the current working
        # directory; run from the package directory like the other methods.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        original_cwd = os.getcwd()
        os.chdir(script_dir)

        try:
            with open(json_file_path, "r") as json_file:
                result_container = json.load(json_file)
            geo_path = result_container["geo_path"]

            # Populate result_container["geometry"] (vertices, wall
            # centers, areas, volume) from the mesh file.
            sync_room_geometry(json_file_path, geo_path)
            _volume, room = get_room_geometry(geo_file=geo_path)

            with open(json_file_path, "r") as json_file:
                result_container = json.load(json_file)
            update_result_percentage(result_container, json_file_path, 10)

            if check_should_cancel(json_file_path):
                return

            vgroups_names = create_vgroups_names(result_container["geo_path"])

            simulation_settings = result_container.get("simulationSettings", {})
            coord_source = [
                result_container["results"][0]["sourceX"],
                result_container["results"][0]["sourceY"],
                result_container["results"][0]["sourceZ"],
            ]
            coord_rec = [
                result_container["results"][0]["responses"][0]["x"],
                result_container["results"][0]["responses"][0]["y"],
                result_container["results"][0]["responses"][0]["z"],
            ]
            abs_coeffs_loaded = result_container["absorption_coefficients"]
            freq_bands = np.array(result_container["results"][0]["frequencies"])

            # Reorder room geometry and absorption coefficients into
            # DEISM's expected wall order: x1, x2, y1, y2, z1, z2.
            vertices = np.array(result_container["geometry"][0]["vertices"])
            wall_centers_loaded = result_container["geometry"][0]["wall_centers"]
            room_volumn = result_container["geometry"][0]["room_volumn"]
            room_areas_loaded = result_container["geometry"][0]["room_areas"]

            wall_order = get_deism_surface_order(vgroups_names, wall_centers_loaded)
            absorption_coefficients = np.zeros((6, len(freq_bands)))
            wall_centers = np.zeros((6, 3))
            room_areas = np.zeros((6, 1))
            for index, wall in enumerate(wall_order):
                absorption_coefficients[index, :] = parse_value(abs_coeffs_loaded[wall])
                wall_centers[index, :] = parse_value(wall_centers_loaded[wall])
                room_areas[index, :] = parse_value(room_areas_loaded[wall])
            update_result_percentage(result_container, json_file_path, 25)

            with use_real_stdio():
                deism = create_deism_instance("RIR", room)
            apply_simulation_settings_to_deism(deism, simulation_settings)
            apply_choras_runtime_overrides(deism, coord_source, coord_rec)
            update_result_percentage(result_container, json_file_path, 35)

            deism.update_room(vertices, wall_centers, room_volumn, room_areas)
            run_parameter_conflict_checks(deism, result_container, json_file_path)
            update_result_percentage(result_container, json_file_path, 45)

            deism.update_wall_materials(
                absorption_coefficients, freq_bands, "absorpCoefficient"
            )
            update_result_percentage(result_container, json_file_path, 55)

            deism.update_freqs()
            update_result_percentage(result_container, json_file_path, 65)

            with use_real_stdio():
                deism.update_source_receiver()
            update_result_percentage(result_container, json_file_path, 75)

            with use_real_stdio():
                deism.update_directivities()
            update_result_percentage(result_container, json_file_path, 85)

            with use_real_stdio():
                deism.run_DEISM()
            update_result_percentage(result_container, json_file_path, 95)

            rir = deism.get_results()
            rir = rir / np.max(np.abs(rir))

            result_container["results"][0]["responses"][0]["receiverResults"] = (
                rir.tolist()
            )
            result_container["fs_auralization"] = int(
                deism.params.get("sampleRate", 44100)
            )

            plt.plot(rir)
            plt.savefig(os.path.join(os.path.dirname(json_file_path), "rir.png"))
            plt.close()

            update_result_percentage(result_container, json_file_path, 100)
            print("deism_method: simulation done!")

        except Exception as exc:
            print(f"Error in deism_method: {exc}")
            print(traceback.format_exc())
            raise
        finally:
            os.chdir(original_cwd)
