"""Module implementing a CHORAS interface for deeponet."""
import glob
import json
import os
import re
import shutil
from pathlib import Path

import gmsh
import h5py
import numpy as np
from scipy.io import wavfile

from deeponet_acoustics.end2end.inference import inference
from deeponet_acoustics.end2end.train import train

from ._config import build_dg_config, build_inference_config, build_train_config
from .definition import SimulationMethod


class DeepONetMethod(SimulationMethod):
    """Interface class to run the deeponet method.

    The class implements method to run the calculations for the
    deeponet simulation method. All required configuration parameters
    are expected to be provided in the input JSON file passed during
    initialization.
    """

    def __init__(self, input_json_path: str | Path | None = None):
        """Initialize the deeponet method interface for the given JSON file."""
        super().__init__(input_json_path)

    def run_simulation(self) -> None:
        """Run the simulation."""
        self._deeponet_method(self.input_json_path)

    def _deeponet_method(
        self,
        json_file_path: str | Path,
        output_json_path: str | Path | None = None,
    ) -> None:
        """Run the full deeponet pipeline.

        Pipeline steps:
        1. Load user-facing JSON (read-only) and build the three internal configs
        2. Resolve relative paths against the module directory
        3. Run DG simulation to generate high-fidelity training data
        4. Convert DG NPZ output to HDF5 train/validation splits per source
        5. Train DeepONet
        6. Run DeepONet inference
        7. Write results.json with the predicted impulse responses
        """
        dirname = os.path.dirname(__file__)

        if output_json_path is None:
            output_json_path = os.path.join(
                dirname, "headless_backend", "output", "results.json"
            )

        with open(json_file_path, "r", encoding="utf-8") as file:
            user_input = json.load(file)

        msh_path = Path(_resolve_path(user_input["msh_path"], dirname))
        geo_path = Path(_resolve_path(user_input["geo_path"], dirname))
        resolved_input = {
            **user_input,
            "msh_path": str(msh_path),
            "geo_path": str(geo_path),
        }

        dg_cfg = build_dg_config(resolved_input)
        train_cfg = build_train_config(resolved_input, msh_path)
        inf_cfg = build_inference_config(resolved_input)

        dg_cfg["output_path"] = _resolve_path(dg_cfg["output_path"], dirname)
        train_cfg["input_dir"] = _resolve_path(train_cfg["input_dir"], dirname)
        train_cfg["output_dir"] = _resolve_path(train_cfg["output_dir"], dirname)

        inf_cfg["test_data_dir"] = os.path.join(
            train_cfg["input_dir"], train_cfg["testing_data_dir"]
        )
        inf_cfg["model_dir"] = os.path.join(
            train_cfg["output_dir"], train_cfg["id"]
        )

        dg_json_path = _write_dg_json(dg_cfg, dirname)
        _run_dg_simulation(dg_json_path)

        output_path = dg_cfg["output_path"]
        output_filename = dg_cfg["output_filename"]
        file_format = dg_cfg["file_format"]
        dg_simulation_settings = dg_cfg["simulationSettings"]

        os.makedirs(output_path, exist_ok=True)

        mesh, pressures, time_steps, source_positions, results_dg = (
            _load_and_process_dg_results(output_path, output_filename, file_format)
        )

        # TODO: DG only supports 1 source
        for i, source_position in enumerate(source_positions):
            umesh, upressures, ushape = _process_source_data(results_dg, source_position)

            file_path_train_h5 = os.path.join(
                output_path,
                train_cfg["training_data_dir"],
                f"src{i}",
                f"{output_filename}.h5",
            )
            _save_h5_training_data(
                file_path_train_h5,
                mesh,
                pressures,
                time_steps,
                source_position,
                umesh,
                upressures,
                ushape,
            )

            simulation_params_path_train_json = os.path.join(
                output_path,
                train_cfg["training_data_dir"],
                f"src{i}",
                "simulation_parameters.json",
            )
            _save_simulation_parameters(
                simulation_params_path_train_json,
                source_position,
                dg_simulation_settings,
                results_dg,
            )

            _prepare_validation_data(
                output_path,
                train_cfg,
                i,
                file_path_train_h5,
                simulation_params_path_train_json,
            )

        train(train_cfg)
        inference(train_cfg, inf_cfg)

        _write_results_json(train_cfg, output_json_path)

        print("deeponet simulation completed successfully!")


def _resolve_path(path: str, base_dir: str) -> str:
    """Resolve ``path`` to an absolute path; relative paths join ``base_dir``."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _write_dg_json(dg_cfg: dict, dirname: str) -> str:
    """Persist the DG config to a tmp JSON file consumable by ``dg_method``."""
    tmp_dir = os.path.join(dirname, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    dg_json_path = os.path.join(tmp_dir, "dg_tmp.json")
    with open(dg_json_path, "w", encoding="utf-8") as dg_file:
        json.dump(dg_cfg, dg_file, indent=4)
    return dg_json_path


def _run_dg_simulation(json_file_path: str | Path) -> None:
    """Run the DG simulation, generating NPZ output without touching the user JSON."""
    # Lazy import: dg_method is a sibling method package, not a venv dep.
    # Importing it eagerly would couple unrelated unit tests to the DG install.
    from dg_method.dg_interface.DGinterface import dg_method

    if not gmsh.isInitialized():
        gmsh.initialize()

    try:
        dg_method(json_file_path, save_results_to_json=False)
    finally:
        gmsh.finalize()


def _load_and_process_dg_results(
    output_path: str, output_filename: str, file_format: str
) -> tuple:
    """Load DG results and extract mesh, pressure, time, and source data."""
    results_dg = np.load(os.path.join(output_path, f"{output_filename}.{file_format}"))

    mesh = np.array(results_dg["rec"]).T.astype(np.float64)
    pressures = np.array(results_dg["IR_Uncorrected"]).T.astype(np.float16)
    time_steps = np.linspace(0, results_dg["total_time"], results_dg["Ntimesteps"])
    source_positions = np.array([results_dg["source_xyz"]]).astype(np.float64)

    return mesh, pressures, time_steps, source_positions, results_dg


def _process_source_data(results_dg: dict, source_position: np.ndarray) -> tuple:
    """Process source-specific mesh and pressure data."""
    umesh = np.array(results_dg["IC_mesh"]).T.astype(np.float64)
    upressures = np.array(results_dg["IC_pressure"]).astype(np.float16)

    # TODO: the umesh generated by DGFEM should be uniformly distributed.
    # For now we should be using MLPs not requiring uniform grids.
    # Should be the number of points/elements in the x, y, z dimensions used if e.g.
    # CNNs are used in the branch net
    ushape = np.array([-1, -1, -1]).astype(np.int64)

    # TODO: dg should be fixed to only returning unique mesh points
    print(f"# coordinates from DG: {umesh.shape[0]}")
    umesh, unique_indices = np.unique(umesh, axis=0, return_index=True)
    upressures = upressures[unique_indices]
    print(f"# coordinates after removing duplicates: {umesh.shape[0]}")

    return umesh, upressures, ushape


def _save_h5_training_data(
    file_path_h5: str,
    mesh: np.ndarray,
    pressures: np.ndarray,
    time_steps: np.ndarray,
    source_position: np.ndarray,
    umesh: np.ndarray,
    upressures: np.ndarray,
    ushape: np.ndarray,
) -> None:
    """Save training data to HDF5 format."""
    os.makedirs(os.path.dirname(file_path_h5), exist_ok=True)
    Path(file_path_h5).unlink(missing_ok=True)

    with h5py.File(file_path_h5, "w") as f:
        f.create_dataset("mesh", data=mesh)
        ds_p = f.create_dataset("pressures", data=pressures)
        ds_p.attrs["time_steps"] = time_steps

        f.create_dataset("source_position", data=source_position)

        ds_umesh = f.create_dataset("umesh", data=umesh)
        ds_umesh.attrs["umesh_shape"] = ushape
        f.create_dataset("upressures", data=upressures)


def _save_simulation_parameters(
    file_path_json: str,
    source_position: np.ndarray,
    dg_settings: dict,
    results_dg: dict,
) -> None:
    """Save simulation parameters next to the training data."""
    os.makedirs(os.path.dirname(file_path_json), exist_ok=True)

    simulation_params = {
        "SimulationParameters": {
            "SourcePosition": source_position.tolist(),
            "c": dg_settings["dg_c0"],
            "dt": results_dg["dt_old"].tolist(),
            "fmax": dg_settings["dg_freq_upper_limit"],
            "rho": dg_settings["dg_rho0"],
        }
    }

    with open(file_path_json, "w") as json_file:
        json.dump(simulation_params, json_file, indent=4)


def _prepare_validation_data(
    output_path: str,
    train_cfg: dict,
    source_index: int,
    train_h5_path: str,
    train_params_path: str,
) -> None:
    """Prepare validation data by copying training data into the val tree."""
    file_path_val_h5 = os.path.join(
        output_path,
        train_cfg["testing_data_dir"],
        f"src{source_index}",
        os.path.basename(train_h5_path),
    )
    os.makedirs(os.path.dirname(file_path_val_h5), exist_ok=True)
    Path(file_path_val_h5).unlink(missing_ok=True)
    shutil.copy(train_h5_path, file_path_val_h5)

    simulation_params_path_root_json = os.path.join(
        output_path, train_cfg["training_data_dir"], "simulation_parameters.json"
    )
    Path(simulation_params_path_root_json).unlink(missing_ok=True)
    shutil.copy(train_params_path, simulation_params_path_root_json)

    simulation_params_path_val_json = os.path.join(
        output_path,
        train_cfg["testing_data_dir"],
        f"src{source_index}",
        "simulation_parameters.json",
    )
    Path(simulation_params_path_val_json).unlink(missing_ok=True)
    shutil.copy(train_params_path, simulation_params_path_val_json)


def _parse_receiver_position_from_filename(
    filename: str,
) -> tuple[list[float], list[float]]:
    """Parse source and receiver positions from a prediction WAV filename.

    Expected format: ``0_x0=['0.40', '1.00', '1.20']_r0=['1.00', '1.00', '1.10']_pred.wav``
    """
    x0_match = re.search(r"x0=\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]", filename)
    r0_match = re.search(r"r0=\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]", filename)

    if not x0_match or not r0_match:
        raise ValueError(f"Could not parse positions from filename: {filename}")

    source_pos = [float(x0_match.group(i)) for i in (1, 2, 3)]
    receiver_pos = [float(r0_match.group(i)) for i in (1, 2, 3)]

    return source_pos, receiver_pos


def _read_wav_impulse_response(wav_path: str) -> list[float]:
    """Read impulse response from a WAV file as a list of floats."""
    sample_rate, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    if len(data.shape) > 1:
        data = data[:, 0]

    return data.tolist()


def _write_results_json(
    train_cfg: dict, output_json_path: str | Path
) -> None:
    """Write ``results.json`` with the predicted impulse responses.

    Reads the ``*_pred.wav`` files produced by DeepONet inference and
    populates a ``results`` list grouped by source position.
    """
    receivers_dir = os.path.join(
        train_cfg["output_dir"], train_cfg["id"], "figs", "receivers"
    )

    pred_wav_files = glob.glob(os.path.join(receivers_dir, "*_pred.wav"))

    if not pred_wav_files:
        print(f"Warning: No prediction WAV files found in {receivers_dir}")
        return

    results_by_source: dict[tuple[float, ...], dict] = {}

    for wav_file in pred_wav_files:
        filename = os.path.basename(wav_file)

        try:
            source_pos, receiver_pos = _parse_receiver_position_from_filename(filename)
            ir_data = _read_wav_impulse_response(wav_file)

            source_key = tuple(source_pos)

            if source_key not in results_by_source:
                results_by_source[source_key] = {
                    "sourceX": source_pos[0],
                    "sourceY": source_pos[1],
                    "sourceZ": source_pos[2],
                    "resultType": "DON",
                    "percentage": 100,
                    "responses": [],
                }

            results_by_source[source_key]["responses"].append(
                {
                    "x": receiver_pos[0],
                    "y": receiver_pos[1],
                    "z": receiver_pos[2],
                    "receiverResults": [],
                    "receiverResultsUncorrected": ir_data,
                }
            )

        except Exception as e:
            print(f"Warning: Could not process {filename}: {e}")
            continue

    output_data = {"results": list(results_by_source.values())}

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    Path(output_json_path).unlink(missing_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4)

    print(f"Results written to: {output_json_path}")
