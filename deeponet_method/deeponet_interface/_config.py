"""Config builders that map the user-facing JSON to the three internal
configs consumed by the DG simulation, DeepONet training, and DeepONet
inference. Pure functions: no I/O, no path resolution, no globals.
"""
from copy import deepcopy
from pathlib import Path


DG_RHO0 = 1.213
DG_PPW = 2
DG_CFL = 1
DG_OUTPUT_PATH = "tmp/deeponet/"
DG_OUTPUT_FILENAME = "dg_sim_results"
DG_FILE_FORMAT = "npz"

TRAIN_INPUT_DIR = "tmp/deeponet/"
TRAIN_OUTPUT_DIR = "tmp/deeponet/results/"
TRAIN_DATA_DIR = "train_data"
VAL_DATA_DIR = "val_data"
TRAIN_TMAX = 1000
TRAIN_NORMALIZE_DATA = True
TRAIN_USE_ADAPTIVE_WEIGHTS = True

INFERENCE_TMAX = 1000
INFERENCE_WRITE_FULL_WAVE_FIELD = False
INFERENCE_SNAP_TO_GRID = True
INFERENCE_WRITE_IR_PLOTS = True
INFERENCE_WRITE_IR_ANIMATIONS = False
INFERENCE_WRITE_IR_WAV = True
INFERENCE_RECV_GROUPS = ["recv_positions"]

_NET_FIELDS = ("architecture", "activation", "num_hidden_layers", "num_hidden_neurons")


def build_dg_config(user_input: dict) -> dict:
    """Build the on-disk DG JSON consumed by ``dg_method``.

    Carries through ``msh_path``, ``geo_path``, ``absorption_coefficients``,
    and ``results`` (with ``resultType`` set to ``"DON"`` so DG knows it's
    being called from the deeponet wrapper).
    """
    sim = user_input["simulationSettings"]
    results = deepcopy(user_input["results"])
    for entry in results:
        entry["resultType"] = "DON"

    return {
        "simulationSettings": {
            "dg_freq_upper_limit": sim["fmax"],
            "dg_c0": sim["c0"],
            "dg_rho0": DG_RHO0,
            "dg_ir_length": sim["ir_length"],
            "dg_ppw": DG_PPW,
            "dg_cfl": DG_CFL,
        },
        "output_path": DG_OUTPUT_PATH,
        "output_filename": DG_OUTPUT_FILENAME,
        "file_format": DG_FILE_FORMAT,
        "msh_path": user_input["msh_path"],
        "geo_path": user_input["geo_path"],
        "absorption_coefficients": deepcopy(user_input["absorption_coefficients"]),
        "results": results,
    }


def build_train_config(user_input: dict, msh_path: str | Path) -> dict:
    """Build the DeepONet training config.

    ``id`` is derived from the mesh filename stem so the model output
    directory is tied to the geometry it was trained on.
    """
    sim = user_input["simulationSettings"]
    stem = Path(msh_path).stem

    return {
        "id": stem,
        "input_dir": TRAIN_INPUT_DIR,
        "output_dir": TRAIN_OUTPUT_DIR,
        "training_data_dir": TRAIN_DATA_DIR,
        "testing_data_dir": VAL_DATA_DIR,
        "tmax": TRAIN_TMAX,
        "f0_feat": list(sim["f0_feat"]),
        "normalize_data": TRAIN_NORMALIZE_DATA,
        "iterations": sim["iterations"],
        "use_adaptive_weights": TRAIN_USE_ADAPTIVE_WEIGHTS,
        "decay_steps": sim["decay_steps"],
        "decay_rate": sim["decay_rate"],
        "learning_rate": sim["learning_rate"],
        "optimizer": sim["optimizer"],
        "batch_size_branch": sim["batch_size_branch"],
        "batch_size_coord": sim["batch_size_coord"],
        "branch_net": _build_net_subdict(sim, "bn_"),
        "trunk_net": _build_net_subdict(sim, "tn_"),
        "num_output_neurons": sim["num_output_neurons"],
    }


def build_inference_config(user_input: dict) -> dict:
    """Build the DeepONet inference config.

    ``recv_positions`` is the flattened, order-preserving deduped list of
    ``(x, y, z)`` triples drawn from every ``results[].responses[]``.
    """
    return {
        "tmax": INFERENCE_TMAX,
        "write_full_wave_field": INFERENCE_WRITE_FULL_WAVE_FIELD,
        "snap_to_grid": INFERENCE_SNAP_TO_GRID,
        "write_ir_plots": INFERENCE_WRITE_IR_PLOTS,
        "write_ir_animations": INFERENCE_WRITE_IR_ANIMATIONS,
        "write_ir_wav": INFERENCE_WRITE_IR_WAV,
        "recv_positions": _collect_recv_positions(user_input["results"]),
        "receiver_position_groups": list(INFERENCE_RECV_GROUPS),
    }


def _build_net_subdict(sim: dict, prefix: str) -> dict:
    """Collapse ``bn_*`` or ``tn_*`` prefixed user fields into a nested net dict."""
    return {field: sim[f"{prefix}{field}"] for field in _NET_FIELDS}


def _collect_recv_positions(results: list[dict]) -> list[list[float]]:
    """Flatten ``results[].responses[].(x, y, z)`` across all sources, dedupe
    preserving first-seen order.
    """
    seen: set[tuple[float, float, float]] = set()
    positions: list[list[float]] = []
    for entry in results:
        for response in entry.get("responses", []):
            triple = (response["x"], response["y"], response["z"])
            if triple in seen:
                continue
            seen.add(triple)
            positions.append([triple[0], triple[1], triple[2]])
    return positions
