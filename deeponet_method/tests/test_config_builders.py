"""Unit tests for the user-JSON to internal-config builders."""
from pathlib import Path

import pytest

from deeponet_interface._config import (
    DG_CFL,
    DG_FILE_FORMAT,
    DG_OUTPUT_FILENAME,
    DG_OUTPUT_PATH,
    DG_PPW,
    DG_RHO0,
    INFERENCE_RECV_GROUPS,
    INFERENCE_TMAX,
    TRAIN_DATA_DIR,
    TRAIN_INPUT_DIR,
    TRAIN_OUTPUT_DIR,
    TRAIN_TMAX,
    VAL_DATA_DIR,
    build_dg_config,
    build_inference_config,
    build_train_config,
)


def test_build_dg_config_simulation_settings(default_input_data):
    cfg = build_dg_config(default_input_data)
    sim = cfg["simulationSettings"]

    user_sim = default_input_data["simulationSettings"]
    assert sim["dg_freq_upper_limit"] == user_sim["fmax"]
    assert sim["dg_c0"] == user_sim["c0"]
    assert sim["dg_ir_length"] == user_sim["ir_length"]
    assert sim["dg_rho0"] == DG_RHO0
    assert sim["dg_ppw"] == DG_PPW
    assert sim["dg_cfl"] == DG_CFL


def test_build_dg_config_output_block(default_input_data):
    cfg = build_dg_config(default_input_data)
    assert cfg["output_path"] == DG_OUTPUT_PATH
    assert cfg["output_filename"] == DG_OUTPUT_FILENAME
    assert cfg["file_format"] == DG_FILE_FORMAT


def test_build_dg_config_passthrough_paths_and_geometry(default_input_data):
    cfg = build_dg_config(default_input_data)
    assert cfg["msh_path"] == default_input_data["msh_path"]
    assert cfg["geo_path"] == default_input_data["geo_path"]
    assert cfg["absorption_coefficients"] == default_input_data["absorption_coefficients"]


def test_build_dg_config_marks_results_as_DON(default_input_data):
    cfg = build_dg_config(default_input_data)
    for entry in cfg["results"]:
        assert entry["resultType"] == "DON"


def test_build_dg_config_does_not_mutate_user_input(default_input_data):
    original_result_type = default_input_data["results"][0].get("resultType")
    build_dg_config(default_input_data)
    assert default_input_data["results"][0].get("resultType") == original_result_type


def test_build_train_config_id_from_msh_stem(default_input_data):
    cfg = build_train_config(
        default_input_data, Path("/some/abs/path/test_room_deeponet.msh")
    )
    assert cfg["id"] == "test_room_deeponet"


def test_build_train_config_id_accepts_str_path(default_input_data):
    cfg = build_train_config(default_input_data, "another_room.msh")
    assert cfg["id"] == "another_room"


def test_build_train_config_hardcoded_defaults(default_input_data):
    cfg = build_train_config(default_input_data, Path("x.msh"))
    assert cfg["input_dir"] == TRAIN_INPUT_DIR
    assert cfg["output_dir"] == TRAIN_OUTPUT_DIR
    assert cfg["training_data_dir"] == TRAIN_DATA_DIR
    assert cfg["testing_data_dir"] == VAL_DATA_DIR
    assert cfg["tmax"] == TRAIN_TMAX
    assert cfg["normalize_data"] is True
    assert cfg["use_adaptive_weights"] is True


@pytest.mark.parametrize(
    "key",
    [
        "f0_feat",
        "iterations",
        "decay_steps",
        "decay_rate",
        "learning_rate",
        "optimizer",
        "batch_size_branch",
        "batch_size_coord",
        "num_output_neurons",
    ],
)
def test_build_train_config_passes_user_field_through(default_input_data, key):
    cfg = build_train_config(default_input_data, Path("x.msh"))
    assert cfg[key] == default_input_data["simulationSettings"][key]


def test_build_train_config_branch_net_nesting(default_input_data):
    cfg = build_train_config(default_input_data, Path("x.msh"))
    sim = default_input_data["simulationSettings"]
    assert cfg["branch_net"] == {
        "architecture": sim["bn_architecture"],
        "activation": sim["bn_activation"],
        "num_hidden_layers": sim["bn_num_hidden_layers"],
        "num_hidden_neurons": sim["bn_num_hidden_neurons"],
    }


def test_build_train_config_trunk_net_nesting(default_input_data):
    cfg = build_train_config(default_input_data, Path("x.msh"))
    sim = default_input_data["simulationSettings"]
    assert cfg["trunk_net"] == {
        "architecture": sim["tn_architecture"],
        "activation": sim["tn_activation"],
        "num_hidden_layers": sim["tn_num_hidden_layers"],
        "num_hidden_neurons": sim["tn_num_hidden_neurons"],
    }


def test_build_train_config_no_flat_bn_or_tn_keys(default_input_data):
    cfg = build_train_config(default_input_data, Path("x.msh"))
    for key in cfg:
        assert not key.startswith("bn_")
        assert not key.startswith("tn_")


def test_build_inference_config_hardcoded_flags(default_input_data):
    cfg = build_inference_config(default_input_data)
    assert cfg["tmax"] == INFERENCE_TMAX
    assert cfg["write_full_wave_field"] is False
    assert cfg["snap_to_grid"] is True
    assert cfg["write_ir_plots"] is True
    assert cfg["write_ir_animations"] is False
    assert cfg["write_ir_wav"] is True
    assert cfg["receiver_position_groups"] == INFERENCE_RECV_GROUPS


def test_build_inference_config_recv_positions_dedupe():
    user_input = {
        "results": [
            {
                "sourceX": 1, "sourceY": 2, "sourceZ": 3,
                "responses": [
                    {"x": 5.0, "y": 3.5, "z": 1.5},
                    {"x": 5.0, "y": 3.5, "z": 1.5},
                    {"x": 4.0, "y": 2.0, "z": 1.0},
                ],
            },
            {
                "sourceX": 9, "sourceY": 9, "sourceZ": 9,
                "responses": [
                    {"x": 4.0, "y": 2.0, "z": 1.0},
                    {"x": 7.0, "y": 7.0, "z": 7.0},
                ],
            },
        ]
    }
    cfg = build_inference_config(user_input)
    assert cfg["recv_positions"] == [
        [5.0, 3.5, 1.5],
        [4.0, 2.0, 1.0],
        [7.0, 7.0, 7.0],
    ]


def test_build_inference_config_single_receiver(default_input_data):
    cfg = build_inference_config(default_input_data)
    response = default_input_data["results"][0]["responses"][0]
    assert cfg["recv_positions"] == [[response["x"], response["y"], response["z"]]]


def test_build_inference_config_empty_results():
    cfg = build_inference_config({"results": []})
    assert cfg["recv_positions"] == []
