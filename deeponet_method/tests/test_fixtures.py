"""Shape assertions for the user-facing test input JSON."""
import os


def test_user_facing_required_top_level_keys(default_input_data):
    for key in ("geo_path", "msh_path", "absorption_coefficients",
                "simulationSettings", "results"):
        assert key in default_input_data


def test_user_facing_simulation_settings(default_input_data):
    sim = default_input_data["simulationSettings"]
    for key in ("fmax", "c0", "ir_length", "iterations",
                "decay_steps", "decay_rate", "learning_rate", "optimizer",
                "batch_size_branch", "batch_size_coord",
                "bn_architecture", "bn_activation",
                "bn_num_hidden_layers", "bn_num_hidden_neurons",
                "tn_architecture", "tn_activation",
                "tn_num_hidden_layers", "tn_num_hidden_neurons",
                "num_output_neurons"):
        assert key in sim, f"missing simulationSettings.{key}"


def test_user_facing_no_array_fields(default_input_data):
    """The settings schema doesn't support arrays — they must be hardcoded in code."""
    assert "f0_feat" not in default_input_data["simulationSettings"]


def test_user_facing_results_shape(default_input_data):
    assert len(default_input_data["results"]) > 0
    first = default_input_data["results"][0]
    assert "sourceX" in first
    assert "sourceY" in first
    assert "sourceZ" in first
    assert "responses" in first
    assert len(first["responses"]) > 0


def test_user_facing_no_legacy_merged_keys(default_input_data):
    """Guard against re-introducing the pre-refactor merged config shape."""
    for legacy_key in ("dg_setup", "deeponet_train_setup", "deeponet_inference_setup"):
        assert legacy_key not in default_input_data


def test_create_temporary_input_file_fixture(create_temporary_input_file):
    assert os.path.exists(create_temporary_input_file)
    assert create_temporary_input_file.endswith(".json")
