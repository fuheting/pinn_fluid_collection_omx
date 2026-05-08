"""Phase 20 contracts for model-specific comparison procurement."""

import json

import numpy as np

from pinn_fluid.experiments import ExperimentConfig
from pinn_fluid.figures import generate_figure_bundle
from pinn_fluid.result_procurement import run_result_procurement


EXPECTED_REFERENCE_GENERATORS = {
    "darcy": "fd_darcy_reference",
    "stokes": "stokes_streamfunction_reference",
    "oseen": "oseen_streamfunction_reference",
    "navier_stokes": "navier_stokes_streamfunction_reference",
}


def test_procurement_manifest_summarizes_model_specific_reference_generators(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "comparison",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=12,
        viscosity=0.25,
        darcy_reference_iterations=20,
    )

    manifest = run_result_procurement(config, git_commit="phase20-test")

    assert manifest["reference_generators"] == EXPECTED_REFERENCE_GENERATORS
    assert manifest["all_metrics_finite"] is True
    assert manifest["all_histories_decreased"] is True
    for entry in manifest["results"]:
        metadata = entry["reference_metadata"]
        assert metadata["reference_generator_name"] == EXPECTED_REFERENCE_GENERATORS[entry["model"]]
        assert metadata["reference_kind"] in {"finite-difference", "manufactured"}
        assert "reference_continuity_rms" in entry["metrics"] or entry["model"] == "darcy"


def test_procured_model_specific_bundle_has_required_fields_and_figures(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "comparison",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=13,
        viscosity=0.25,
        darcy_reference_iterations=20,
    )

    manifest = run_result_procurement(config, git_commit="phase20-test")
    figure_manifest = generate_figure_bundle(config.output_dir)

    assert [entry["model"] for entry in figure_manifest["models"]] == [
        "darcy",
        "stokes",
        "oseen",
        "navier_stokes",
    ]
    for entry in manifest["results"]:
        model_dir = tmp_path / "comparison" / entry["model"]
        assert (model_dir / "fields.npz").is_file()
        assert (model_dir / "metrics.json").is_file()
        assert (model_dir / "history.json").is_file()
        with np.load(model_dir / "fields.npz") as fields:
            assert "reference_metadata_json" in fields.files
            metadata = json.loads(str(fields["reference_metadata_json"]))
            assert metadata["reference_generator_name"] == EXPECTED_REFERENCE_GENERATORS[entry["model"]]
            if entry["model"] != "darcy":
                assert {
                    "reference_speed",
                    "reference_continuity",
                    "reference_x_momentum",
                    "reference_y_momentum",
                    "reference_residual",
                }.issubset(fields.files)

    for entry in figure_manifest["models"]:
        assert (tmp_path / "comparison" / entry["field_panel"]).is_file()
        assert (tmp_path / "comparison" / entry["convergence_panel"]).is_file()
        assert (tmp_path / "comparison" / entry["pressure_velocity_quiver"]).is_file()
