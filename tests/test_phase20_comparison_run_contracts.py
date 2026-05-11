"""Phase 20 contracts for model-specific comparison procurement."""

import json

import numpy as np

from pinn_fluid.experiments import ExperimentConfig
from pinn_fluid.figures import generate_figure_bundle
from pinn_fluid.result_procurement import run_result_procurement


EXPECTED_REFERENCE_GENERATORS = {
    "darcy": "fenicsx_darcy_shared_domain",
    "stokes": "fenicsx_stokes_shared_domain",
    "oseen": "fenicsx_oseen_shared_domain",
    "navier_stokes": "fenicsx_navier_stokes_shared_domain",
}


def _openfoam_sample_paths(tmp_path, openfoam_sample_writer):
    sample_paths = {}
    for model in EXPECTED_REFERENCE_GENERATORS:
        sample_dir = tmp_path / model
        sample_dir.mkdir()
        sample_paths[model] = openfoam_sample_writer(sample_dir / "sharedDomainGrid.xy", grid_points=5)
    return sample_paths


def test_procurement_manifest_summarizes_model_specific_reference_generators(
    tmp_path,
    openfoam_sample_writer,
):
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
        fenicsx_reference_sample_paths=_openfoam_sample_paths(tmp_path, openfoam_sample_writer),
    )

    manifest = run_result_procurement(config, git_commit="phase20-test")

    assert manifest["reference_generators"] == EXPECTED_REFERENCE_GENERATORS
    assert manifest["all_metrics_finite"] is True
    assert manifest["all_histories_decreased"] is True
    for entry in manifest["results"]:
        metadata = entry["reference_metadata"]
        assert metadata["reference_generator_name"] == EXPECTED_REFERENCE_GENERATORS[entry["model"]]
        assert metadata["reference_kind"] == "dedicated-solver"
        if entry["model"] != "darcy":
            assert "reference_continuity_rms" in entry["metrics"]


def test_procured_model_specific_bundle_has_required_fields_and_figures(
    tmp_path,
    openfoam_sample_writer,
):
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
        fenicsx_reference_sample_paths=_openfoam_sample_paths(tmp_path, openfoam_sample_writer),
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
            if entry["model"] == "darcy":
                assert {
                    "reference_velocity",
                    "reference_u",
                    "reference_v",
                    "reference_speed",
                    "reference_residual",
                }.issubset(fields.files)
            else:
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
