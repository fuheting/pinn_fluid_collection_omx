"""Tests for Phase 11 result procurement runner manifests."""

import json
import math

from pinn_fluid.experiments import ExperimentConfig
from pinn_fluid.result_procurement import run_result_procurement


def test_result_procurement_writes_manifest_for_configured_output_dir(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "sanity",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=0,
        viscosity=0.25,
        darcy_reference_iterations=20,
    )

    manifest = run_result_procurement(config, git_commit="test-commit")

    manifest_path = tmp_path / "sanity" / "run_manifest.json"
    assert manifest_path.is_file()
    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest == manifest
    assert manifest["git_commit"] == "test-commit"
    assert manifest["output_dir"] == str(tmp_path / "sanity")
    assert manifest["config"]["grid_points"] == 5
    assert manifest["all_metrics_finite"] is True
    assert manifest["all_histories_decreased"] is True
    assert [entry["model"] for entry in manifest["results"]] == [
        "darcy",
        "stokes",
        "oseen",
        "navier_stokes",
    ]

    for entry in manifest["results"]:
        assert entry["output_dir"] == str(tmp_path / "sanity" / entry["model"])
        assert entry["history_decreased"] is True
        assert entry["metrics_finite"] is True
        assert entry["history"]["final_total"] < entry["history"]["initial_total"]
        assert entry["history"]["steps"] == 4
        assert all(math.isfinite(value) for value in entry["metrics"].values())


def test_result_procurement_manifest_points_to_existing_artifacts(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "artifacts",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=1,
        viscosity=0.25,
        darcy_reference_iterations=20,
    )

    manifest = run_result_procurement(config, git_commit="test-commit")

    for entry in manifest["results"]:
        for relative_path in entry["artifacts"].values():
            assert (tmp_path / "artifacts" / relative_path).is_file()
    assert (tmp_path / "artifacts" / "summary" / "cross_model_report.json").is_file()
    assert (tmp_path / "artifacts" / "summary" / "cross_model_report.md").is_file()
