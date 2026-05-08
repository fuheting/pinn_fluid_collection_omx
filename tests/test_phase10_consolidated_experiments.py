"""Tests for Stokes/Oseen experiment extension and consolidated report."""

import json

from pinn_fluid.experiments import ExperimentConfig, run_all_experiments


def test_run_all_experiments_adds_stokes_oseen_and_report(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path,
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=2,
        viscosity=0.25,
    )

    results = run_all_experiments(config)

    assert [result.model for result in results] == [
        "darcy",
        "stokes",
        "oseen",
        "navier_stokes",
    ]
    assert all(result.history.reduced for result in results)
    assert all(result.metrics["velocity_l2"] >= 0.0 for result in results)
    assert all(result.metrics["residual_rms"] >= 0.0 for result in results)
    assert results[1].reference == "manufactured_stokes_streamfunction"
    assert results[2].reference == "manufactured_oseen_streamfunction"
    assert results[3].reference == "manufactured_navier_stokes_streamfunction"
    for result in results[1:]:
        assert {"inlet", "outlet", "wall"}.issubset(result.history.components)

    report = json.loads((tmp_path / "summary" / "cross_model_report.json").read_text())
    assert [entry["model"] for entry in report["results"]] == [
        "darcy",
        "stokes",
        "oseen",
        "navier_stokes",
    ]
    assert (tmp_path / "summary" / "cross_model_report.md").is_file()
