"""Tests for Darcy and Navier-Stokes experiment vertical slices."""

import json

import numpy as np

from pinn_fluid.experiments import (
    ExperimentConfig,
    run_darcy_experiment,
    run_poiseuille_navier_stokes_experiment,
)


def test_darcy_experiment_saves_fields_history_residuals_and_metrics(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path,
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=0,
    )

    result = run_darcy_experiment(config)

    assert result.model == "darcy"
    assert result.reference == "finite_difference_laplace"
    assert result.history.reduced
    assert set(result.history.components) == {"interior", "inlet", "outlet", "wall"}
    assert result.metrics["pressure_l2"] >= 0.0
    assert result.metrics["velocity_l2"] >= 0.0
    assert result.metrics["residual_rms"] >= 0.0

    field_data = np.load(tmp_path / result.artifacts["fields_npz"])
    assert {"coordinates", "predicted_pressure", "reference_pressure", "residual"}.issubset(
        field_data.files
    )
    assert field_data["coordinates"].shape == (25, 2)

    history_payload = json.loads((tmp_path / result.artifacts["history_json"]).read_text())
    assert history_payload["total"] == result.history.total
    assert (tmp_path / result.artifacts["loss_plot_png"]).is_file()
    assert (tmp_path / result.artifacts["field_plot_png"]).is_file()


def test_navier_stokes_experiment_uses_model_specific_reference(tmp_path, openfoam_config_factory):
    config = openfoam_config_factory(
        output_dir=tmp_path,
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=1,
        viscosity=0.25,
    )

    result = run_poiseuille_navier_stokes_experiment(config)

    assert result.model == "navier_stokes"
    assert result.reference == "openfoam_simplefoam_shared_domain"
    assert result.history.reduced
    assert set(result.history.components) == {
        "continuity",
        "x_momentum",
        "y_momentum",
        "inlet",
        "outlet",
        "wall",
    }
    assert result.metrics["velocity_l2"] >= 0.0
    assert result.metrics["pressure_l2"] >= 0.0
    assert result.metrics["residual_rms"] >= 0.0

    field_data = np.load(tmp_path / result.artifacts["fields_npz"])
    assert {"predicted_u", "predicted_v", "reference_u", "reference_v", "residual"}.issubset(
        field_data.files
    )
    assert field_data["coordinates"].shape == (25, 2)
    assert {"reference_continuity", "reference_x_momentum", "reference_y_momentum"}.issubset(
        field_data.files
    )
