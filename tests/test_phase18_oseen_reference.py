"""Phase 18 tests for Oseen-specific actual fields."""

import json

import numpy as np

import pinn_fluid.experiments as experiments
from pinn_fluid.experiments import (
    ExperimentConfig,
    _grid_coordinates,
    run_oseen_experiment,
    run_poiseuille_navier_stokes_experiment,
)
from pinn_fluid.figures import generate_figure_bundle


def test_oseen_reference_fields_are_divergence_free_and_document_convection():
    assert hasattr(experiments, "_oseen_reference_fields")

    fields = experiments._oseen_reference_fields(
        grid_points=17,
        viscosity=0.25,
        peak_velocity=1.0,
    )

    assert set(fields) == {
        "u",
        "v",
        "pressure",
        "velocity",
        "speed",
        "continuity",
        "x_momentum",
        "y_momentum",
        "residual",
        "convection_velocity",
    }
    assert fields["u"].shape == (289, 1)
    assert fields["v"].shape == (289, 1)
    assert fields["pressure"].shape == (289, 1)
    assert fields["velocity"].shape == (289, 2)
    assert fields["convection_velocity"].shape == (289, 2)
    assert np.allclose(fields["convection_velocity"], np.array([[1.0, 0.0]] * 289))
    assert np.allclose(fields["velocity"], np.column_stack((fields["u"], fields["v"])))
    assert np.max(np.abs(fields["continuity"])) < 1e-10
    assert np.all(np.isfinite(fields["x_momentum"]))
    assert np.all(np.isfinite(fields["y_momentum"]))
    assert np.all(np.isfinite(fields["residual"]))


def test_oseen_reference_boundary_behavior_matches_shared_openings():
    grid_points = 17
    coordinates = _grid_coordinates(grid_points).numpy()
    fields = experiments._oseen_reference_fields(
        grid_points=grid_points,
        viscosity=0.25,
        peak_velocity=1.0,
    )
    u = fields["u"].reshape(grid_points, grid_points)
    v = fields["v"].reshape(grid_points, grid_points)

    top_inlet = (
        (np.isclose(coordinates[:, 1], 1.0))
        & (coordinates[:, 0] > 0.0)
        & (coordinates[:, 0] < 0.25)
    ).reshape(grid_points, grid_points)
    bottom_outlet = (
        (np.isclose(coordinates[:, 1], 0.0))
        & (coordinates[:, 0] > 0.75)
        & (coordinates[:, 0] < 1.0)
    ).reshape(grid_points, grid_points)

    assert np.mean(v[top_inlet]) < 0.0
    assert np.mean(v[bottom_outlet]) < 0.0
    assert np.allclose(u[:, -1], 0.0, atol=1e-10)
    assert np.allclose(u[:, 0], 0.0, atol=1e-10)
    assert np.allclose(v[5:, -1], 0.0, atol=1e-10)
    assert np.allclose(v[:12, 0], 0.0, atol=1e-10)


def test_run_oseen_experiment_uses_oseen_specific_reference_and_schema(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path,
        grid_points=9,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=9,
        viscosity=0.25,
        peak_velocity=1.0,
    )

    result = run_oseen_experiment(config)

    assert result.model == "oseen"
    assert result.reference == "manufactured_oseen_streamfunction"
    assert result.reference_metadata["reference_generator_name"] == "oseen_streamfunction_reference"
    assert result.reference_metadata["pde_model_represented"] == "Oseen incompressible momentum and continuity"
    assert result.reference_metadata["reference_kind"] == "manufactured"
    assert result.reference_metadata["convection_velocity"] == [1.0, 0.0]
    assert result.metrics["velocity_l2"] >= 0.0
    assert result.metrics["pressure_l2"] >= 0.0
    assert result.metrics["residual_rms"] >= 0.0
    assert result.metrics["reference_continuity_rms"] < 1e-10
    assert result.metrics["reference_momentum_rms"] >= 0.0

    with np.load(tmp_path / result.artifacts["fields_npz"]) as fields:
        assert {
            "predicted_u",
            "predicted_v",
            "predicted_pressure",
            "reference_u",
            "reference_v",
            "reference_pressure",
            "reference_speed",
            "reference_continuity",
            "reference_x_momentum",
            "reference_y_momentum",
            "reference_residual",
            "reference_convection_velocity",
            "residual",
            "reference_metadata_json",
        }.issubset(fields.files)
        assert fields["coordinates"].shape == (81, 2)
        assert np.allclose(fields["reference_convection_velocity"], np.array([[1.0, 0.0]] * 81))
        metadata = json.loads(str(fields["reference_metadata_json"]))
        assert metadata == result.reference_metadata

    manifest = generate_figure_bundle(tmp_path, models=("oseen",))
    assert manifest["models"][0]["model"] == "oseen"
    assert (tmp_path / manifest["models"][0]["field_panel"]).is_file()
    assert (tmp_path / manifest["models"][0]["pressure_velocity_quiver"]).is_file()


def test_navier_stokes_remains_on_shared_demo_reference(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path,
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=10,
        viscosity=0.25,
        darcy_reference_iterations=20,
    )

    navier_stokes = run_poiseuille_navier_stokes_experiment(config)

    assert navier_stokes.reference == "shared_patch_unit_square_reference"
    assert navier_stokes.reference_metadata["reference_kind"] == "demo-only"
