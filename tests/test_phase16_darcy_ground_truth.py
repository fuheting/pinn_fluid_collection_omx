"""Phase 16 tests for hardened Darcy actual fields."""

import json

import numpy as np

import pinn_fluid.experiments as experiments
from pinn_fluid.experiments import (
    ExperimentConfig,
    _fd_darcy_reference,
    _grid_coordinates,
    run_darcy_experiment,
)


def test_fd_darcy_pressure_extrema_are_on_inlet_and_outlet_patches():
    grid_points = 17
    pressure, _ = _fd_darcy_reference(grid_points=grid_points, iterations=500)
    pressure_grid = pressure.reshape(grid_points, grid_points)
    top_inlet_end = round(0.25 * (grid_points - 1))
    bottom_outlet_start = round(0.75 * (grid_points - 1))

    inlet_pressure = pressure_grid[: top_inlet_end + 1, -1]
    outlet_pressure = pressure_grid[bottom_outlet_start:, 0]
    interior_pressure = pressure_grid[1:-1, 1:-1]

    assert np.allclose(inlet_pressure, 1.0)
    assert np.allclose(outlet_pressure, 0.0)
    assert np.max(interior_pressure) < 1.0
    assert np.min(interior_pressure) > 0.0


def test_fd_darcy_velocity_points_from_high_pressure_toward_low_pressure():
    grid_points = 17
    coordinates = _grid_coordinates(grid_points).numpy()
    _, velocity = _fd_darcy_reference(grid_points=grid_points, iterations=500)
    flow_axis = np.array([0.75, -1.0], dtype=np.float64)
    flow_axis = flow_axis / np.linalg.norm(flow_axis)

    interior = (
        (coordinates[:, 0] > 0.2)
        & (coordinates[:, 0] < 0.8)
        & (coordinates[:, 1] > 0.2)
        & (coordinates[:, 1] < 0.8)
    )

    assert np.mean(velocity[interior] @ flow_axis) > 0.0


def test_fd_darcy_walls_have_no_normal_flow_away_from_openings():
    grid_points = 17
    _, velocity = _fd_darcy_reference(grid_points=grid_points, iterations=500)
    u = velocity[:, 0].reshape(grid_points, grid_points)
    v = velocity[:, 1].reshape(grid_points, grid_points)
    top_inlet_end = round(0.25 * (grid_points - 1))
    bottom_outlet_start = round(0.75 * (grid_points - 1))

    assert np.allclose(u[0, 1:-1], 0.0, atol=1e-12)
    assert np.allclose(u[-1, 1:-1], 0.0, atol=1e-12)
    assert np.allclose(v[:bottom_outlet_start, 0], 0.0, atol=1e-12)
    assert np.allclose(v[top_inlet_end + 1 :, -1], 0.0, atol=1e-12)


def test_fd_darcy_reference_fields_expose_pressure_components_speed_and_residuals():
    assert hasattr(experiments, "_fd_darcy_reference_fields")
    fields = experiments._fd_darcy_reference_fields(grid_points=9, iterations=500)

    assert set(fields) == {
        "pressure",
        "velocity",
        "u",
        "v",
        "speed",
        "residual",
    }
    for values in fields.values():
        assert values.shape[0] == 81
        assert np.all(np.isfinite(values))
    assert fields["pressure"].shape == (81, 1)
    assert fields["u"].shape == (81, 1)
    assert fields["v"].shape == (81, 1)
    assert fields["speed"].shape == (81, 1)
    assert fields["residual"].shape == (81, 1)
    assert np.allclose(fields["u"], fields["velocity"][:, :1])
    assert np.allclose(fields["v"], fields["velocity"][:, 1:2])
    assert np.allclose(fields["speed"], np.linalg.norm(fields["velocity"], axis=1, keepdims=True))
    assert float(np.max(np.abs(fields["residual"]))) < 0.02


def test_darcy_experiment_saves_explicit_actual_field_components_and_diagnostics(
    tmp_path,
    openfoam_config_factory,
):
    config = openfoam_config_factory(
        output_dir=tmp_path,
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=6,
        darcy_reference_iterations=80,
    )

    result = run_darcy_experiment(config)

    with np.load(tmp_path / result.artifacts["fields_npz"]) as fields:
        assert {
            "reference_pressure",
            "reference_velocity",
            "reference_u",
            "reference_v",
            "reference_speed",
            "reference_residual",
            "predicted_u",
            "predicted_v",
            "predicted_speed",
            "residual",
            "reference_metadata_json",
        }.issubset(fields.files)
        assert np.allclose(fields["reference_u"], fields["reference_velocity"][:, :1])
        assert np.allclose(fields["reference_v"], fields["reference_velocity"][:, 1:2])
        assert np.allclose(
            fields["reference_speed"],
            np.linalg.norm(fields["reference_velocity"], axis=1, keepdims=True),
        )
        assert np.all(np.isfinite(fields["reference_residual"]))
        metadata = json.loads(str(fields["reference_metadata_json"]))
        assert metadata["reference_kind"] == "dedicated-solver"
        assert metadata["reference_generator_name"] == "fenicsx_darcy_shared_domain"
        assert metadata["compared_model"] == "darcy"
