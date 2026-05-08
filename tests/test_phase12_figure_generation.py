"""Tests for Phase 12 figure generation from saved experiment artifacts."""

import json

import numpy as np

from pinn_fluid.figures import generate_figure_bundle


def _coordinates(grid_points: int) -> np.ndarray:
    values = np.linspace(0.0, 1.0, grid_points)
    grid_x, grid_y = np.meshgrid(values, values, indexing="ij")
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1)


def _write_history(path, *, boundary_name: str = "boundary") -> None:
    payload = {
        "total": [4.0, 2.0, 1.0],
        "components": {
            boundary_name: [2.0, 1.0, 0.5],
            "continuity": [1.0, 0.7, 0.4],
        },
    }
    path.write_text(json.dumps(payload))


def _assert_png(path) -> None:
    assert path.is_file()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_generate_figure_bundle_creates_darcy_field_and_convergence_panels(tmp_path):
    model_dir = tmp_path / "darcy"
    model_dir.mkdir()
    grid_points = 3
    coordinates = _coordinates(grid_points)
    predicted_pressure = np.linspace(0.0, 1.0, grid_points * grid_points).reshape(-1, 1)
    reference_pressure = np.flipud(predicted_pressure)
    predicted_velocity = np.column_stack(
        (predicted_pressure.reshape(-1), 1.0 - predicted_pressure.reshape(-1))
    )
    reference_velocity = predicted_velocity * 0.5
    residual = np.full((grid_points * grid_points, 1), 0.25)
    np.savez(
        model_dir / "fields.npz",
        coordinates=coordinates,
        predicted_pressure=predicted_pressure,
        reference_pressure=reference_pressure,
        predicted_velocity=predicted_velocity,
        reference_velocity=reference_velocity,
        residual=residual,
    )
    _write_history(model_dir / "history.json", boundary_name="wall")

    manifest = generate_figure_bundle(tmp_path)

    assert [entry["model"] for entry in manifest["models"]] == ["darcy"]
    darcy_entry = manifest["models"][0]
    assert darcy_entry["field_panel"] == "figures/darcy_fields.png"
    assert darcy_entry["convergence_panel"] == "figures/darcy_convergence.png"
    _assert_png(tmp_path / darcy_entry["field_panel"])
    _assert_png(tmp_path / darcy_entry["convergence_panel"])
    saved_manifest = json.loads((tmp_path / "figures" / "figure_manifest.json").read_text())
    assert saved_manifest == manifest


def test_generate_figure_bundle_creates_velocity_pressure_panels(tmp_path):
    model_dir = tmp_path / "stokes"
    model_dir.mkdir()
    grid_points = 3
    coordinates = _coordinates(grid_points)
    base = np.linspace(0.0, 1.0, grid_points * grid_points).reshape(-1, 1)
    np.savez(
        model_dir / "fields.npz",
        coordinates=coordinates,
        predicted_u=base,
        predicted_v=base * 0.25,
        predicted_pressure=base * 2.0,
        reference_u=base * 0.9,
        reference_v=base * 0.1,
        reference_pressure=base * 1.5,
        residual=np.full_like(base, 0.05),
    )
    _write_history(model_dir / "history.json")

    manifest = generate_figure_bundle(tmp_path)

    stokes_entry = manifest["models"][0]
    assert stokes_entry["model"] == "stokes"
    assert stokes_entry["field_panel"] == "figures/stokes_fields.png"
    assert stokes_entry["convergence_panel"] == "figures/stokes_convergence.png"
    assert stokes_entry["field_panel_title"] == "stokes field comparison"
    assert stokes_entry["convergence_panel_title"] == "stokes convergence"
    assert stokes_entry["convergence_legend"] == [
        "total",
        "boundary",
        "continuity",
    ]
    _assert_png(tmp_path / stokes_entry["field_panel"])
    _assert_png(tmp_path / stokes_entry["convergence_panel"])


def test_generate_figure_bundle_creates_separate_predicted_actual_and_residual_images(tmp_path):
    model_dir = tmp_path / "navier_stokes"
    model_dir.mkdir()
    grid_points = 3
    coordinates = _coordinates(grid_points)
    base = np.linspace(0.0, 1.0, grid_points * grid_points).reshape(-1, 1)
    np.savez(
        model_dir / "fields.npz",
        coordinates=coordinates,
        predicted_u=base,
        predicted_v=base * 0.25,
        predicted_pressure=base * 2.0,
        reference_u=base * 0.9,
        reference_v=base * 0.1,
        reference_pressure=base * 1.5,
        residual=np.full_like(base, 0.05),
    )
    _write_history(model_dir / "history.json")

    manifest = generate_figure_bundle(tmp_path)

    entry = manifest["models"][0]
    separate = entry["separate_field_images"]
    expected_keys = {
        "predicted_u",
        "actual_u",
        "residual_u",
        "predicted_v",
        "actual_v",
        "residual_v",
        "predicted_p",
        "actual_p",
        "residual_p",
    }
    assert set(separate) == expected_keys
    for key in expected_keys:
        assert separate[key]["title"]
        assert separate[key]["legend"] == "scalar value"
        _assert_png(tmp_path / separate[key]["path"])
