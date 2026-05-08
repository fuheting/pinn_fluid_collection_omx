"""Tests for Phase 12 figure generation from saved experiment artifacts."""

import json

import matplotlib.pyplot as plt
import numpy as np

import pinn_fluid.figures as figures
from pinn_fluid.figures import (
    _save_scalar_field_image,
    generate_figure_bundle,
)


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


def _read_rgb_png(path):
    image = plt.imread(path)
    if image.dtype != np.uint8:
        image = (image[:, :, :3] * 255).astype(np.uint8)
    return image[:, :, :3]


def _has_non_white_pixels(image, *, y_slice, x_slice) -> bool:
    region = image[y_slice, x_slice]
    return bool(np.any(region != 255))


def _has_dark_pixels(image, *, y_slice, x_slice) -> bool:
    region = image[y_slice, x_slice]
    return bool(np.any(np.all(region < 80, axis=2)))


def _has_color_pixels(image, *, y_slice, x_slice, color) -> bool:
    region = image[y_slice, x_slice]
    target = np.array(color, dtype=np.int16)
    delta = np.abs(region.astype(np.int16) - target)
    return bool(np.any(np.all(delta <= 20, axis=2)))


def test_figure_generation_uses_matplotlib_without_fallback_renderer_symbols():
    assert not hasattr(figures, "_fallback_scalar_image")
    assert not hasattr(figures, "_fallback_field_panel")
    assert not hasattr(figures, "_fallback_convergence_panel")
    assert not hasattr(figures, "_fallback_pressure_velocity_quiver")


def test_scalar_field_image_draws_visible_title_axes_and_colorbar(tmp_path):
    path = tmp_path / "scalar.png"
    values = np.linspace(0.0, 1.0, 9)

    _save_scalar_field_image(values, grid_points=3, title="Predicted u flow field", path=path)

    image = _read_rgb_png(path)
    assert _has_non_white_pixels(image, y_slice=slice(20, 70), x_slice=slice(80, 300))
    assert _has_non_white_pixels(image, y_slice=slice(300, 475), x_slice=slice(40, 120))
    assert _has_non_white_pixels(image, y_slice=slice(80, 410), x_slice=slice(480, 570))


def test_scalar_field_image_highlights_shared_inlet_and_outlet(tmp_path):
    path = tmp_path / "scalar_boundaries.png"
    values = np.linspace(0.0, 1.0, 16)

    _save_scalar_field_image(values, grid_points=4, title="Predicted u flow field", path=path)

    image = _read_rgb_png(path)
    assert _has_color_pixels(image, y_slice=slice(45, 160), x_slice=slice(60, 230), color=(255, 0, 0))
    assert _has_color_pixels(image, y_slice=slice(310, 470), x_slice=slice(295, 470), color=(0, 0, 255))


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
    assert darcy_entry["field_panel_layout"] == {
        "columns": ["predicted", "actual", "residual"],
        "rows": ["pressure", "speed"],
    }
    assert darcy_entry["boundary_markers"] == {
        "inlet": {
            "label": "inlet",
            "location": "top",
            "x_range": [0.0, 0.25],
            "color": "red",
        },
        "outlet": {
            "label": "outlet",
            "location": "bottom",
            "x_range": [0.75, 1.0],
            "color": "blue",
        },
    }
    assert darcy_entry["field_panel_tiles"] == [
        "pressure predicted",
        "pressure actual",
        "pressure residual",
        "speed predicted",
        "speed actual",
        "speed residual",
    ]
    assert darcy_entry["field_panel_color_limits"]["pressure"]["predicted_actual"] == [0.0, 1.0]
    assert darcy_entry["pressure_velocity_quiver"] == "figures/darcy_pressure_velocity_quiver.png"
    _assert_png(tmp_path / darcy_entry["field_panel"])
    _assert_png(tmp_path / darcy_entry["convergence_panel"])
    _assert_png(tmp_path / darcy_entry["pressure_velocity_quiver"])
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
    assert stokes_entry["field_panel_layout"] == {
        "columns": ["predicted", "actual", "residual"],
        "rows": ["u", "v", "pressure"],
    }
    assert stokes_entry["field_panel_tiles"] == [
        "u predicted",
        "u actual",
        "u residual",
        "v predicted",
        "v actual",
        "v residual",
        "pressure predicted",
        "pressure actual",
        "pressure residual",
    ]
    assert stokes_entry["field_panel_color_limits"]["pressure"]["predicted_actual"] == [0.0, 2.0]
    assert stokes_entry["pressure_velocity_quiver"] == "figures/stokes_pressure_velocity_quiver.png"
    assert stokes_entry["convergence_panel_title"] == "stokes convergence"
    assert stokes_entry["convergence_legend"] == [
        "total",
        "boundary",
        "continuity",
    ]
    assert stokes_entry["convergence_axes"] == {
        "x": "iteration",
        "y": "objective",
        "yscale": "log",
    }
    assert stokes_entry["convergence_legend_title"] == "loss components"
    _assert_png(tmp_path / stokes_entry["field_panel"])
    _assert_png(tmp_path / stokes_entry["convergence_panel"])
    _assert_png(tmp_path / stokes_entry["pressure_velocity_quiver"])


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
        "predicted_speed",
        "actual_speed",
        "residual_magnitude",
    }
    assert set(separate) == expected_keys
    for key in expected_keys:
        assert separate[key]["title"]
        assert separate[key]["legend"] == "scalar value"
        assert separate[key]["colormap"] == "turbo"
        assert len(separate[key]["colorbar_ticks"]) == 3
        assert len(separate[key]["colorbar_tick_labels"]) == 3
        _assert_png(tmp_path / separate[key]["path"])
    assert separate["predicted_u"]["color_limits"] == [0.0, 1.0]
    assert separate["actual_u"]["color_limits"] == [0.0, 1.0]
    assert separate["predicted_p"]["color_limits"] == [0.0, 2.0]
    assert separate["actual_p"]["color_limits"] == [0.0, 2.0]
