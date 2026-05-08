"""Figure generation utilities for saved PINN experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ("darcy", "stokes", "oseen", "navier_stokes")
FIELD_COLORMAP = "turbo"
SCALAR_LEGEND = "scalar value"
CONVERGENCE_AXES = {"x": "iteration", "y": "objective", "yscale": "log"}
CONVERGENCE_LEGEND_TITLE = "loss components"
COMPARISON_COLUMNS = ("predicted", "actual", "residual")
BOUNDARY_MARKER_COLORS = {
    "inlet": (220, 0, 0),
    "outlet": (0, 90, 255),
}
BOUNDARY_MARKER_OFFSET = 0.035
BOUNDARY_MARKERS = {
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


def _load_history(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not payload.get("total"):
        raise ValueError(f"{path} must contain a non-empty total history")
    components = payload.get("components", {})
    if not isinstance(components, dict):
        raise ValueError(f"{path} components must be a JSON object")
    return payload


def _grid_points(fields: np.lib.npyio.NpzFile) -> int:
    coordinates = np.asarray(fields["coordinates"])
    points = int(round(np.sqrt(coordinates.shape[0])))
    if points * points != coordinates.shape[0]:
        raise ValueError("coordinates must describe a square grid")
    return points


def _scalar(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _magnitude(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("velocity fields must have shape (points, 2)")
    return np.sqrt(np.sum(array**2, axis=1))


def _reshape(values: np.ndarray, grid_points: int) -> np.ndarray:
    return _scalar(values).reshape(grid_points, grid_points).T


def _format_tick(value: float) -> str:
    return f"{value:.6g}"


def _value_limits(values: np.ndarray) -> list[float]:
    flattened = _scalar(values)
    return [float(np.min(flattened)), float(np.max(flattened))]


def _combined_limits(*values: np.ndarray) -> list[float]:
    flattened = np.concatenate([_scalar(value) for value in values])
    return [float(np.min(flattened)), float(np.max(flattened))]


def _colorbar_ticks(values: np.ndarray, limits: list[float] | None = None) -> list[float]:
    low, high = _value_limits(values) if limits is None else [float(limits[0]), float(limits[1])]
    middle = 0.5 * (low + high)
    return [low, middle, high]


def _colorbar_tick_labels(values: np.ndarray, limits: list[float] | None = None) -> list[str]:
    return [_format_tick(value) for value in _colorbar_ticks(values, limits)]


def _draw_boundary_markers(axis, *, include_labels: bool = False, linewidth: float = 3.0) -> None:
    inlet_label = BOUNDARY_MARKERS["inlet"]["label"] if include_labels else None
    outlet_label = BOUNDARY_MARKERS["outlet"]["label"] if include_labels else None
    axis.plot(
        BOUNDARY_MARKERS["inlet"]["x_range"],
        [1.0 + BOUNDARY_MARKER_OFFSET, 1.0 + BOUNDARY_MARKER_OFFSET],
        color="red",
        linewidth=linewidth,
        solid_capstyle="butt",
        clip_on=False,
        label=inlet_label,
    )
    axis.plot(
        BOUNDARY_MARKERS["outlet"]["x_range"],
        [-BOUNDARY_MARKER_OFFSET, -BOUNDARY_MARKER_OFFSET],
        color="blue",
        linewidth=linewidth,
        solid_capstyle="butt",
        clip_on=False,
        label=outlet_label,
    )


def _save_field_panel(
    panels: list[tuple[str, np.ndarray]],
    *,
    grid_points: int,
    columns: int,
    title: str,
    path: Path,
    column_labels: tuple[str, ...] = (),
    row_labels: tuple[str, ...] = (),
    color_limits: list[list[float] | None] | None = None,
) -> None:
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns + (0.6 if row_labels else 0.0), 2.8 * rows),
    )
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    flat_axes = axes_array.reshape(-1)
    for index, (axis, (panel_title, values)) in enumerate(zip(flat_axes, panels)):
        limits = None if color_limits is None else color_limits[index]
        image = axis.imshow(
            _reshape(values, grid_points),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=FIELD_COLORMAP,
            vmin=None if limits is None else limits[0],
            vmax=None if limits is None else limits[1],
        )
        _draw_boundary_markers(axis)
        if not column_labels:
            axis.set_title(panel_title, fontsize=9)
        else:
            axis.set_title("")
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        ticks = _colorbar_ticks(values, limits)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(_colorbar_tick_labels(values, limits))
        colorbar.set_label(SCALAR_LEGEND, fontsize=8, labelpad=10)
    for row, label in enumerate(row_labels):
        axes_array[row, 0].set_ylabel(
            label,
            rotation=0,
            labelpad=38,
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    for column, label in enumerate(column_labels):
        axes_array[-1, column].set_xlabel(label, fontsize=12, fontweight="bold", labelpad=10)
    for axis in flat_axes[len(panels) :]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_scalar_field_image(
    values: np.ndarray,
    *,
    grid_points: int,
    title: str,
    path: Path,
    limits: list[float] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(4.2, 3.4))
    image = axis.imshow(
        _reshape(values, grid_points),
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=FIELD_COLORMAP,
        vmin=None if limits is None else limits[0],
        vmax=None if limits is None else limits[1],
    )
    _draw_boundary_markers(axis, include_labels=True)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend(loc="upper right", fontsize="x-small")
    colorbar = fig.colorbar(image, ax=axis)
    ticks = _colorbar_ticks(values, limits)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(_colorbar_tick_labels(values, limits))
    colorbar.set_label(SCALAR_LEGEND)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_convergence_panel(history: dict[str, object], *, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history["total"], label="total", linewidth=2)
    components = history.get("components", {})
    assert isinstance(components, dict)
    for name in sorted(components):
        axis.plot(components[name], label=name, linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("iteration")
    axis.set_ylabel("objective")
    axis.set_yscale("log")
    axis.legend(fontsize="x-small", title=CONVERGENCE_LEGEND_TITLE)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _history_legend(history: dict[str, object]) -> list[str]:
    components = history.get("components", {})
    assert isinstance(components, dict)
    return ["total"] + sorted(components)


def _darcy_panels(fields: np.lib.npyio.NpzFile) -> list[tuple[str, np.ndarray]]:
    predicted_pressure = _scalar(fields["predicted_pressure"])
    reference_pressure = _scalar(fields["reference_pressure"])
    predicted_speed = _magnitude(fields["predicted_velocity"])
    reference_speed = _magnitude(fields["reference_velocity"])
    return [
        ("pressure predicted", predicted_pressure),
        ("pressure actual", reference_pressure),
        ("pressure residual", np.abs(predicted_pressure - reference_pressure)),
        ("speed predicted", predicted_speed),
        ("speed actual", reference_speed),
        ("speed residual", np.abs(predicted_speed - reference_speed)),
    ]


def _velocity_pressure_panels(fields: np.lib.npyio.NpzFile) -> list[tuple[str, np.ndarray]]:
    predicted_u = _scalar(fields["predicted_u"])
    predicted_v = _scalar(fields["predicted_v"])
    predicted_p = _scalar(fields["predicted_pressure"])
    reference_u = _scalar(fields["reference_u"])
    reference_v = _scalar(fields["reference_v"])
    reference_p = _scalar(fields["reference_pressure"])
    return [
        ("u predicted", predicted_u),
        ("u actual", reference_u),
        ("u residual", np.abs(predicted_u - reference_u)),
        ("v predicted", predicted_v),
        ("v actual", reference_v),
        ("v residual", np.abs(predicted_v - reference_v)),
        ("pressure predicted", predicted_p),
        ("pressure actual", reference_p),
        ("pressure residual", np.abs(predicted_p - reference_p)),
    ]


def _darcy_separate_images(fields: np.lib.npyio.NpzFile) -> dict[str, tuple[str, np.ndarray]]:
    predicted_pressure = _scalar(fields["predicted_pressure"])
    reference_pressure = _scalar(fields["reference_pressure"])
    return {
        "predicted_p": ("Darcy predicted pressure", predicted_pressure),
        "actual_p": ("Darcy actual pressure", reference_pressure),
        "residual_p": ("Darcy pressure residual", np.abs(predicted_pressure - reference_pressure)),
        "predicted_speed": ("Darcy predicted velocity magnitude", _magnitude(fields["predicted_velocity"])),
        "actual_speed": ("Darcy actual velocity magnitude", _magnitude(fields["reference_velocity"])),
        "residual_magnitude": ("Darcy equation residual magnitude", np.abs(_scalar(fields["residual"]))),
    }


def _velocity_pressure_separate_images(fields: np.lib.npyio.NpzFile) -> dict[str, tuple[str, np.ndarray]]:
    predicted_u = _scalar(fields["predicted_u"])
    predicted_v = _scalar(fields["predicted_v"])
    predicted_p = _scalar(fields["predicted_pressure"])
    actual_u = _scalar(fields["reference_u"])
    actual_v = _scalar(fields["reference_v"])
    actual_p = _scalar(fields["reference_pressure"])
    predicted_speed = np.sqrt(predicted_u**2 + predicted_v**2)
    actual_speed = np.sqrt(actual_u**2 + actual_v**2)
    return {
        "predicted_u": ("Predicted u flow field", predicted_u),
        "actual_u": ("Actual u flow field", actual_u),
        "residual_u": ("Residual u flow field", np.abs(predicted_u - actual_u)),
        "predicted_v": ("Predicted v flow field", predicted_v),
        "actual_v": ("Actual v flow field", actual_v),
        "residual_v": ("Residual v flow field", np.abs(predicted_v - actual_v)),
        "predicted_p": ("Predicted pressure field", predicted_p),
        "actual_p": ("Actual pressure field", actual_p),
        "residual_p": ("Residual pressure field", np.abs(predicted_p - actual_p)),
        "predicted_speed": ("Predicted velocity magnitude", predicted_speed),
        "actual_speed": ("Actual velocity magnitude", actual_speed),
        "residual_magnitude": ("Equation residual magnitude", np.abs(_scalar(fields["residual"]))),
    }


def _comparison_color_limits(
    panels: list[tuple[str, np.ndarray]],
    row_labels: tuple[str, ...],
) -> tuple[list[list[float] | None], dict[str, dict[str, list[float]]]]:
    limits: list[list[float] | None] = []
    metadata: dict[str, dict[str, list[float]]] = {}
    for row_index, label in enumerate(row_labels):
        offset = row_index * 3
        predicted = panels[offset][1]
        actual = panels[offset + 1][1]
        residual = panels[offset + 2][1]
        predicted_actual = _combined_limits(predicted, actual)
        residual_limits = _value_limits(residual)
        metadata[label] = {
            "predicted_actual": predicted_actual,
            "residual": residual_limits,
        }
        limits.extend([predicted_actual, predicted_actual, residual_limits])
    return limits, metadata


def _separate_image_color_limits(
    images: dict[str, tuple[str, np.ndarray]],
) -> dict[str, list[float]]:
    limits = {key: _value_limits(values) for key, (_, values) in images.items()}
    for predicted_key, actual_key in (
        ("predicted_u", "actual_u"),
        ("predicted_v", "actual_v"),
        ("predicted_p", "actual_p"),
        ("predicted_speed", "actual_speed"),
    ):
        if predicted_key in images and actual_key in images:
            shared = _combined_limits(images[predicted_key][1], images[actual_key][1])
            limits[predicted_key] = shared
            limits[actual_key] = shared
    return limits


def _model_field_panels(model: str, fields: np.lib.npyio.NpzFile) -> tuple[list[tuple[str, np.ndarray]], int]:
    if model == "darcy":
        return _darcy_panels(fields), 3
    return _velocity_pressure_panels(fields), 3


def _model_field_layout(model: str) -> dict[str, list[str]]:
    if model == "darcy":
        return {"columns": list(COMPARISON_COLUMNS), "rows": ["pressure", "speed"]}
    return {"columns": list(COMPARISON_COLUMNS), "rows": ["u", "v", "pressure"]}


def _model_separate_images(model: str, fields: np.lib.npyio.NpzFile) -> dict[str, tuple[str, np.ndarray]]:
    if model == "darcy":
        return _darcy_separate_images(fields)
    return _velocity_pressure_separate_images(fields)


def _write_separate_images(
    *,
    model: str,
    fields: np.lib.npyio.NpzFile,
    grid_points: int,
    root: Path,
    target: Path,
) -> dict[str, dict[str, str]]:
    images: dict[str, dict[str, str]] = {}
    image_values = _model_separate_images(model, fields)
    limits_by_key = _separate_image_color_limits(image_values)
    for key, (title, values) in image_values.items():
        path = target / f"{model}_{key}.png"
        limits = limits_by_key[key]
        _save_scalar_field_image(
            values,
            grid_points=grid_points,
            title=title,
            path=path,
            limits=limits,
        )
        images[key] = {
            "path": path.relative_to(root).as_posix(),
            "title": title,
            "legend": SCALAR_LEGEND,
            "colormap": FIELD_COLORMAP,
            "color_limits": limits,
            "colorbar_ticks": _colorbar_ticks(values, limits),
            "colorbar_tick_labels": _colorbar_tick_labels(values, limits),
        }
    return images


def _quiver_fields(
    model: str,
    fields: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model == "darcy":
        return (
            _scalar(fields["predicted_pressure"]),
            np.asarray(fields["predicted_velocity"], dtype=np.float64),
            _scalar(fields["reference_pressure"]),
            np.asarray(fields["reference_velocity"], dtype=np.float64),
        )
    predicted_velocity = np.column_stack((_scalar(fields["predicted_u"]), _scalar(fields["predicted_v"])))
    reference_velocity = np.column_stack((_scalar(fields["reference_u"]), _scalar(fields["reference_v"])))
    return (
        _scalar(fields["predicted_pressure"]),
        predicted_velocity,
        _scalar(fields["reference_pressure"]),
        reference_velocity,
    )


def _save_pressure_velocity_quiver(
    *,
    model: str,
    fields: np.lib.npyio.NpzFile,
    grid_points: int,
    root: Path,
    target: Path,
) -> str:
    predicted_pressure, predicted_velocity, actual_pressure, actual_velocity = _quiver_fields(model, fields)
    pressure_limits = _combined_limits(predicted_pressure, actual_pressure)
    path = target / f"{model}_pressure_velocity_quiver.png"
    x_values = np.linspace(0.0, 1.0, grid_points)
    y_values = np.linspace(0.0, 1.0, grid_points)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    step = max(1, grid_points // 12)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True, sharey=True)
    for axis, label, pressure, velocity in (
        (axes[0], "predicted", predicted_pressure, predicted_velocity),
        (axes[1], "actual", actual_pressure, actual_velocity),
    ):
        pressure_field = _reshape(pressure, grid_points)
        u_field = _reshape(velocity[:, 0], grid_points)
        v_field = _reshape(velocity[:, 1], grid_points)
        image = axis.contourf(
            grid_x,
            grid_y,
            pressure_field,
            levels=20,
            cmap=FIELD_COLORMAP,
            vmin=pressure_limits[0],
            vmax=pressure_limits[1],
        )
        axis.quiver(
            grid_x[::step, ::step],
            grid_y[::step, ::step],
            u_field[::step, ::step],
            v_field[::step, ::step],
            color="black",
            scale=20,
            width=0.004,
        )
        _draw_boundary_markers(axis)
        axis.set_title(f"{label} pressure + velocity")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    colorbar.set_ticks(_colorbar_ticks(predicted_pressure, pressure_limits))
    colorbar.set_ticklabels(_colorbar_tick_labels(predicted_pressure, pressure_limits))
    colorbar.set_label("pressure")
    fig.suptitle(f"{model} pressure contours with velocity arrows")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.relative_to(root).as_posix()


def generate_figure_bundle(
    output_dir: Path | str,
    *,
    figures_dir: Path | str | None = None,
    models: Iterable[str] = MODEL_ORDER,
) -> dict[str, object]:
    """Generate field and convergence panels from saved experiment artifacts."""

    root = Path(output_dir)
    target = root / "figures" if figures_dir is None else Path(figures_dir)
    target.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for model in models:
        model_dir = root / model
        if not model_dir.exists():
            continue
        fields_path = model_dir / "fields.npz"
        history_path = model_dir / "history.json"
        if not fields_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"{model_dir} must contain fields.npz and history.json")

        with np.load(fields_path) as fields:
            grid_points = _grid_points(fields)
            panels, columns = _model_field_panels(model, fields)
            field_layout = _model_field_layout(model)
            panel_limits, panel_limit_metadata = _comparison_color_limits(
                panels,
                tuple(field_layout["rows"]),
            )
            field_panel = target / f"{model}_fields.png"
            field_panel_title = f"{model} field comparison"
            _save_field_panel(
                panels,
                grid_points=grid_points,
                columns=columns,
                title=field_panel_title,
                path=field_panel,
                column_labels=tuple(field_layout["columns"]),
                row_labels=tuple(field_layout["rows"]),
                color_limits=panel_limits,
            )
            separate_field_images = _write_separate_images(
                model=model,
                fields=fields,
                grid_points=grid_points,
                root=root,
                target=target,
            )
            pressure_velocity_quiver = _save_pressure_velocity_quiver(
                model=model,
                fields=fields,
                grid_points=grid_points,
                root=root,
                target=target,
            )

        convergence_panel = target / f"{model}_convergence.png"
        convergence_panel_title = f"{model} convergence"
        history = _load_history(history_path)
        _save_convergence_panel(
            history,
            title=convergence_panel_title,
            path=convergence_panel,
        )
        entries.append(
            {
                "model": model,
                "field_panel": field_panel.relative_to(root).as_posix(),
                "field_panel_title": field_panel_title,
                "field_panel_layout": field_layout,
                "field_panel_tiles": [label for label, _ in panels],
                "field_panel_color_limits": panel_limit_metadata,
                "boundary_markers": BOUNDARY_MARKERS,
                "pressure_velocity_quiver": pressure_velocity_quiver,
                "convergence_panel": convergence_panel.relative_to(root).as_posix(),
                "convergence_panel_title": convergence_panel_title,
                "convergence_legend": _history_legend(history),
                "convergence_axes": dict(CONVERGENCE_AXES),
                "convergence_legend_title": CONVERGENCE_LEGEND_TITLE,
                "separate_field_images": separate_field_images,
            }
        )

    manifest = {"output_dir": str(root), "figures_dir": str(target), "models": entries}
    (target / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for generating figures from an experiment output directory."""

    parser = argparse.ArgumentParser(
        description="Generate Phase 12 figure panels from saved experiment artifacts."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = generate_figure_bundle(args.output_dir, figures_dir=args.figures_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["FIELD_COLORMAP", "MODEL_ORDER", "generate_figure_bundle", "main"]
