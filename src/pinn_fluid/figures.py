"""Figure generation utilities for saved PINN experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from typing import Iterable
import zlib

import numpy as np

MODEL_ORDER = ("darcy", "stokes", "oseen", "navier_stokes")
FIELD_COLORMAP = "turbo"
SCALAR_LEGEND = "scalar value"
CONVERGENCE_AXES = {"x": "iteration", "y": "objective", "yscale": "log"}
CONVERGENCE_LEGEND_TITLE = "loss components"


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


def _colorbar_ticks(values: np.ndarray) -> list[float]:
    flattened = _scalar(values)
    low = float(np.min(flattened))
    high = float(np.max(flattened))
    middle = 0.5 * (low + high)
    return [low, middle, high]


def _colorbar_tick_labels(values: np.ndarray) -> list[str]:
    return [_format_tick(value) for value in _colorbar_ticks(values)]


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape
    rows = [b"\x00" + image[row].astype(np.uint8).tobytes() for row in range(height)]
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _draw_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _fallback_scalar_image(values: np.ndarray, grid_points: int, scale: int = 20) -> np.ndarray:
    field = _reshape(values, grid_points)
    low = float(np.min(field))
    high = float(np.max(field))
    span = high - low if high > low else 1.0
    normalized = ((field - low) / span * 255.0).astype(np.uint8)
    image = np.stack((normalized, 255 - normalized, np.full_like(normalized, 128)), axis=2)
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _fallback_field_panel(
    panels: list[tuple[str, np.ndarray]],
    *,
    grid_points: int,
    columns: int,
    path: Path,
) -> None:
    tiles = [_fallback_scalar_image(values, grid_points) for _, values in panels]
    tile_height, tile_width, _ = tiles[0].shape
    rows = int(np.ceil(len(tiles) / columns))
    image = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        image[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    _write_rgb_png(path, image)


def _fallback_convergence_panel(history: dict[str, object], *, path: Path) -> None:
    image = np.full((260, 420, 3), 255, dtype=np.uint8)
    components = history.get("components", {})
    assert isinstance(components, dict)
    series = [history["total"]] + [components[name] for name in sorted(components)]
    values = np.array([value for item in series for value in item], dtype=np.float64)
    values = np.log10(np.clip(values, 1e-12, None))
    low = float(values.min())
    high = float(values.max())
    span = high - low if high > low else 1.0
    colors = [(0, 0, 0), (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    for index, item in enumerate(series):
        log_values = np.log10(np.clip(np.array(item, dtype=np.float64), 1e-12, None))
        points = []
        for step, value in enumerate(log_values):
            x = 40 + round(step * 340 / max(1, len(log_values) - 1))
            y = 220 - round((float(value) - low) * 180 / span)
            points.append((x, y))
        for start, end in zip(points, points[1:]):
            _draw_line(image, start, end, colors[index % len(colors)])
    _write_rgb_png(path, image)


def _save_field_panel(
    panels: list[tuple[str, np.ndarray]],
    *,
    grid_points: int,
    columns: int,
    title: str,
    path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_field_panel(panels, grid_points=grid_points, columns=columns, path=path)
        return

    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.8 * rows))
    flat_axes = np.atleast_1d(axes).reshape(-1)
    for axis, (panel_title, values) in zip(flat_axes, panels):
        image = axis.imshow(
            _reshape(values, grid_points),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=FIELD_COLORMAP,
        )
        axis.set_title(panel_title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        ticks = _colorbar_ticks(values)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(_colorbar_tick_labels(values))
        colorbar.set_label(SCALAR_LEGEND, fontsize=8)
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
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _write_rgb_png(path, _fallback_scalar_image(values, grid_points, scale=28))
        return

    fig, axis = plt.subplots(figsize=(4.2, 3.4))
    image = axis.imshow(
        _reshape(values, grid_points),
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=FIELD_COLORMAP,
    )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    colorbar = fig.colorbar(image, ax=axis)
    ticks = _colorbar_ticks(values)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(_colorbar_tick_labels(values))
    colorbar.set_label(SCALAR_LEGEND)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_convergence_panel(history: dict[str, object], *, title: str, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_convergence_panel(history, path=path)
        return

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
    return [
        ("predicted pressure", predicted_pressure),
        ("reference pressure", reference_pressure),
        ("pressure error", np.abs(predicted_pressure - reference_pressure)),
        ("predicted speed", _magnitude(fields["predicted_velocity"])),
        ("reference speed", _magnitude(fields["reference_velocity"])),
        ("residual magnitude", np.abs(_scalar(fields["residual"]))),
    ]


def _velocity_pressure_panels(fields: np.lib.npyio.NpzFile) -> list[tuple[str, np.ndarray]]:
    predicted_u = _scalar(fields["predicted_u"])
    predicted_v = _scalar(fields["predicted_v"])
    reference_u = _scalar(fields["reference_u"])
    reference_v = _scalar(fields["reference_v"])
    return [
        ("predicted u", predicted_u),
        ("predicted v", predicted_v),
        ("predicted speed", np.sqrt(predicted_u**2 + predicted_v**2)),
        ("predicted pressure", _scalar(fields["predicted_pressure"])),
        ("reference u", reference_u),
        ("reference v", reference_v),
        ("reference speed", np.sqrt(reference_u**2 + reference_v**2)),
        ("reference pressure", _scalar(fields["reference_pressure"])),
        ("residual magnitude", np.abs(_scalar(fields["residual"]))),
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
    }


def _model_field_panels(model: str, fields: np.lib.npyio.NpzFile) -> tuple[list[tuple[str, np.ndarray]], int]:
    if model == "darcy":
        return _darcy_panels(fields), 3
    return _velocity_pressure_panels(fields), 3


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
    for key, (title, values) in _model_separate_images(model, fields).items():
        path = target / f"{model}_{key}.png"
        _save_scalar_field_image(values, grid_points=grid_points, title=title, path=path)
        images[key] = {
            "path": path.relative_to(root).as_posix(),
            "title": title,
            "legend": SCALAR_LEGEND,
            "colormap": FIELD_COLORMAP,
            "colorbar_ticks": _colorbar_ticks(values),
            "colorbar_tick_labels": _colorbar_tick_labels(values),
        }
    return images


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
            field_panel = target / f"{model}_fields.png"
            field_panel_title = f"{model} field comparison"
            _save_field_panel(
                panels,
                grid_points=grid_points,
                columns=columns,
                title=field_panel_title,
                path=field_panel,
            )
            separate_field_images = _write_separate_images(
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
