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
COMPARISON_COLUMNS = ("predicted", "actual", "residual")
BOUNDARY_MARKER_COLORS = {
    "inlet": (220, 0, 0),
    "outlet": (0, 90, 255),
}
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


_FONT_5X7 = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "01010", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "01010", "00100", "00100", "00100", "01010", "10001"),
    "Y": ("10001", "01010", "00100", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11110", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
}


def _draw_rect(
    image: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    image[max(0, y0) : min(image.shape[0], y1), max(0, x0) : min(image.shape[1], x1)] = color


def _draw_text(
    image: np.ndarray,
    text: str,
    *,
    x: int,
    y: int,
    color: tuple[int, int, int] = (0, 0, 0),
    scale: int = 1,
    bold: bool = False,
) -> None:
    cursor = x
    for character in text.upper():
        if character == " ":
            cursor += 4 * scale
            continue
        glyph = _FONT_5X7.get(character)
        if glyph is None:
            cursor += 6 * scale
            continue
        for row, pattern in enumerate(glyph):
            for column, pixel in enumerate(pattern):
                if pixel == "1":
                    for offset in range(2 if bold else 1):
                        _draw_rect(
                            image,
                            x0=cursor + column * scale + offset,
                            y0=y + row * scale,
                            x1=cursor + (column + 1) * scale + offset,
                            y1=y + (row + 1) * scale,
                            color=color,
                        )
        cursor += 6 * scale


def _turbo_rgb(normalized: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            (48, 18, 59),
            (49, 110, 196),
            (55, 214, 206),
            (127, 246, 88),
            (249, 221, 50),
            (239, 80, 35),
            (122, 4, 3),
        ],
        dtype=np.float64,
    )
    scaled = np.clip(normalized, 0.0, 1.0) * (len(stops) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    weight = scaled - lower
    rgb = stops[lower] * (1.0 - weight[..., None]) + stops[upper] * weight[..., None]
    return rgb.astype(np.uint8)


def _marker_x_range(
    x_range: list[float],
    *,
    left: int,
    width: int,
) -> tuple[int, int]:
    x0 = left + round(x_range[0] * width)
    x1 = left + round(x_range[1] * width)
    return x0, max(x0 + 1, x1)


def _draw_boundary_markers(
    image: np.ndarray,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    label_offset: int = 0,
) -> None:
    thickness = max(4, round(min(width, height) * 0.04))
    inlet_x0, inlet_x1 = _marker_x_range(BOUNDARY_MARKERS["inlet"]["x_range"], left=left, width=width)
    outlet_x0, outlet_x1 = _marker_x_range(BOUNDARY_MARKERS["outlet"]["x_range"], left=left, width=width)
    _draw_rect(
        image,
        x0=inlet_x0,
        y0=top,
        x1=inlet_x1,
        y1=top + thickness,
        color=BOUNDARY_MARKER_COLORS["inlet"],
    )
    _draw_rect(
        image,
        x0=outlet_x0,
        y0=top + height - thickness,
        x1=outlet_x1,
        y1=top + height,
        color=BOUNDARY_MARKER_COLORS["outlet"],
    )
    if label_offset:
        _draw_text(
            image,
            BOUNDARY_MARKERS["inlet"]["label"],
            x=inlet_x0,
            y=max(0, top - label_offset),
            color=BOUNDARY_MARKER_COLORS["inlet"],
            scale=1,
            bold=True,
        )
        _draw_text(
            image,
            BOUNDARY_MARKERS["outlet"]["label"],
            x=max(left, outlet_x1 - 36),
            y=min(image.shape[0] - 8, top + height + 4),
            color=BOUNDARY_MARKER_COLORS["outlet"],
            scale=1,
            bold=True,
        )


def _fallback_scalar_image(
    values: np.ndarray,
    grid_points: int,
    scale: int = 20,
    limits: list[float] | None = None,
) -> np.ndarray:
    field = _reshape(values, grid_points)
    low, high = _value_limits(field) if limits is None else [float(limits[0]), float(limits[1])]
    span = high - low if high > low else 1.0
    normalized = (field - low) / span
    image = _turbo_rgb(normalized)
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _fallback_scalar_canvas(
    values: np.ndarray,
    *,
    grid_points: int,
    title: str,
    scale: int,
    limits: list[float] | None = None,
    show_title: bool = True,
) -> np.ndarray:
    field = _fallback_scalar_image(values, grid_points, scale=scale, limits=limits)
    field_height, field_width, _ = field.shape
    top = 30 if show_title else 8
    left = 36
    right = 112
    bottom = 52
    image = np.full(
        (top + field_height + bottom, left + field_width + right, 3),
        255,
        dtype=np.uint8,
    )
    image[top : top + field_height, left : left + field_width] = field
    _draw_boundary_markers(
        image,
        left=left,
        top=top,
        width=field_width,
        height=field_height,
        label_offset=10 if show_title else 0,
    )
    if show_title:
        _draw_text(image, title, x=8, y=8, scale=1)
    _draw_line(image, (left, top + field_height), (left + field_width, top + field_height), (0, 0, 0))
    _draw_line(image, (left, top), (left, top + field_height), (0, 0, 0))
    _draw_text(image, "x", x=left + field_width // 2, y=top + field_height + 12, scale=1)
    _draw_text(image, "y", x=12, y=top + field_height // 2, scale=1)

    bar_x = left + field_width + 18
    bar_y = top
    bar_width = 16
    for offset in range(field_height):
        value = 1.0 - offset / max(1, field_height - 1)
        color = tuple(int(channel) for channel in _turbo_rgb(np.array(value)))
        _draw_rect(
            image,
            x0=bar_x,
            y0=bar_y + offset,
            x1=bar_x + bar_width,
            y1=bar_y + offset + 1,
            color=color,
        )
    labels = _colorbar_tick_labels(values, limits)
    for label, y in (
        (labels[2], bar_y),
        (labels[1], bar_y + field_height // 2 - 3),
        (labels[0], bar_y + field_height - 7),
    ):
        _draw_text(image, label, x=bar_x + bar_width + 5, y=y, scale=1)
    _draw_text(
        image,
        SCALAR_LEGEND,
        x=bar_x,
        y=bar_y + field_height + 12,
        scale=1,
    )
    return image


def _fallback_field_panel(
    panels: list[tuple[str, np.ndarray]],
    *,
    grid_points: int,
    columns: int,
    path: Path,
    column_labels: tuple[str, ...] = (),
    row_labels: tuple[str, ...] = (),
    color_limits: list[list[float] | None] | None = None,
) -> None:
    has_comparison_labels = bool(column_labels or row_labels)
    tiles = [
        _fallback_scalar_canvas(
            values,
            grid_points=grid_points,
            title=title,
            scale=20,
            limits=None if color_limits is None else color_limits[index],
            show_title=not has_comparison_labels,
        )
        for index, (title, values) in enumerate(panels)
    ]
    tile_height, tile_width, _ = tiles[0].shape
    rows = int(np.ceil(len(tiles) / columns))
    footer_height = 38 if column_labels else 0
    label_width = 116 if row_labels else 0
    image = np.full(
        (rows * tile_height + footer_height, label_width + columns * tile_width, 3),
        255,
        dtype=np.uint8,
    )
    for row, label in enumerate(row_labels):
        _draw_text(
            image,
            label,
            x=8,
            y=row * tile_height + tile_height // 2 - 8,
            scale=2,
            bold=True,
        )
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        image[
            row * tile_height : (row + 1) * tile_height,
            label_width + column * tile_width : label_width + (column + 1) * tile_width,
        ] = tile
    for column, label in enumerate(column_labels):
        _draw_text(
            image,
            label,
            x=label_width + column * tile_width + 8,
            y=rows * tile_height + 10,
            scale=2,
            bold=True,
        )
    _write_rgb_png(path, image)


def _fallback_convergence_panel(
    history: dict[str, object],
    *,
    title: str,
    path: Path,
) -> None:
    image = np.full((260, 420, 3), 255, dtype=np.uint8)
    components = history.get("components", {})
    assert isinstance(components, dict)
    labels = ["total"] + sorted(components)
    series = [history["total"]] + [components[name] for name in sorted(components)]
    values = np.array([value for item in series for value in item], dtype=np.float64)
    values = np.log10(np.clip(values, 1e-12, None))
    low = float(values.min())
    high = float(values.max())
    span = high - low if high > low else 1.0
    colors = [(0, 0, 0), (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    _draw_text(image, title, x=120, y=8, scale=1)
    _draw_line(image, (40, 220), (380, 220), (0, 0, 0))
    _draw_line(image, (40, 40), (40, 220), (0, 0, 0))
    _draw_text(image, CONVERGENCE_AXES["x"], x=168, y=238, scale=1)
    _draw_text(image, CONVERGENCE_AXES["y"], x=4, y=118, scale=1)
    _draw_text(image, CONVERGENCE_LEGEND_TITLE, x=306, y=46, scale=1)
    for index, item in enumerate(series):
        log_values = np.log10(np.clip(np.array(item, dtype=np.float64), 1e-12, None))
        points = []
        for step, value in enumerate(log_values):
            x = 40 + round(step * 340 / max(1, len(log_values) - 1))
            y = 220 - round((float(value) - low) * 180 / span)
            points.append((x, y))
        for start, end in zip(points, points[1:]):
            _draw_line(image, start, end, colors[index % len(colors)])
        legend_y = 62 + index * 14
        color = colors[index % len(colors)]
        _draw_line(image, (306, legend_y + 4), (326, legend_y + 4), color)
        _draw_text(image, labels[index], x=332, y=legend_y, color=color, scale=1)
    _write_rgb_png(path, image)


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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_field_panel(
            panels,
            grid_points=grid_points,
            columns=columns,
            path=path,
            column_labels=column_labels,
            row_labels=row_labels,
            color_limits=color_limits,
        )
        return

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
        axis.plot(
            BOUNDARY_MARKERS["inlet"]["x_range"],
            [1.0, 1.0],
            color="red",
            linewidth=3,
            solid_capstyle="butt",
        )
        axis.plot(
            BOUNDARY_MARKERS["outlet"]["x_range"],
            [0.0, 0.0],
            color="blue",
            linewidth=3,
            solid_capstyle="butt",
        )
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _write_rgb_png(
            path,
            _fallback_scalar_canvas(values, grid_points=grid_points, title=title, scale=28, limits=limits),
        )
        return

    fig, axis = plt.subplots(figsize=(4.2, 3.4))
    image = axis.imshow(
        _reshape(values, grid_points),
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=FIELD_COLORMAP,
        vmin=None if limits is None else limits[0],
        vmax=None if limits is None else limits[1],
    )
    axis.plot(
        BOUNDARY_MARKERS["inlet"]["x_range"],
        [1.0, 1.0],
        color="red",
        linewidth=3,
        solid_capstyle="butt",
        label=BOUNDARY_MARKERS["inlet"]["label"],
    )
    axis.plot(
        BOUNDARY_MARKERS["outlet"]["x_range"],
        [0.0, 0.0],
        color="blue",
        linewidth=3,
        solid_capstyle="butt",
        label=BOUNDARY_MARKERS["outlet"]["label"],
    )
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_convergence_panel(history, title=title, path=path)
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


def _draw_fallback_arrows(
    image: np.ndarray,
    velocity: np.ndarray,
    *,
    left: int,
    top: int,
    grid_points: int,
    scale: int,
) -> None:
    step = max(1, grid_points // 6)
    field_size = grid_points * scale
    max_speed = float(np.max(np.sqrt(np.sum(velocity**2, axis=1))))
    if max_speed <= 0.0:
        max_speed = 1.0
    for x_index in range(0, grid_points, step):
        for y_index in range(0, grid_points, step):
            flat_index = x_index * grid_points + y_index
            u, v = velocity[flat_index]
            start_x = left + round(x_index * scale + scale / 2)
            start_y = top + field_size - round(y_index * scale + scale / 2)
            end_x = start_x + round(float(u) / max_speed * scale * 0.8)
            end_y = start_y - round(float(v) / max_speed * scale * 0.8)
            _draw_line(image, (start_x, start_y), (end_x, end_y), (0, 0, 0))


def _fallback_pressure_velocity_quiver(
    *,
    pressure: np.ndarray,
    velocity: np.ndarray,
    grid_points: int,
    title: str,
    path: Path,
    limits: list[float],
) -> None:
    image = _fallback_scalar_canvas(
        pressure,
        grid_points=grid_points,
        title=title,
        scale=24,
        limits=limits,
    )
    _draw_fallback_arrows(image, velocity, left=36, top=30, grid_points=grid_points, scale=24)
    _write_rgb_png(path, image)


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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_pressure_velocity_quiver(
            pressure=predicted_pressure,
            velocity=predicted_velocity,
            grid_points=grid_points,
            title=f"{model} predicted pressure and velocity",
            path=path,
            limits=pressure_limits,
        )
        return path.relative_to(root).as_posix()

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
        axis.plot(BOUNDARY_MARKERS["inlet"]["x_range"], [1.0, 1.0], color="red", linewidth=3)
        axis.plot(BOUNDARY_MARKERS["outlet"]["x_range"], [0.0, 0.0], color="blue", linewidth=3)
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
