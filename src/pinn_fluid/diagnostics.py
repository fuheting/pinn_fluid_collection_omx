"""Boundary and collocation diagnostics for the shared unit-square flow domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.figures import (
    BOUNDARY_MARKER_COLORS,
    _draw_boundary_markers,
    _draw_rect,
    _draw_text,
    _write_rgb_png,
)

COORDINATE_CONVENTION = "cartesian_unit_square_y_up"
COORDINATE_DESCRIPTION = "x increases left-to-right; y=0 is bottom; y=1 is top"


def _as_numpy(values: object) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _coordinate_summary(points: np.ndarray) -> dict[str, float | int]:
    if points.size == 0:
        raise ValueError("diagnostic point groups must be non-empty")
    return {
        "count": int(points.shape[0]),
        "x_min": float(np.min(points[:, 0])),
        "x_max": float(np.max(points[:, 0])),
        "y_min": float(np.min(points[:, 1])),
        "y_max": float(np.max(points[:, 1])),
    }


def boundary_diagnostic_report(
    *,
    interior_points_per_axis: int = 9,
    boundary_points_per_patch: int = 9,
) -> dict[str, object]:
    """Return coordinate ranges for inlet, outlet, wall, and interior samples."""

    interior = _as_numpy(interior_collocation_points(interior_points_per_axis))
    boundary = boundary_collocation_points(boundary_points_per_patch)
    inlet = _as_numpy(boundary["inlet"]["coordinates"])
    outlet = _as_numpy(boundary["outlet"]["coordinates"])
    walls = _as_numpy(boundary["walls"]["coordinates"])

    return {
        "coordinate_convention": COORDINATE_CONVENTION,
        "coordinate_description": COORDINATE_DESCRIPTION,
        "groups": {
            "inlet": _coordinate_summary(inlet),
            "outlet": _coordinate_summary(outlet),
            "walls": _coordinate_summary(walls),
            "interior": _coordinate_summary(interior),
        },
    }


def _pixel(point: np.ndarray, *, left: int, top: int, size: int) -> tuple[int, int]:
    x = left + round(float(point[0]) * size)
    y = top + round((1.0 - float(point[1])) * size)
    return x, y


def _draw_points(
    image: np.ndarray,
    points: np.ndarray,
    *,
    left: int,
    top: int,
    size: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for point in points:
        x, y = _pixel(point, left=left, top=top, size=size)
        _draw_rect(
            image,
            x0=x - radius,
            y0=y - radius,
            x1=x + radius + 1,
            y1=y + radius + 1,
            color=color,
        )


def write_boundary_diagnostic_plot(
    path: Path | str,
    *,
    report_path: Path | str | None = None,
    interior_points_per_axis: int = 9,
    boundary_points_per_patch: int = 9,
) -> dict[str, object]:
    """Write a PNG mask diagnostic and optionally the matching JSON report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = boundary_diagnostic_report(
        interior_points_per_axis=interior_points_per_axis,
        boundary_points_per_patch=boundary_points_per_patch,
    )

    interior = _as_numpy(interior_collocation_points(interior_points_per_axis))
    boundary = boundary_collocation_points(boundary_points_per_patch)
    inlet = _as_numpy(boundary["inlet"]["coordinates"])
    outlet = _as_numpy(boundary["outlet"]["coordinates"])
    walls = _as_numpy(boundary["walls"]["coordinates"])

    size = 320
    top = 42
    left = 58
    image = np.full((430, 460, 3), 255, dtype=np.uint8)
    _draw_text(image, "boundary masks", x=118, y=12, scale=2, bold=True)
    _draw_rect(image, x0=left, y0=top, x1=left + size + 1, y1=top + 1, color=(0, 0, 0))
    _draw_rect(image, x0=left, y0=top + size, x1=left + size + 1, y1=top + size + 1, color=(0, 0, 0))
    _draw_rect(image, x0=left, y0=top, x1=left + 1, y1=top + size + 1, color=(0, 0, 0))
    _draw_rect(image, x0=left + size, y0=top, x1=left + size + 1, y1=top + size + 1, color=(0, 0, 0))
    _draw_boundary_markers(image, left=left, top=top, width=size, height=size, label_offset=12)
    _draw_points(image, interior, left=left, top=top, size=size, color=(60, 60, 60), radius=2)
    _draw_points(image, walls, left=left, top=top, size=size, color=(30, 160, 60), radius=3)
    _draw_points(image, inlet, left=left, top=top, size=size, color=BOUNDARY_MARKER_COLORS["inlet"], radius=4)
    _draw_points(image, outlet, left=left, top=top, size=size, color=BOUNDARY_MARKER_COLORS["outlet"], radius=4)
    _draw_text(image, "x", x=left + size // 2, y=top + size + 18, scale=1)
    _draw_text(image, "y", x=20, y=top + size // 2, scale=1)
    _draw_text(image, COORDINATE_DESCRIPTION, x=58, y=392, scale=1)
    _draw_text(image, "red inlet  blue outlet  green walls  gray interior", x=58, y=410, scale=1)
    _write_rgb_png(output, image)

    if report_path is not None:
        report_output = Path(report_path)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for boundary diagnostics."""

    parser = argparse.ArgumentParser(description="Write unit-square boundary diagnostics.")
    parser.add_argument("output", type=Path, help="PNG path for the boundary diagnostic plot")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--interior-points-per-axis", type=int, default=9)
    parser.add_argument("--boundary-points-per-patch", type=int, default=9)
    args = parser.parse_args(argv)
    report = write_boundary_diagnostic_plot(
        args.output,
        report_path=args.report,
        interior_points_per_axis=args.interior_points_per_axis,
        boundary_points_per_patch=args.boundary_points_per_patch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "COORDINATE_CONVENTION",
    "boundary_diagnostic_report",
    "main",
    "write_boundary_diagnostic_plot",
]
