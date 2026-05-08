"""Boundary and collocation diagnostics for the shared unit-square flow domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.figures import BOUNDARY_MARKERS

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

    fig, axis = plt.subplots(figsize=(5.6, 5.2))
    axis.scatter(interior[:, 0], interior[:, 1], color="0.35", s=16, label="interior")
    axis.scatter(walls[:, 0], walls[:, 1], color="forestgreen", s=26, label="walls")
    axis.scatter(inlet[:, 0], inlet[:, 1], color="red", s=38, label="inlet")
    axis.scatter(outlet[:, 0], outlet[:, 1], color="blue", s=38, label="outlet")
    axis.plot(BOUNDARY_MARKERS["inlet"]["x_range"], [1.0, 1.0], color="red", linewidth=4)
    axis.plot(BOUNDARY_MARKERS["outlet"]["x_range"], [0.0, 0.0], color="blue", linewidth=4)
    axis.set_title("Boundary masks")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.05, 1.05)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.5, alpha=0.4)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    fig.text(0.12, 0.02, COORDINATE_DESCRIPTION, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)

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
