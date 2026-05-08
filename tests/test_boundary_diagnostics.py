"""Diagnostics for shared unit-square boundary and collocation masks."""

import json

from pinn_fluid.diagnostics import (
    boundary_diagnostic_report,
    write_boundary_diagnostic_plot,
)


def _assert_png(path) -> None:
    assert path.is_file()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_boundary_diagnostic_report_documents_cartesian_top_left_inlet_and_bottom_right_outlet():
    report = boundary_diagnostic_report(interior_points_per_axis=2, boundary_points_per_patch=3)

    assert report["coordinate_convention"] == "cartesian_unit_square_y_up"
    assert report["groups"]["inlet"] == {
        "count": 3,
        "x_min": 0.0,
        "x_max": 0.25,
        "y_min": 1.0,
        "y_max": 1.0,
    }
    assert report["groups"]["outlet"] == {
        "count": 3,
        "x_min": 0.75,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 0.0,
    }
    assert report["groups"]["walls"]["count"] == 12
    assert report["groups"]["interior"]["count"] == 4


def test_boundary_diagnostic_plot_writes_png_and_json_report(tmp_path):
    plot_path = tmp_path / "boundary_masks.png"
    json_path = tmp_path / "boundary_masks.json"

    report = write_boundary_diagnostic_plot(
        plot_path,
        report_path=json_path,
        interior_points_per_axis=2,
        boundary_points_per_patch=3,
    )

    _assert_png(plot_path)
    assert json.loads(json_path.read_text()) == report
