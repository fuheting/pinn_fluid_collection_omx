"""Tests for shared collocation sampling foundations."""

import torch

from pinn_fluid.domains import (
    boundary_collocation_points,
    interior_collocation_points,
    unit_square_flow_patches,
)
from pinn_fluid.models.stokes import stokes_boundary_targets


def test_interior_collocation_points_stay_inside_unit_square():
    points = interior_collocation_points(3, dtype=torch.float64)

    expected = torch.tensor(
        [
            [0.25, 0.25],
            [0.25, 0.50],
            [0.25, 0.75],
            [0.50, 0.25],
            [0.50, 0.50],
            [0.50, 0.75],
            [0.75, 0.25],
            [0.75, 0.50],
            [0.75, 0.75],
        ],
        dtype=torch.float64,
    )

    assert points.shape == (9, 2)
    assert torch.allclose(points, expected)
    assert torch.all(points > 0.0)
    assert torch.all(points < 1.0)


def test_boundary_collocation_points_follow_shared_patches():
    patches = unit_square_flow_patches()

    samples = boundary_collocation_points(3, patches=patches, dtype=torch.float64)

    assert torch.allclose(
        samples["inlet"]["coordinates"],
        torch.tensor(
            [[0.0, 1.0], [0.125, 1.0], [0.25, 1.0]],
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        samples["outlet"]["coordinates"],
        torch.tensor(
            [[0.75, 0.0], [0.875, 0.0], [1.0, 0.0]],
            dtype=torch.float64,
        ),
    )

    wall_coordinates = samples["walls"]["coordinates"]
    wall_normals = samples["walls"]["normals"]

    assert wall_coordinates.shape == (12, 2)
    assert wall_normals.shape == (12, 2)
    assert torch.allclose(wall_coordinates[:3, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(wall_normals[:3], torch.tensor([[-1.0, 0.0]] * 3, dtype=torch.float64))
    assert torch.allclose(wall_coordinates[3:6, 0], torch.ones(3, dtype=torch.float64))
    assert torch.allclose(wall_normals[3:6], torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64))
    assert torch.allclose(
        wall_coordinates[6:9],
        torch.tensor([[0.0, 0.0], [0.375, 0.0], [0.75, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(wall_normals[6:9], torch.tensor([[0.0, -1.0]] * 3, dtype=torch.float64))
    assert torch.allclose(
        wall_coordinates[9:12],
        torch.tensor([[0.25, 1.0], [0.625, 1.0], [1.0, 1.0]], dtype=torch.float64),
    )
    assert torch.allclose(wall_normals[9:12], torch.tensor([[0.0, 1.0]] * 3, dtype=torch.float64))


def test_boundary_collocation_rejects_non_positive_counts():
    for sampler in (interior_collocation_points, boundary_collocation_points):
        try:
            sampler(0)
        except ValueError as exc:
            assert "positive" in str(exc)
        else:
            raise AssertionError("expected non-positive sample counts to be rejected")


def test_stokes_boundary_targets_are_derived_from_shared_patches_without_mutation():
    patches = unit_square_flow_patches()

    targets = stokes_boundary_targets(patches)

    assert targets == {
        "inlet": {
            "location": "top",
            "x_range": [0.0, 0.25],
            "variable": "velocity",
            "value": [0.0, -1.0],
        },
        "outlet": {
            "location": "bottom",
            "x_range": [0.75, 1.0],
            "variable": "pressure",
            "value": 0.0,
        },
        "walls": {
            "variable": "velocity",
            "condition": "no_slip",
            "value": [0.0, 0.0],
        },
    }
    assert patches == unit_square_flow_patches()
