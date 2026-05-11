"""Phase 2 tests for the shared domain and Darcy-flow instance."""

import torch

from pinn_fluid.domains import unit_square_flow_patches
from pinn_fluid.models.darcy import (
    DEFAULT_DARCY_LOSS_WEIGHTS,
    boundary_residuals,
    darcy_velocity,
    laplace_residual,
)


def test_unit_square_boundary_patches_match_reference_problem():
    patches = unit_square_flow_patches()

    assert patches["inlet"] == {
        "location": "top",
        "x_range": [0.0, 0.25],
        "type": "dirichlet",
        "variable": "pressure",
        "value": 1.0,
    }
    assert patches["outlet"] == {
        "location": "bottom",
        "x_range": [0.75, 1.0],
        "type": "dirichlet",
        "variable": "pressure",
        "value": 0.0,
    }
    assert patches["walls"] == {
        "type": "neumann",
        "condition": "no_normal_flow",
    }


def test_darcy_loss_weights_match_phase2_formulation():
    assert DEFAULT_DARCY_LOSS_WEIGHTS == {
        "interior": 1.0,
        "inlet": 10.0,
        "outlet": 10.0,
        "wall": 1.0,
    }


def test_laplace_residual_is_zero_for_harmonic_pressure():
    points = torch.tensor(
        [[0.2, 0.2], [0.4, 0.5], [0.8, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    pressure = points[:, :1] + 2.0 * points[:, 1:2]

    residual = laplace_residual(pressure, points)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)


def test_darcy_velocity_is_negative_pressure_gradient():
    points = torch.tensor(
        [[0.2, 0.2], [0.4, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    pressure = points[:, :1] + 2.0 * points[:, 1:2]

    velocity = darcy_velocity(pressure, points)

    expected = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]], dtype=torch.float64)
    assert torch.allclose(velocity, expected, atol=1e-10)


def test_boundary_residuals_follow_velocity_inlet_pressure_outlet_and_no_normal_flow():
    inlet_gradients = torch.tensor([[0.0, 1.0], [0.2, 0.75]], dtype=torch.float64)
    outlet_pressure = torch.tensor([[0.2], [-0.1]], dtype=torch.float64)
    wall_gradients = torch.tensor(
        [[0.0, 0.5], [1.5, 0.0], [0.0, -0.25]],
        dtype=torch.float64,
    )
    wall_normals = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float64,
    )

    residuals = boundary_residuals(
        inlet_gradients=inlet_gradients,
        outlet_pressure=outlet_pressure,
        wall_gradients=wall_gradients,
        wall_normals=wall_normals,
        inlet_velocity=(0.0, -1.0),
    )

    expected_inlet = torch.tensor([[0.0, 0.0], [-0.2, 0.25]], dtype=torch.float64)
    assert torch.allclose(residuals["inlet"], expected_inlet)
    assert torch.allclose(residuals["outlet"], outlet_pressure)
    assert torch.allclose(residuals["wall"], torch.zeros((3, 1), dtype=torch.float64))
