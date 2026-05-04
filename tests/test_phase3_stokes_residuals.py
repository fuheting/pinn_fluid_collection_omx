"""Phase 3 tests for the Stokes-flow residual foundation."""

import torch

from pinn_fluid.models.stokes import (
    DEFAULT_STOKES_LOSS_WEIGHTS,
    no_slip_residual,
    stokes_residuals,
)


def test_stokes_loss_weights_match_phase3_formulation():
    assert DEFAULT_STOKES_LOSS_WEIGHTS == {
        "continuity": 1.0,
        "x_momentum": 1.0,
        "y_momentum": 1.0,
        "inlet": 10.0,
        "outlet": 10.0,
        "wall": 10.0,
    }


def test_constant_pressure_and_zero_velocity_satisfy_stokes_residuals():
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.7], [0.9, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    u = points[:, :1] * 0.0
    v = points[:, 1:2] * 0.0
    pressure = points[:, :1] * 0.0 + 3.0

    residuals = stokes_residuals(u, v, pressure, points, viscosity=1.0)

    assert set(residuals) == {"continuity", "x_momentum", "y_momentum"}
    for residual in residuals.values():
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)


def test_linear_couette_field_has_zero_momentum_and_constant_continuity():
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    u = points[:, :1]
    v = points[:, 1:2] * 0.0
    pressure = points[:, :1] * 0.0

    residuals = stokes_residuals(u, v, pressure, points, viscosity=0.5)

    assert torch.allclose(residuals["continuity"], torch.ones_like(u), atol=1e-10)
    assert torch.allclose(residuals["x_momentum"], torch.zeros_like(u), atol=1e-10)
    assert torch.allclose(residuals["y_momentum"], torch.zeros_like(u), atol=1e-10)


def test_no_slip_residual_returns_velocity_components():
    wall_velocity = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4]],
        dtype=torch.float64,
    )

    residual = no_slip_residual(wall_velocity)

    assert torch.equal(residual, wall_velocity)
