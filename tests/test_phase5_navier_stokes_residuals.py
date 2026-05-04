"""Phase 5 tests for the Navier-Stokes residual foundation."""

import torch

from pinn_fluid.models.navier_stokes import (
    DEFAULT_NAVIER_STOKES_LOSS_WEIGHTS,
    navier_stokes_residuals,
)
from pinn_fluid.models.oseen import oseen_residuals


def test_navier_stokes_loss_weights_match_phase5_foundation():
    assert DEFAULT_NAVIER_STOKES_LOSS_WEIGHTS == {
        "continuity": 1.0,
        "x_momentum": 1.0,
        "y_momentum": 1.0,
        "inlet": 10.0,
        "outlet": 10.0,
        "wall": 10.0,
    }


def test_zero_velocity_and_constant_pressure_have_zero_residuals():
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.7], [0.9, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    zero = points[:, :1] * 0.0
    pressure = zero + 2.0

    residuals = navier_stokes_residuals(zero, zero, pressure, points, viscosity=0.5)

    assert set(residuals) == {"continuity", "x_momentum", "y_momentum"}
    assert torch.allclose(residuals["continuity"], zero, atol=1e-10)
    assert torch.allclose(residuals["x_momentum"], zero, atol=1e-10)
    assert torch.allclose(residuals["y_momentum"], zero, atol=1e-10)


def test_navier_stokes_matches_oseen_when_convection_velocity_is_current_velocity():
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    u = points[:, :1]
    v = points[:, 1:2] * 2.0
    pressure = points[:, :1] * 0.0
    velocity = torch.cat((u, v), dim=1)

    navier_stokes = navier_stokes_residuals(u, v, pressure, points, viscosity=0.5)
    oseen = oseen_residuals(u, v, pressure, points, velocity, viscosity=0.5)

    for name in oseen:
        assert torch.allclose(navier_stokes[name], oseen[name], atol=1e-10)


def test_linear_velocity_field_keeps_nonlinear_advection_explicit():
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    u = points[:, :1]
    v = points[:, 1:2] * 2.0
    pressure = points[:, :1] * 0.0

    residuals = navier_stokes_residuals(u, v, pressure, points, viscosity=0.5)

    assert torch.allclose(residuals["continuity"], torch.full_like(u, 3.0), atol=1e-10)
    assert torch.allclose(residuals["x_momentum"], u, atol=1e-10)
    assert torch.allclose(residuals["y_momentum"], v * 2.0, atol=1e-10)
