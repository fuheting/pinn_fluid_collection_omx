"""Phase 4 tests for the Oseen residual foundation."""

import torch

from pinn_fluid.models.oseen import DEFAULT_OSEEN_LOSS_WEIGHTS, oseen_residuals
from pinn_fluid.models.stokes import stokes_residuals


def test_oseen_loss_weights_match_phase4_foundation():
    assert DEFAULT_OSEEN_LOSS_WEIGHTS == {
        "continuity": 1.0,
        "x_momentum": 1.0,
        "y_momentum": 1.0,
        "inlet": 10.0,
        "outlet": 10.0,
        "wall": 10.0,
    }


def test_oseen_residuals_reduce_to_stokes_when_convection_is_zero():
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.7], [0.9, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    u = points[:, :1] ** 2
    v = points[:, 1:2] ** 2
    pressure = points[:, :1] + 3.0 * points[:, 1:2]
    convection = torch.zeros_like(points)

    oseen = oseen_residuals(u, v, pressure, points, convection, viscosity=0.5)
    stokes = stokes_residuals(u, v, pressure, points, viscosity=0.5)

    assert set(oseen) == {"continuity", "x_momentum", "y_momentum"}
    for name in stokes:
        assert torch.allclose(oseen[name], stokes[name], atol=1e-10)


def test_linear_field_keeps_convection_terms_explicit():
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    u = points[:, :1]
    v = points[:, 1:2] * 0.0
    pressure = points[:, :1] * 0.0
    convection = torch.tensor(
        [[2.0, 3.0], [2.0, 3.0]],
        dtype=torch.float64,
    )

    residuals = oseen_residuals(u, v, pressure, points, convection, viscosity=0.5)

    assert torch.allclose(residuals["continuity"], torch.ones_like(u), atol=1e-10)
    assert torch.allclose(residuals["x_momentum"], torch.full_like(u, 2.0), atol=1e-10)
    assert torch.allclose(residuals["y_momentum"], torch.zeros_like(u), atol=1e-10)
