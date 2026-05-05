"""Tests for the lightweight Navier-Stokes training foundation."""

import torch

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.models.stokes import StokesVelocityPressureField
from pinn_fluid.solvers.navier_stokes import (
    NavierStokesTrainingConfig,
    navier_stokes_loss_components,
    navier_stokes_total_loss,
    train_navier_stokes_smoke,
)


class ZeroNavierStokesField(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = coordinates[:, :1] * 0.0
        return {
            "u": zero,
            "v": zero,
            "pressure": zero,
        }


def test_navier_stokes_loss_components_cover_interior_and_boundary_terms():
    interior = interior_collocation_points(2, dtype=torch.float64)
    boundary = boundary_collocation_points(2, dtype=torch.float64)

    components = navier_stokes_loss_components(
        ZeroNavierStokesField(),
        interior,
        boundary,
        viscosity=1.0,
    )

    assert set(components) == {
        "continuity",
        "x_momentum",
        "y_momentum",
        "inlet",
        "outlet",
        "wall",
    }
    assert torch.allclose(components["continuity"], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(components["x_momentum"], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(components["y_momentum"], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(components["inlet"], torch.tensor(0.5, dtype=torch.float64))
    assert torch.allclose(components["outlet"], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(components["wall"], torch.tensor(0.0, dtype=torch.float64))


def test_navier_stokes_total_loss_applies_default_phase_weights():
    interior = interior_collocation_points(2, dtype=torch.float64)
    boundary = boundary_collocation_points(2, dtype=torch.float64)

    total = navier_stokes_total_loss(
        ZeroNavierStokesField(),
        interior,
        boundary,
        viscosity=1.0,
    )

    assert torch.allclose(total, torch.tensor(5.0, dtype=torch.float64))


def test_train_navier_stokes_smoke_reduces_loss_on_shared_collocation_points():
    torch.manual_seed(0)
    field = StokesVelocityPressureField(hidden_width=8, hidden_layers=1)
    config = NavierStokesTrainingConfig(
        interior_points_per_axis=2,
        boundary_points_per_patch=2,
        steps=20,
        learning_rate=0.03,
        viscosity=1.0,
    )

    history = train_navier_stokes_smoke(field, config)

    assert len(history) == 21
    assert history[-1] < history[0]
