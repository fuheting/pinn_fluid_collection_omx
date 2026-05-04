"""Tests for the lightweight Oseen training foundation."""

import torch

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.models.stokes import StokesVelocityPressureField
from pinn_fluid.solvers.oseen import (
    OseenTrainingConfig,
    oseen_loss_components,
    oseen_total_loss,
    train_oseen_smoke,
)


class ZeroOseenField(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = coordinates[:, :1] * 0.0
        return {
            "u": zero,
            "v": zero,
            "pressure": zero,
        }


def test_oseen_loss_components_cover_interior_and_boundary_terms():
    interior = interior_collocation_points(2, dtype=torch.float64)
    boundary = boundary_collocation_points(2, dtype=torch.float64)

    components = oseen_loss_components(
        ZeroOseenField(),
        interior,
        boundary,
        viscosity=1.0,
        convection_velocity=(1.0, 0.0),
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


def test_oseen_total_loss_applies_default_phase_weights():
    interior = interior_collocation_points(2, dtype=torch.float64)
    boundary = boundary_collocation_points(2, dtype=torch.float64)

    total = oseen_total_loss(
        ZeroOseenField(),
        interior,
        boundary,
        viscosity=1.0,
        convection_velocity=(1.0, 0.0),
    )

    assert torch.allclose(total, torch.tensor(5.0, dtype=torch.float64))


def test_train_oseen_smoke_reduces_loss_on_shared_collocation_points():
    torch.manual_seed(0)
    field = StokesVelocityPressureField(hidden_width=8, hidden_layers=1)
    config = OseenTrainingConfig(
        interior_points_per_axis=2,
        boundary_points_per_patch=2,
        steps=20,
        learning_rate=0.03,
        viscosity=1.0,
        convection_velocity=(1.0, 0.0),
    )

    history = train_oseen_smoke(field, config)

    assert len(history) == 21
    assert history[-1] < history[0]
