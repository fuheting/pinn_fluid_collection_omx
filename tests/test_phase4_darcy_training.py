"""Tests for the lightweight Darcy training foundation."""

import torch

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.models.darcy import DarcyPressureField
from pinn_fluid.solvers.darcy import (
    DarcyTrainingConfig,
    darcy_loss_components,
    darcy_total_loss,
    train_darcy_smoke,
)


class ZeroPressure(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates[:, :1] * 0.0


def test_darcy_loss_components_cover_interior_and_boundary_terms():
    interior = interior_collocation_points(2, dtype=torch.float64)
    boundary = boundary_collocation_points(2, dtype=torch.float64)

    components = darcy_loss_components(ZeroPressure(), interior, boundary)

    assert set(components) == {"interior", "inlet", "outlet", "wall"}
    assert torch.allclose(components["interior"], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(components["inlet"], torch.tensor(0.5, dtype=torch.float64))
    assert torch.allclose(components["outlet"], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(components["wall"], torch.tensor(0.0, dtype=torch.float64))


def test_darcy_total_loss_applies_default_phase_weights():
    interior = interior_collocation_points(2, dtype=torch.float64)
    boundary = boundary_collocation_points(2, dtype=torch.float64)

    total = darcy_total_loss(ZeroPressure(), interior, boundary)

    assert torch.allclose(total, torch.tensor(5.0, dtype=torch.float64))


def test_train_darcy_smoke_reduces_loss_on_shared_collocation_points():
    torch.manual_seed(0)
    field = DarcyPressureField(hidden_width=8, hidden_layers=1)
    config = DarcyTrainingConfig(
        interior_points_per_axis=2,
        boundary_points_per_patch=2,
        steps=20,
        learning_rate=0.05,
    )

    history = train_darcy_smoke(field, config)

    assert len(history) == 21
    assert history[-1] < history[0]
