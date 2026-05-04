"""Lightweight Darcy training utilities for the shared unit-square benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.models.darcy import (
    DEFAULT_DARCY_LOSS_WEIGHTS,
    boundary_residuals,
    laplace_residual,
    pressure_gradient,
)


@dataclass(frozen=True)
class DarcyTrainingConfig:
    """Configuration for the local Darcy training smoke loop."""

    interior_points_per_axis: int = 4
    boundary_points_per_patch: int = 4
    steps: int = 50
    learning_rate: float = 1e-2


def _coordinates_for_autograd(coordinates: torch.Tensor) -> torch.Tensor:
    return coordinates.detach().clone().requires_grad_(True)


def darcy_loss_components(
    model: torch.nn.Module,
    interior_coordinates: torch.Tensor,
    boundary_samples: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Return mean-squared Darcy residual losses for one collocation batch."""

    interior = _coordinates_for_autograd(interior_coordinates)
    pressure = model(interior)
    interior_residual = laplace_residual(pressure, interior)

    inlet_coordinates = boundary_samples["inlet"]["coordinates"]
    outlet_coordinates = boundary_samples["outlet"]["coordinates"]
    wall_coordinates = _coordinates_for_autograd(boundary_samples["walls"]["coordinates"])
    wall_pressure = model(wall_coordinates)
    wall_gradients = pressure_gradient(wall_pressure, wall_coordinates)

    residuals = boundary_residuals(
        inlet_pressure=model(inlet_coordinates),
        outlet_pressure=model(outlet_coordinates),
        wall_gradients=wall_gradients,
        wall_normals=boundary_samples["walls"]["normals"],
    )

    return {
        "interior": interior_residual.square().mean(),
        "inlet": residuals["inlet"].square().mean(),
        "outlet": residuals["outlet"].square().mean(),
        "wall": residuals["wall"].square().mean(),
    }


def darcy_total_loss(
    model: torch.nn.Module,
    interior_coordinates: torch.Tensor,
    boundary_samples: dict[str, dict[str, torch.Tensor]],
    *,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return the weighted Darcy residual loss for one collocation batch."""

    active_weights = DEFAULT_DARCY_LOSS_WEIGHTS if weights is None else weights
    components = darcy_loss_components(model, interior_coordinates, boundary_samples)
    weighted_terms = [
        active_weights[name] * components[name]
        for name in ("interior", "inlet", "outlet", "wall")
    ]
    return sum(weighted_terms)


def train_darcy_smoke(
    model: torch.nn.Module,
    config: DarcyTrainingConfig,
) -> list[float]:
    """Run a tiny deterministic Darcy optimization loop and return loss history."""

    interior = interior_collocation_points(config.interior_points_per_axis)
    boundary = boundary_collocation_points(config.boundary_points_per_patch)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = [float(darcy_total_loss(model, interior, boundary).detach())]

    for _ in range(config.steps):
        optimizer.zero_grad()
        loss = darcy_total_loss(model, interior, boundary)
        loss.backward()
        optimizer.step()
        history.append(float(darcy_total_loss(model, interior, boundary).detach()))

    return history


__all__ = [
    "DarcyTrainingConfig",
    "darcy_loss_components",
    "darcy_total_loss",
    "train_darcy_smoke",
]
