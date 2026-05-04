"""Lightweight Oseen training utilities for the shared unit-square benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.models.oseen import DEFAULT_OSEEN_LOSS_WEIGHTS, oseen_residuals
from pinn_fluid.models.stokes import no_slip_residual


@dataclass(frozen=True)
class OseenTrainingConfig:
    """Configuration for the local Oseen training smoke loop."""

    interior_points_per_axis: int = 4
    boundary_points_per_patch: int = 4
    steps: int = 50
    learning_rate: float = 1e-2
    viscosity: float = 1.0
    convection_velocity: tuple[float, float] = (1.0, 0.0)


def _coordinates_for_autograd(coordinates: torch.Tensor) -> torch.Tensor:
    return coordinates.detach().clone().requires_grad_(True)


def _velocity(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat((outputs["u"], outputs["v"]), dim=1)


def _convection_tensor(
    convection_velocity: tuple[float, float],
    coordinates: torch.Tensor,
) -> torch.Tensor:
    return torch.tensor(
        convection_velocity,
        dtype=coordinates.dtype,
        device=coordinates.device,
    ).reshape(1, 2).expand(coordinates.shape[0], 2)


def oseen_loss_components(
    model: torch.nn.Module,
    interior_coordinates: torch.Tensor,
    boundary_samples: dict[str, dict[str, torch.Tensor]],
    *,
    viscosity: float,
    convection_velocity: tuple[float, float],
    inlet_velocity: tuple[float, float] = (0.0, -1.0),
) -> dict[str, torch.Tensor]:
    """Return mean-squared Oseen residual losses for one collocation batch."""

    interior = _coordinates_for_autograd(interior_coordinates)
    interior_outputs = model(interior)
    residuals = oseen_residuals(
        interior_outputs["u"],
        interior_outputs["v"],
        interior_outputs["pressure"],
        interior,
        _convection_tensor(convection_velocity, interior),
        viscosity=viscosity,
    )

    inlet_coordinates = boundary_samples["inlet"]["coordinates"]
    inlet_outputs = model(inlet_coordinates)
    inlet_target = torch.tensor(
        inlet_velocity,
        dtype=inlet_coordinates.dtype,
        device=inlet_coordinates.device,
    ).reshape(1, 2)

    outlet_outputs = model(boundary_samples["outlet"]["coordinates"])
    wall_outputs = model(boundary_samples["walls"]["coordinates"])

    return {
        "continuity": residuals["continuity"].square().mean(),
        "x_momentum": residuals["x_momentum"].square().mean(),
        "y_momentum": residuals["y_momentum"].square().mean(),
        "inlet": (_velocity(inlet_outputs) - inlet_target).square().mean(),
        "outlet": outlet_outputs["pressure"].square().mean(),
        "wall": no_slip_residual(_velocity(wall_outputs)).square().mean(),
    }


def oseen_total_loss(
    model: torch.nn.Module,
    interior_coordinates: torch.Tensor,
    boundary_samples: dict[str, dict[str, torch.Tensor]],
    *,
    viscosity: float,
    convection_velocity: tuple[float, float],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return the weighted Oseen residual loss for one collocation batch."""

    active_weights = DEFAULT_OSEEN_LOSS_WEIGHTS if weights is None else weights
    components = oseen_loss_components(
        model,
        interior_coordinates,
        boundary_samples,
        viscosity=viscosity,
        convection_velocity=convection_velocity,
    )
    weighted_terms = [
        active_weights[name] * components[name]
        for name in ("continuity", "x_momentum", "y_momentum", "inlet", "outlet", "wall")
    ]
    return sum(weighted_terms)


def train_oseen_smoke(
    model: torch.nn.Module,
    config: OseenTrainingConfig,
) -> list[float]:
    """Run a tiny deterministic Oseen optimization loop and return loss history."""

    interior = interior_collocation_points(config.interior_points_per_axis)
    boundary = boundary_collocation_points(config.boundary_points_per_patch)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def current_loss() -> torch.Tensor:
        return oseen_total_loss(
            model,
            interior,
            boundary,
            viscosity=config.viscosity,
            convection_velocity=config.convection_velocity,
        )

    history = [float(current_loss().detach())]

    for _ in range(config.steps):
        optimizer.zero_grad()
        loss = current_loss()
        loss.backward()
        optimizer.step()
        history.append(float(current_loss().detach()))

    return history


__all__ = [
    "OseenTrainingConfig",
    "oseen_loss_components",
    "oseen_total_loss",
    "train_oseen_smoke",
]
