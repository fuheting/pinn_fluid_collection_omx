"""Stokes-flow residual utilities for the shared unit-square example."""

from __future__ import annotations

from typing import Any

import torch

from pinn_fluid.models.fields import MLPField

DEFAULT_STOKES_LOSS_WEIGHTS: dict[str, float] = {
    "continuity": 1.0,
    "x_momentum": 1.0,
    "y_momentum": 1.0,
    "inlet": 10.0,
    "outlet": 10.0,
    "wall": 10.0,
}


class StokesVelocityPressureField(torch.nn.Module):
    """Minimal neural field for Stokes velocity `(u, v)` and pressure `p`."""

    def __init__(self, *, hidden_width: int = 32, hidden_layers: int = 2) -> None:
        super().__init__()
        self.field = MLPField(
            input_dim=2,
            output_dim=3,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
        )

    def forward(self, coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named Stokes field components at `coordinates`."""

        values = self.field(coordinates)
        return {
            "u": values[:, 0:1],
            "v": values[:, 1:2],
            "pressure": values[:, 2:3],
        }


def _gradient(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        field,
        coordinates,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]


def _axis_second_derivative(
    gradient_component: torch.Tensor,
    coordinates: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    if gradient_component.requires_grad:
        second_gradient = torch.autograd.grad(
            gradient_component,
            coordinates,
            grad_outputs=torch.ones_like(gradient_component),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
    else:
        second_gradient = None
    if second_gradient is None:
        return torch.zeros_like(gradient_component)
    return second_gradient[:, axis : axis + 1]


def _laplacian(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    gradient = _gradient(field, coordinates)
    terms = [
        _axis_second_derivative(gradient[:, axis : axis + 1], coordinates, axis)
        for axis in range(coordinates.shape[1])
    ]
    return sum(terms)


def stokes_residuals(
    u: torch.Tensor,
    v: torch.Tensor,
    pressure: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    viscosity: float,
) -> dict[str, torch.Tensor]:
    """Return steady incompressible Stokes residuals for `(u, v, p)`."""

    u_gradient = _gradient(u, coordinates)
    v_gradient = _gradient(v, coordinates)
    pressure_gradient = _gradient(pressure, coordinates)

    continuity = u_gradient[:, :1] + v_gradient[:, 1:2]
    x_momentum = -pressure_gradient[:, :1] + viscosity * _laplacian(u, coordinates)
    y_momentum = -pressure_gradient[:, 1:2] + viscosity * _laplacian(v, coordinates)

    return {
        "continuity": continuity,
        "x_momentum": x_momentum,
        "y_momentum": y_momentum,
    }


def no_slip_residual(wall_velocity: torch.Tensor) -> torch.Tensor:
    """Return wall velocity as the no-slip residual."""

    return wall_velocity


def stokes_boundary_targets(
    patches: dict[str, dict[str, Any]],
    *,
    inlet_velocity: tuple[float, float] = (0.0, -1.0),
    wall_velocity: tuple[float, float] = (0.0, 0.0),
) -> dict[str, dict[str, Any]]:
    """Map shared flow patches to Stokes velocity and pressure targets."""

    inlet = patches["inlet"]
    outlet = patches["outlet"]
    return {
        "inlet": {
            "location": inlet["location"],
            "x_range": list(inlet["x_range"]),
            "variable": "velocity",
            "value": list(inlet_velocity),
        },
        "outlet": {
            "location": outlet["location"],
            "x_range": list(outlet["x_range"]),
            "variable": "pressure",
            "value": outlet["value"],
        },
        "walls": {
            "variable": "velocity",
            "condition": "no_slip",
            "value": list(wall_velocity),
        },
    }


__all__ = [
    "DEFAULT_STOKES_LOSS_WEIGHTS",
    "StokesVelocityPressureField",
    "no_slip_residual",
    "stokes_residuals",
    "stokes_boundary_targets",
]
