"""Stokes-flow residual utilities for the shared unit-square example."""

from __future__ import annotations

import torch

DEFAULT_STOKES_LOSS_WEIGHTS: dict[str, float] = {
    "continuity": 1.0,
    "x_momentum": 1.0,
    "y_momentum": 1.0,
    "inlet": 10.0,
    "outlet": 10.0,
    "wall": 10.0,
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


__all__ = [
    "DEFAULT_STOKES_LOSS_WEIGHTS",
    "no_slip_residual",
    "stokes_residuals",
]
