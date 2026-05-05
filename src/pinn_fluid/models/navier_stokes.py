"""Navier-Stokes residual utilities for the shared unit-square example."""

from __future__ import annotations

import torch

DEFAULT_NAVIER_STOKES_LOSS_WEIGHTS: dict[str, float] = {
    "continuity": 1.0,
    "x_momentum": 1.0,
    "y_momentum": 1.0,
    "inlet": 10.0,
    "outlet": 10.0,
    "wall": 10.0,
}


def poiseuille_pressure_drop(
    viscosity: float,
    peak_velocity: float = 1.0,
    *,
    length: float = 1.0,
) -> float:
    """Return the pressure drop for the unit-height parabolic channel profile."""

    return 8.0 * viscosity * peak_velocity * length


def poiseuille_channel_solution(
    coordinates: torch.Tensor,
    *,
    viscosity: float,
    peak_velocity: float = 1.0,
    pressure_reference: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return an analytic horizontal Poiseuille solution on the unit square."""

    x = coordinates[:, :1]
    y = coordinates[:, 1:2]
    u = 4.0 * peak_velocity * y * (1.0 - y)
    v = x * 0.0
    pressure = pressure_reference - poiseuille_pressure_drop(viscosity, peak_velocity) * x
    return {
        "u": u,
        "v": v,
        "pressure": pressure,
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


def _convective_derivative(
    field: torch.Tensor,
    coordinates: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    gradient = _gradient(field, coordinates)
    return (velocity * gradient).sum(dim=1, keepdim=True)


def navier_stokes_residuals(
    u: torch.Tensor,
    v: torch.Tensor,
    pressure: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    viscosity: float,
) -> dict[str, torch.Tensor]:
    """Return steady incompressible Navier-Stokes residuals for `(u, v, p)`."""

    u_gradient = _gradient(u, coordinates)
    v_gradient = _gradient(v, coordinates)
    pressure_gradient = _gradient(pressure, coordinates)
    velocity = torch.cat((u, v), dim=1)

    continuity = u_gradient[:, :1] + v_gradient[:, 1:2]
    x_momentum = (
        _convective_derivative(u, coordinates, velocity)
        - pressure_gradient[:, :1]
        + viscosity * _laplacian(u, coordinates)
    )
    y_momentum = (
        _convective_derivative(v, coordinates, velocity)
        - pressure_gradient[:, 1:2]
        + viscosity * _laplacian(v, coordinates)
    )

    return {
        "continuity": continuity,
        "x_momentum": x_momentum,
        "y_momentum": y_momentum,
    }


__all__ = [
    "DEFAULT_NAVIER_STOKES_LOSS_WEIGHTS",
    "navier_stokes_residuals",
    "poiseuille_channel_solution",
    "poiseuille_pressure_drop",
]
