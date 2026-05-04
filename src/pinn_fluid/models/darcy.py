"""Darcy-flow residual utilities for the Phase 2 unit-square example."""

from __future__ import annotations

import torch

from pinn_fluid.models.fields import MLPField

DEFAULT_DARCY_LOSS_WEIGHTS: dict[str, float] = {
    "interior": 1.0,
    "inlet": 10.0,
    "outlet": 10.0,
    "wall": 1.0,
}


class DarcyPressureField(MLPField):
    """Minimal neural pressure field `p_theta(x, y)` for Darcy flow."""

    def __init__(self, *, hidden_width: int = 32, hidden_layers: int = 2) -> None:
        super().__init__(
            input_dim=2,
            output_dim=1,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
        )


def pressure_gradient(pressure: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Return the pressure gradient with respect to `(x, y)` coordinates."""

    return torch.autograd.grad(
        pressure,
        coordinates,
        grad_outputs=torch.ones_like(pressure),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplace_residual(pressure: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Return `p_xx + p_yy` for a scalar pressure field."""

    gradient = pressure_gradient(pressure, coordinates)
    terms = []
    for axis in range(coordinates.shape[1]):
        component = gradient[:, axis : axis + 1]
        if component.requires_grad:
            second_gradient = torch.autograd.grad(
                component,
                coordinates,
                grad_outputs=torch.ones_like(component),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            second_gradient = None
        if second_gradient is None:
            second = torch.zeros_like(component)
        else:
            second = second_gradient[:, axis : axis + 1]
        terms.append(second)
    return sum(terms)


def darcy_velocity(pressure: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Return Darcy velocity for constant permeability `K = 1`."""

    return -pressure_gradient(pressure, coordinates)


def boundary_residuals(
    *,
    inlet_pressure: torch.Tensor,
    outlet_pressure: torch.Tensor,
    wall_gradients: torch.Tensor,
    wall_normals: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return residuals for inlet, outlet, and no-normal-flow walls."""

    wall_flux = (wall_gradients * wall_normals).sum(dim=1, keepdim=True)
    return {
        "inlet": inlet_pressure - 1.0,
        "outlet": outlet_pressure,
        "wall": wall_flux,
    }


__all__ = [
    "DarcyPressureField",
    "DEFAULT_DARCY_LOSS_WEIGHTS",
    "boundary_residuals",
    "darcy_velocity",
    "laplace_residual",
    "pressure_gradient",
]
