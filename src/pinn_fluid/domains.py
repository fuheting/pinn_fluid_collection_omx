"""Model-agnostic domain definitions for repository flow examples."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

UNIT_SQUARE_DOMAIN: dict[str, list[float]] = {
    "x_range": [0.0, 1.0],
    "y_range": [0.0, 1.0],
}

_UNIT_SQUARE_FLOW_PATCHES: dict[str, dict[str, Any]] = {
    "inlet": {
        "location": "top",
        "x_range": [0.0, 0.25],
        "type": "dirichlet",
        "variable": "pressure",
        "value": 1.0,
    },
    "outlet": {
        "location": "bottom",
        "x_range": [0.75, 1.0],
        "type": "dirichlet",
        "variable": "pressure",
        "value": 0.0,
    },
    "walls": {
        "type": "neumann",
        "condition": "no_normal_flow",
    },
}


def unit_square_flow_patches() -> dict[str, dict[str, Any]]:
    """Return the shared unit-square boundary patches for flow models."""

    return deepcopy(_UNIT_SQUARE_FLOW_PATCHES)


def _validate_positive_count(count: int, name: str) -> None:
    if count <= 0:
        raise ValueError(f"{name} must be positive")


def _linspace(start: float, end: float, count: int, **tensor_kwargs: Any) -> torch.Tensor:
    return torch.linspace(start, end, count, **tensor_kwargs)


def interior_collocation_points(
    points_per_axis: int,
    *,
    domain: dict[str, list[float]] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return deterministic interior collocation points for the unit-square domain."""

    _validate_positive_count(points_per_axis, "points_per_axis")
    active_domain = UNIT_SQUARE_DOMAIN if domain is None else domain
    tensor_kwargs = {"dtype": dtype, "device": device}

    x_start, x_end = active_domain["x_range"]
    y_start, y_end = active_domain["y_range"]
    x_values = _linspace(x_start, x_end, points_per_axis + 2, **tensor_kwargs)[1:-1]
    y_values = _linspace(y_start, y_end, points_per_axis + 2, **tensor_kwargs)[1:-1]
    grid_x, grid_y = torch.meshgrid(x_values, y_values, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)


def _horizontal_segment(
    x_range: list[float],
    y_value: float,
    count: int,
    **tensor_kwargs: Any,
) -> torch.Tensor:
    x_values = _linspace(x_range[0], x_range[1], count, **tensor_kwargs)
    y_values = torch.full_like(x_values, y_value)
    return torch.stack((x_values, y_values), dim=1)


def _vertical_segment(
    x_value: float,
    y_range: list[float],
    count: int,
    **tensor_kwargs: Any,
) -> torch.Tensor:
    y_values = _linspace(y_range[0], y_range[1], count, **tensor_kwargs)
    x_values = torch.full_like(y_values, x_value)
    return torch.stack((x_values, y_values), dim=1)


def boundary_collocation_points(
    points_per_patch: int,
    *,
    patches: dict[str, dict[str, Any]] | None = None,
    domain: dict[str, list[float]] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return deterministic boundary collocation points for shared flow patches."""

    _validate_positive_count(points_per_patch, "points_per_patch")
    active_domain = UNIT_SQUARE_DOMAIN if domain is None else domain
    active_patches = unit_square_flow_patches() if patches is None else patches
    tensor_kwargs = {"dtype": dtype, "device": device}

    x_start, x_end = active_domain["x_range"]
    y_start, y_end = active_domain["y_range"]
    inlet = active_patches["inlet"]
    outlet = active_patches["outlet"]

    inlet_coordinates = _horizontal_segment(
        inlet["x_range"],
        y_end,
        points_per_patch,
        **tensor_kwargs,
    )
    outlet_coordinates = _horizontal_segment(
        outlet["x_range"],
        y_start,
        points_per_patch,
        **tensor_kwargs,
    )

    left_wall = _vertical_segment(x_start, [y_start, y_end], points_per_patch, **tensor_kwargs)
    right_wall = _vertical_segment(x_end, [y_start, y_end], points_per_patch, **tensor_kwargs)
    bottom_wall = _horizontal_segment(
        [x_start, outlet["x_range"][0]],
        y_start,
        points_per_patch,
        **tensor_kwargs,
    )
    top_wall = _horizontal_segment(
        [inlet["x_range"][1], x_end],
        y_end,
        points_per_patch,
        **tensor_kwargs,
    )
    wall_coordinates = torch.cat((left_wall, right_wall, bottom_wall, top_wall), dim=0)
    wall_normals = torch.cat(
        (
            torch.tensor([[-1.0, 0.0]], **tensor_kwargs).repeat(points_per_patch, 1),
            torch.tensor([[1.0, 0.0]], **tensor_kwargs).repeat(points_per_patch, 1),
            torch.tensor([[0.0, -1.0]], **tensor_kwargs).repeat(points_per_patch, 1),
            torch.tensor([[0.0, 1.0]], **tensor_kwargs).repeat(points_per_patch, 1),
        ),
        dim=0,
    )

    return {
        "inlet": {"coordinates": inlet_coordinates},
        "outlet": {"coordinates": outlet_coordinates},
        "walls": {"coordinates": wall_coordinates, "normals": wall_normals},
    }


__all__ = [
    "UNIT_SQUARE_DOMAIN",
    "boundary_collocation_points",
    "interior_collocation_points",
    "unit_square_flow_patches",
]
