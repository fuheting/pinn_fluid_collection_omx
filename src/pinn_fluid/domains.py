"""Model-agnostic domain definitions for repository flow examples."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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


__all__ = ["UNIT_SQUARE_DOMAIN", "unit_square_flow_patches"]
