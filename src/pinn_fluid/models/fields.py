"""Reusable neural field modules for PINN examples."""

from __future__ import annotations

import torch


class MLPField(torch.nn.Module):
    """Small fully connected coordinate-to-field network."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_width: int = 32,
        hidden_layers: int = 2,
        activation: type[torch.nn.Module] = torch.nn.Tanh,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")

        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(current_dim, hidden_width))
            layers.append(activation())
            current_dim = hidden_width
        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return field values at `coordinates`."""

        return self.network(coordinates)


__all__ = ["MLPField"]
