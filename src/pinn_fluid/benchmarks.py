"""Consolidated smoke benchmark summaries for implemented flow models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pinn_fluid.models.darcy import DarcyPressureField
from pinn_fluid.models.stokes import StokesVelocityPressureField
from pinn_fluid.solvers.darcy import DarcyTrainingConfig, train_darcy_smoke
from pinn_fluid.solvers.navier_stokes import (
    NavierStokesTrainingConfig,
    train_navier_stokes_smoke,
)
from pinn_fluid.solvers.oseen import OseenTrainingConfig, train_oseen_smoke
from pinn_fluid.solvers.stokes import StokesTrainingConfig, train_stokes_smoke


@dataclass(frozen=True)
class SmokeBenchmarkConfig:
    """Configuration shared by the local smoke benchmark summaries."""

    seed: int = 0
    hidden_width: int = 8
    hidden_layers: int = 1
    interior_points_per_axis: int = 2
    boundary_points_per_patch: int = 2
    steps: int = 20
    learning_rate: float = 0.03
    viscosity: float = 1.0
    oseen_convection_velocity: tuple[float, float] = (1.0, 0.0)


@dataclass(frozen=True)
class SmokeBenchmarkResult:
    """Loss summary for one local smoke benchmark run."""

    model: str
    initial_loss: float
    final_loss: float
    steps: int

    @property
    def reduction(self) -> float:
        """Return the absolute loss reduction."""

        return self.initial_loss - self.final_loss

    @property
    def reduction_ratio(self) -> float:
        """Return the fractional loss reduction relative to the initial loss."""

        if self.initial_loss == 0.0:
            return 0.0
        return self.reduction / self.initial_loss

    @property
    def reduced(self) -> bool:
        """Return whether the final loss is lower than the initial loss."""

        return self.final_loss < self.initial_loss


def summarize_loss_history(model: str, history: list[float]) -> SmokeBenchmarkResult:
    """Return a structured loss summary for one smoke-training history."""

    if not history:
        raise ValueError("history must contain at least one loss")
    return SmokeBenchmarkResult(
        model=model,
        initial_loss=history[0],
        final_loss=history[-1],
        steps=len(history) - 1,
    )


def _seed_for_model(seed: int, offset: int) -> int:
    return seed + offset


def run_smoke_benchmarks(
    config: SmokeBenchmarkConfig | None = None,
) -> list[SmokeBenchmarkResult]:
    """Run tiny deterministic smoke loops and return phase-ordered summaries."""

    active_config = SmokeBenchmarkConfig() if config is None else config
    results: list[SmokeBenchmarkResult] = []

    torch.manual_seed(_seed_for_model(active_config.seed, 0))
    darcy_field = DarcyPressureField(
        hidden_width=active_config.hidden_width,
        hidden_layers=active_config.hidden_layers,
    )
    darcy_history = train_darcy_smoke(
        darcy_field,
        DarcyTrainingConfig(
            interior_points_per_axis=active_config.interior_points_per_axis,
            boundary_points_per_patch=active_config.boundary_points_per_patch,
            steps=active_config.steps,
            learning_rate=active_config.learning_rate,
        ),
    )
    results.append(summarize_loss_history("darcy", darcy_history))

    torch.manual_seed(_seed_for_model(active_config.seed, 1))
    stokes_field = StokesVelocityPressureField(
        hidden_width=active_config.hidden_width,
        hidden_layers=active_config.hidden_layers,
    )
    stokes_history = train_stokes_smoke(
        stokes_field,
        StokesTrainingConfig(
            interior_points_per_axis=active_config.interior_points_per_axis,
            boundary_points_per_patch=active_config.boundary_points_per_patch,
            steps=active_config.steps,
            learning_rate=active_config.learning_rate,
            viscosity=active_config.viscosity,
        ),
    )
    results.append(summarize_loss_history("stokes", stokes_history))

    torch.manual_seed(_seed_for_model(active_config.seed, 2))
    oseen_field = StokesVelocityPressureField(
        hidden_width=active_config.hidden_width,
        hidden_layers=active_config.hidden_layers,
    )
    oseen_history = train_oseen_smoke(
        oseen_field,
        OseenTrainingConfig(
            interior_points_per_axis=active_config.interior_points_per_axis,
            boundary_points_per_patch=active_config.boundary_points_per_patch,
            steps=active_config.steps,
            learning_rate=active_config.learning_rate,
            viscosity=active_config.viscosity,
            convection_velocity=active_config.oseen_convection_velocity,
        ),
    )
    results.append(summarize_loss_history("oseen", oseen_history))

    torch.manual_seed(_seed_for_model(active_config.seed, 3))
    navier_stokes_field = StokesVelocityPressureField(
        hidden_width=active_config.hidden_width,
        hidden_layers=active_config.hidden_layers,
    )
    navier_stokes_history = train_navier_stokes_smoke(
        navier_stokes_field,
        NavierStokesTrainingConfig(
            interior_points_per_axis=active_config.interior_points_per_axis,
            boundary_points_per_patch=active_config.boundary_points_per_patch,
            steps=active_config.steps,
            learning_rate=active_config.learning_rate,
            viscosity=active_config.viscosity,
        ),
    )
    results.append(summarize_loss_history("navier_stokes", navier_stokes_history))

    return results


__all__ = [
    "SmokeBenchmarkConfig",
    "SmokeBenchmarkResult",
    "run_smoke_benchmarks",
    "summarize_loss_history",
]
