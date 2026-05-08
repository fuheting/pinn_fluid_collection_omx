"""Lightweight experiment orchestration for PINN/reference comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.models.darcy import (
    DEFAULT_DARCY_LOSS_WEIGHTS,
    DarcyPressureField,
    darcy_velocity,
    laplace_residual,
)
from pinn_fluid.models.navier_stokes import navier_stokes_residuals
from pinn_fluid.models.oseen import oseen_residuals
from pinn_fluid.models.stokes import StokesVelocityPressureField, stokes_residuals
from pinn_fluid.solvers.darcy import darcy_loss_components

COORDINATE_CONVENTION_METADATA: dict[str, object] = {
    "name": "cartesian_unit_square_y_up",
    "domain": "unit_square",
    "x_axis": "x increases left-to-right",
    "y_axis": "y=0 bottom, y=1 top",
    "inlet": "top-left horizontal segment: x in [0.0, 0.25], y=1",
    "outlet": "bottom-right horizontal segment: x in [0.75, 1.0], y=0",
    "flattening_order": "x-major with y varying fastest from bottom to top",
    "reshape_order": "reshape(grid_points, grid_points).T for plotting",
    "plot_origin": "lower",
}

DARCY_REFERENCE_METADATA: dict[str, object] = {
    "reference_generator_name": "fd_darcy_reference",
    "pde_model_represented": "Darcy pressure Laplace equation",
    "boundary_condition_type": "pressure Dirichlet inlet/outlet with no-normal-flow walls",
    "coordinate_convention": COORDINATE_CONVENTION_METADATA,
    "reference_kind": "finite-difference",
}

SHARED_PATCH_VECTOR_REFERENCE_METADATA: dict[str, object] = {
    "reference_generator_name": "shared_patch_vector_reference_from_fd_darcy",
    "pde_model_represented": "Darcy pressure Laplace equation with velocity from negative pressure gradient",
    "boundary_condition_type": "Darcy pressure Dirichlet inlet/outlet with no-normal-flow walls",
    "coordinate_convention": COORDINATE_CONVENTION_METADATA,
    "reference_kind": "demo-only",
}


@dataclass(frozen=True)
class TrainingHistory:
    """Total and component-wise objective values by optimizer iteration."""

    total: list[float]
    components: dict[str, list[float]]

    def __post_init__(self) -> None:
        if not self.total:
            raise ValueError("total history must contain at least one value")
        total_length = len(self.total)
        for name, values in self.components.items():
            if len(values) != total_length:
                raise ValueError(f"component {name!r} must have the same length as total")

    @property
    def steps(self) -> int:
        """Return the number of optimizer updates represented by this history."""

        return len(self.total) - 1

    @property
    def reduced(self) -> bool:
        """Return whether the final objective is below the initial objective."""

        return self.total[-1] < self.total[0]

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return component names in stable report order."""

        return tuple(sorted(self.components))

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable history payload."""

        return {
            "total": list(self.total),
            "components": {
                name: list(self.components[name])
                for name in self.component_names
            },
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Summary metadata for one saved PINN/reference experiment."""

    model: str
    reference: str
    grid_shape: tuple[int, int]
    history: TrainingHistory
    metrics: dict[str, float]
    artifacts: dict[str, str]
    reference_metadata: dict[str, object] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable experiment result payload."""

        return {
            "model": self.model,
            "reference": self.reference,
            "grid_shape": list(self.grid_shape),
            "history": self.history.to_json_dict(),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "reference_metadata": deepcopy(self.reference_metadata),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by deterministic local experiment runs."""

    output_dir: Path | str = Path("data/experiments")
    grid_points: int = 9
    training_steps: int = 40
    hidden_width: int = 12
    hidden_layers: int = 1
    learning_rate: float = 0.02
    seed: int = 0
    viscosity: float = 0.25
    peak_velocity: float = 1.0
    darcy_reference_iterations: int = 400


def _as_output_dir(path: Path | str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _relative(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _grid_coordinates(grid_points: int) -> torch.Tensor:
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    values = torch.linspace(0.0, 1.0, grid_points)
    grid_x, grid_y = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)


def _weighted_total(
    components: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    return sum(weights[name] * components[name] for name in components)


def _record(
    total: torch.Tensor,
    components: dict[str, torch.Tensor],
    totals: list[float],
    component_history: dict[str, list[float]],
) -> None:
    totals.append(float(total.detach()))
    for name, value in components.items():
        component_history.setdefault(name, []).append(float(value.detach()))


def _train_with_history(
    model: torch.nn.Module,
    steps: int,
    learning_rate: float,
    loss_builder: Callable[[], dict[str, torch.Tensor]],
    weights: dict[str, float],
) -> TrainingHistory:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    totals: list[float] = []
    component_history: dict[str, list[float]] = {}

    components = loss_builder()
    _record(_weighted_total(components, weights), components, totals, component_history)
    for _ in range(steps):
        optimizer.zero_grad()
        components = loss_builder()
        total = _weighted_total(components, weights)
        total.backward()
        optimizer.step()
        components = loss_builder()
        _record(_weighted_total(components, weights), components, totals, component_history)

    return TrainingHistory(total=totals, components=component_history)


def _save_history(history: TrainingHistory, path: Path) -> None:
    path.write_text(json.dumps(history.to_json_dict(), indent=2, sort_keys=True))


def _save_metrics(metrics: dict[str, float], path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def _plot_history(history: TrainingHistory, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5, 3))
    axis.plot(history.total, label="total", linewidth=2)
    for name in history.component_names:
        axis.plot(history.components[name], label=name, linewidth=1)
    axis.set_xlabel("iteration")
    axis.set_ylabel("loss")
    axis.set_yscale("log")
    axis.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_scalar_field(values: np.ndarray, grid_points: int, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(4, 3))
    image = axis.imshow(values.reshape(grid_points, grid_points).T, origin="lower", extent=(0, 1, 0, 1))
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _velocity_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


def _pressure_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


def _fd_darcy_reference(grid_points: int, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.zeros((grid_points, grid_points), dtype=np.float64)
    top_inlet_end = max(1, int(round(0.25 * (grid_points - 1))))
    bottom_outlet_start = int(round(0.75 * (grid_points - 1)))

    for _ in range(iterations):
        updated = pressure.copy()
        updated[1:-1, 1:-1] = 0.25 * (
            pressure[:-2, 1:-1]
            + pressure[2:, 1:-1]
            + pressure[1:-1, :-2]
            + pressure[1:-1, 2:]
        )
        updated[0, :] = updated[1, :]
        updated[-1, :] = updated[-2, :]
        updated[:, 0] = updated[:, 1]
        updated[:, -1] = updated[:, -2]
        updated[: top_inlet_end + 1, -1] = 1.0
        updated[bottom_outlet_start:, 0] = 0.0
        pressure = updated

    spacing = 1.0 / (grid_points - 1)
    dp_dx, dp_dy = np.gradient(pressure, spacing, spacing, edge_order=1)
    velocity = -np.stack((dp_dx.reshape(-1), dp_dy.reshape(-1)), axis=1)
    return pressure.reshape(-1, 1), velocity


def _laplace_residual_grid(pressure: np.ndarray) -> np.ndarray:
    grid_points = pressure.shape[0]
    spacing = 1.0 / (grid_points - 1)
    residual = np.zeros_like(pressure, dtype=np.float64)
    residual[1:-1, 1:-1] = (
        pressure[:-2, 1:-1]
        + pressure[2:, 1:-1]
        + pressure[1:-1, :-2]
        + pressure[1:-1, 2:]
        - 4.0 * pressure[1:-1, 1:-1]
    ) / (spacing**2)
    return residual.reshape(-1, 1)


def _fd_darcy_reference_fields(grid_points: int, iterations: int) -> dict[str, np.ndarray]:
    """Return named Darcy finite-difference reference fields and diagnostics."""

    pressure, velocity = _fd_darcy_reference(grid_points, iterations)
    pressure_grid = pressure.reshape(grid_points, grid_points)
    return {
        "pressure": pressure,
        "velocity": velocity,
        "u": velocity[:, :1],
        "v": velocity[:, 1:2],
        "speed": np.linalg.norm(velocity, axis=1, keepdims=True),
        "residual": _laplace_residual_grid(pressure_grid),
    }


def _reference_metadata_json(metadata: dict[str, object]) -> str:
    return json.dumps(metadata, sort_keys=True)


def _shared_patch_vector_reference(
    grid_points: int,
    iterations: int,
) -> dict[str, np.ndarray]:
    reference_pressure, reference_velocity = _fd_darcy_reference(grid_points, iterations)
    return {
        "u": reference_velocity[:, :1],
        "v": reference_velocity[:, 1:2],
        "pressure": reference_pressure,
        "velocity": reference_velocity,
    }


def run_darcy_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    """Run the deterministic Darcy PINN/reference vertical-slice experiment."""

    active = ExperimentConfig() if config is None else config
    output_dir = _as_output_dir(active.output_dir)
    model_dir = output_dir / "darcy"
    model_dir.mkdir(exist_ok=True)

    torch.manual_seed(active.seed)
    model = DarcyPressureField(hidden_width=active.hidden_width, hidden_layers=active.hidden_layers)
    interior = interior_collocation_points(max(2, active.grid_points - 2))
    boundary = boundary_collocation_points(max(2, active.grid_points // 2))

    history = _train_with_history(
        model,
        active.training_steps,
        active.learning_rate,
        lambda: darcy_loss_components(model, interior, boundary),
        DEFAULT_DARCY_LOSS_WEIGHTS,
    )

    coordinates = _grid_coordinates(active.grid_points)
    autograd_coordinates = coordinates.detach().clone().requires_grad_(True)
    predicted_pressure = model(autograd_coordinates)
    predicted_velocity = darcy_velocity(predicted_pressure, autograd_coordinates)
    residual = laplace_residual(predicted_pressure, autograd_coordinates)
    reference_fields = _fd_darcy_reference_fields(
        active.grid_points,
        active.darcy_reference_iterations,
    )
    reference_metadata = deepcopy(DARCY_REFERENCE_METADATA)

    predicted_pressure_np = predicted_pressure.detach().numpy()
    predicted_velocity_np = predicted_velocity.detach().numpy()
    predicted_speed_np = np.linalg.norm(predicted_velocity_np, axis=1, keepdims=True)
    residual_np = residual.detach().numpy()
    metrics = {
        "pressure_l2": _pressure_l2(predicted_pressure_np, reference_fields["pressure"]),
        "velocity_l2": _velocity_l2(predicted_velocity_np, reference_fields["velocity"]),
        "residual_rms": float(np.sqrt(np.mean(residual_np**2))),
        "reference_residual_rms": float(np.sqrt(np.mean(reference_fields["residual"] ** 2))),
    }

    fields_path = model_dir / "fields.npz"
    history_path = model_dir / "history.json"
    metrics_path = model_dir / "metrics.json"
    loss_plot_path = model_dir / "loss_history.png"
    field_plot_path = model_dir / "predicted_pressure.png"
    np.savez(
        fields_path,
        coordinates=coordinates.numpy(),
        predicted_pressure=predicted_pressure_np,
        reference_pressure=reference_fields["pressure"],
        predicted_velocity=predicted_velocity_np,
        reference_velocity=reference_fields["velocity"],
        predicted_u=predicted_velocity_np[:, :1],
        predicted_v=predicted_velocity_np[:, 1:2],
        predicted_speed=predicted_speed_np,
        reference_u=reference_fields["u"],
        reference_v=reference_fields["v"],
        reference_speed=reference_fields["speed"],
        reference_residual=reference_fields["residual"],
        residual=residual_np,
        reference_metadata_json=_reference_metadata_json(reference_metadata),
    )
    _save_history(history, history_path)
    _save_metrics(metrics, metrics_path)
    _plot_history(history, loss_plot_path)
    _plot_scalar_field(predicted_pressure_np, active.grid_points, "Darcy predicted pressure", field_plot_path)

    return ExperimentResult(
        model="darcy",
        reference="finite_difference_laplace",
        grid_shape=(active.grid_points, active.grid_points),
        history=history,
        metrics=metrics,
        artifacts={
            "fields_npz": _relative(fields_path, output_dir),
            "history_json": _relative(history_path, output_dir),
            "metrics_json": _relative(metrics_path, output_dir),
            "loss_plot_png": _relative(loss_plot_path, output_dir),
            "field_plot_png": _relative(field_plot_path, output_dir),
        },
        reference_metadata=reference_metadata,
    )


def _vector_loss_components(
    model: torch.nn.Module,
    residual_builder: Callable[[dict[str, torch.Tensor], torch.Tensor], dict[str, torch.Tensor]],
    interior: torch.Tensor,
    boundary_samples: dict[str, dict[str, torch.Tensor]],
    *,
    peak_velocity: float,
) -> dict[str, torch.Tensor]:
    interior_autograd = interior.detach().clone().requires_grad_(True)
    outputs = model(interior_autograd)
    residuals = residual_builder(outputs, interior_autograd)

    inlet_coordinates = boundary_samples["inlet"]["coordinates"]
    inlet_outputs = model(inlet_coordinates)
    inlet_target = torch.tensor(
        (0.0, -peak_velocity),
        dtype=inlet_coordinates.dtype,
        device=inlet_coordinates.device,
    ).reshape(1, 2)

    outlet_outputs = model(boundary_samples["outlet"]["coordinates"])
    wall_outputs = model(boundary_samples["walls"]["coordinates"])
    wall_velocity = torch.cat(
        (wall_outputs["u"], wall_outputs["v"]),
        dim=1,
    )
    inlet_velocity = torch.cat(
        (inlet_outputs["u"], inlet_outputs["v"]),
        dim=1,
    )

    return {
        "continuity": residuals["continuity"].square().mean(),
        "x_momentum": residuals["x_momentum"].square().mean(),
        "y_momentum": residuals["y_momentum"].square().mean(),
        "inlet": (inlet_velocity - inlet_target).square().mean(),
        "outlet": outlet_outputs["pressure"].square().mean(),
        "wall": wall_velocity.square().mean(),
    }


def _run_shared_patch_vector_experiment(
    *,
    model_name: str,
    reference: str,
    config: ExperimentConfig,
    residual_builder: Callable[[dict[str, torch.Tensor], torch.Tensor], dict[str, torch.Tensor]],
) -> ExperimentResult:
    output_dir = _as_output_dir(config.output_dir)
    model_dir = output_dir / model_name
    model_dir.mkdir(exist_ok=True)

    torch.manual_seed(config.seed)
    model = StokesVelocityPressureField(
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    )
    interior = interior_collocation_points(max(2, config.grid_points - 2))
    boundary = boundary_collocation_points(max(2, config.grid_points // 2))
    weights = {
        "continuity": 1.0,
        "x_momentum": 1.0,
        "y_momentum": 1.0,
        "inlet": 10.0,
        "outlet": 10.0,
        "wall": 10.0,
    }
    history = _train_with_history(
        model,
        config.training_steps,
        config.learning_rate,
        lambda: _vector_loss_components(
            model,
            residual_builder,
            interior,
            boundary,
            peak_velocity=config.peak_velocity,
        ),
        weights,
    )

    coordinates = _grid_coordinates(config.grid_points)
    autograd_coordinates = coordinates.detach().clone().requires_grad_(True)
    predicted = model(autograd_coordinates)
    reference_fields = _shared_patch_vector_reference(
        config.grid_points,
        config.darcy_reference_iterations,
    )
    reference_metadata = deepcopy(SHARED_PATCH_VECTOR_REFERENCE_METADATA)
    residuals = residual_builder(predicted, autograd_coordinates)
    residual_np = np.sqrt(
        sum(value.detach().numpy() ** 2 for value in residuals.values())
    )
    predicted_velocity = torch.cat((predicted["u"], predicted["v"]), dim=1).detach().numpy()
    reference_velocity = reference_fields["velocity"]
    predicted_pressure = predicted["pressure"].detach().numpy()
    reference_pressure = reference_fields["pressure"]
    metrics = {
        "pressure_l2": _pressure_l2(predicted_pressure, reference_pressure),
        "velocity_l2": _velocity_l2(predicted_velocity, reference_velocity),
        "residual_rms": float(np.sqrt(np.mean(residual_np**2))),
    }

    fields_path = model_dir / "fields.npz"
    history_path = model_dir / "history.json"
    metrics_path = model_dir / "metrics.json"
    loss_plot_path = model_dir / "loss_history.png"
    field_plot_path = model_dir / "predicted_u.png"
    np.savez(
        fields_path,
        coordinates=coordinates.numpy(),
        predicted_u=predicted["u"].detach().numpy(),
        predicted_v=predicted["v"].detach().numpy(),
        predicted_pressure=predicted_pressure,
        reference_u=reference_fields["u"],
        reference_v=reference_fields["v"],
        reference_pressure=reference_pressure,
        residual=residual_np,
        reference_metadata_json=_reference_metadata_json(reference_metadata),
    )
    _save_history(history, history_path)
    _save_metrics(metrics, metrics_path)
    _plot_history(history, loss_plot_path)
    _plot_scalar_field(predicted["u"].detach().numpy(), config.grid_points, f"{model_name} predicted u", field_plot_path)

    return ExperimentResult(
        model=model_name,
        reference=reference,
        grid_shape=(config.grid_points, config.grid_points),
        history=history,
        metrics=metrics,
        artifacts={
            "fields_npz": _relative(fields_path, output_dir),
            "history_json": _relative(history_path, output_dir),
            "metrics_json": _relative(metrics_path, output_dir),
            "loss_plot_png": _relative(loss_plot_path, output_dir),
            "field_plot_png": _relative(field_plot_path, output_dir),
        },
        reference_metadata=reference_metadata,
    )


def run_poiseuille_navier_stokes_experiment(
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run the Navier-Stokes PINN/reference vertical-slice experiment."""

    active = ExperimentConfig() if config is None else config
    return _run_shared_patch_vector_experiment(
        model_name="navier_stokes",
        reference="shared_patch_unit_square_reference",
        config=active,
        residual_builder=lambda outputs, coordinates: navier_stokes_residuals(
            outputs["u"],
            outputs["v"],
            outputs["pressure"],
            coordinates,
            viscosity=active.viscosity,
        ),
    )


def run_stokes_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    """Run the Stokes extension against the shared unit-square patch reference."""

    active = ExperimentConfig() if config is None else config
    return _run_shared_patch_vector_experiment(
        model_name="stokes",
        reference="shared_patch_unit_square_reference",
        config=active,
        residual_builder=lambda outputs, coordinates: stokes_residuals(
            outputs["u"],
            outputs["v"],
            outputs["pressure"],
            coordinates,
            viscosity=active.viscosity,
        ),
    )


def run_oseen_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    """Run the Oseen extension against the shared unit-square patch reference."""

    active = ExperimentConfig() if config is None else config

    def residual_builder(
        outputs: dict[str, torch.Tensor],
        coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        convection = torch.tensor(
            (active.peak_velocity, 0.0),
            dtype=coordinates.dtype,
            device=coordinates.device,
        ).reshape(1, 2).expand(coordinates.shape[0], 2)
        return oseen_residuals(
            outputs["u"],
            outputs["v"],
            outputs["pressure"],
            coordinates,
            convection,
            viscosity=active.viscosity,
        )

    return _run_shared_patch_vector_experiment(
        model_name="oseen",
        reference="shared_patch_unit_square_reference",
        config=active,
        residual_builder=residual_builder,
    )


def _write_cross_model_report(results: list[ExperimentResult], output_dir: Path) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    payload = {"results": [result.to_json_dict() for result in results]}
    (summary_dir / "cross_model_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )

    lines = [
        "# Cross-Model Experiment Report",
        "",
        "| Model | Reference | Final objective | Velocity L2 | Pressure L2 | Residual RMS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.model} | {result.reference} | {result.history.total[-1]:.6g} | "
            f"{result.metrics['velocity_l2']:.6g} | {result.metrics['pressure_l2']:.6g} | "
            f"{result.metrics['residual_rms']:.6g} |"
        )
    lines.append("")
    (summary_dir / "cross_model_report.md").write_text("\n".join(lines))


def run_all_experiments(config: ExperimentConfig | None = None) -> list[ExperimentResult]:
    """Run Darcy, Stokes, Oseen, and Navier-Stokes experiments and write a report."""

    active = ExperimentConfig() if config is None else config
    output_dir = _as_output_dir(active.output_dir)
    results = [
        run_darcy_experiment(active),
        run_stokes_experiment(active),
        run_oseen_experiment(active),
        run_poiseuille_navier_stokes_experiment(active),
    ]
    _write_cross_model_report(results, output_dir)
    return results


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "TrainingHistory",
    "COORDINATE_CONVENTION_METADATA",
    "DARCY_REFERENCE_METADATA",
    "SHARED_PATCH_VECTOR_REFERENCE_METADATA",
    "_fd_darcy_reference_fields",
    "run_all_experiments",
    "run_darcy_experiment",
    "run_oseen_experiment",
    "run_poiseuille_navier_stokes_experiment",
    "run_stokes_experiment",
]
