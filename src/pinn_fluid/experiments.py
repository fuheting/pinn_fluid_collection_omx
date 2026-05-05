"""Lightweight experiment orchestration for PINN/reference comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Callable
import zlib

import numpy as np
import torch

from pinn_fluid.domains import boundary_collocation_points, interior_collocation_points
from pinn_fluid.models.darcy import (
    DEFAULT_DARCY_LOSS_WEIGHTS,
    DarcyPressureField,
    darcy_velocity,
    laplace_residual,
)
from pinn_fluid.models.navier_stokes import navier_stokes_residuals, poiseuille_channel_solution
from pinn_fluid.models.oseen import oseen_residuals
from pinn_fluid.models.stokes import StokesVelocityPressureField, stokes_residuals
from pinn_fluid.solvers.darcy import darcy_loss_components


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

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable experiment result payload."""

        return {
            "model": self.model,
            "reference": self.reference,
            "grid_shape": list(self.grid_shape),
            "history": self.history.to_json_dict(),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
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


def _channel_boundary_coordinates(grid_points: int) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, grid_points)
    left = torch.stack((torch.zeros_like(values), values), dim=1)
    right = torch.stack((torch.ones_like(values), values), dim=1)
    bottom = torch.stack((values, torch.zeros_like(values)), dim=1)
    top = torch.stack((values, torch.ones_like(values)), dim=1)
    return torch.cat((left, right, bottom, top), dim=0)


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


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape
    rows = [b"\x00" + image[row].astype(np.uint8).tobytes() for row in range(height)]
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk("IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk("IDAT".encode(), zlib.compress(raw))
    payload += chunk("IEND".encode(), b"")
    path.write_bytes(payload)


def _draw_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _fallback_history_png(history: TrainingHistory, path: Path) -> None:
    image = np.full((220, 360, 3), 255, dtype=np.uint8)
    series = [history.total] + [history.components[name] for name in history.component_names]
    values = np.array([value for item in series for value in item], dtype=np.float64)
    values = np.log10(np.clip(values, 1e-12, None))
    low = float(values.min())
    high = float(values.max())
    span = high - low if high > low else 1.0
    colors = [(0, 0, 0), (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    for index, item in enumerate(series):
        log_values = np.log10(np.clip(np.array(item, dtype=np.float64), 1e-12, None))
        points = []
        for step, value in enumerate(log_values):
            x = 30 + round(step * 300 / max(1, len(log_values) - 1))
            y = 190 - round((float(value) - low) * 160 / span)
            points.append((x, y))
        for start, end in zip(points, points[1:]):
            _draw_line(image, start, end, colors[index % len(colors)])
    _write_rgb_png(path, image)


def _fallback_scalar_png(values: np.ndarray, grid_points: int, path: Path) -> None:
    field = values.reshape(grid_points, grid_points).T
    low = float(np.min(field))
    high = float(np.max(field))
    span = high - low if high > low else 1.0
    normalized = ((field - low) / span * 255.0).astype(np.uint8)
    image = np.stack((normalized, 255 - normalized, np.full_like(normalized, 128)), axis=2)
    image = np.repeat(np.repeat(image, 18, axis=0), 18, axis=1)
    _write_rgb_png(path, image)


def _plot_history(history: TrainingHistory, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_history_png(history, path)
        return

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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_scalar_png(values, grid_points, path)
        return

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
    reference_pressure, reference_velocity = _fd_darcy_reference(
        active.grid_points,
        active.darcy_reference_iterations,
    )

    predicted_pressure_np = predicted_pressure.detach().numpy()
    predicted_velocity_np = predicted_velocity.detach().numpy()
    residual_np = residual.detach().numpy()
    metrics = {
        "pressure_l2": _pressure_l2(predicted_pressure_np, reference_pressure),
        "velocity_l2": _velocity_l2(predicted_velocity_np, reference_velocity),
        "residual_rms": float(np.sqrt(np.mean(residual_np**2))),
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
        reference_pressure=reference_pressure,
        predicted_velocity=predicted_velocity_np,
        reference_velocity=reference_velocity,
        residual=residual_np,
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
    )


def _vector_loss_components(
    model: torch.nn.Module,
    residual_builder: Callable[[dict[str, torch.Tensor], torch.Tensor], dict[str, torch.Tensor]],
    interior: torch.Tensor,
    boundary: torch.Tensor,
    *,
    viscosity: float,
    peak_velocity: float,
) -> dict[str, torch.Tensor]:
    interior_autograd = interior.detach().clone().requires_grad_(True)
    outputs = model(interior_autograd)
    residuals = residual_builder(outputs, interior_autograd)
    boundary_outputs = model(boundary)
    boundary_reference = poiseuille_channel_solution(
        boundary,
        viscosity=viscosity,
        peak_velocity=peak_velocity,
    )
    predicted_boundary = torch.cat(
        (boundary_outputs["u"], boundary_outputs["v"], boundary_outputs["pressure"]),
        dim=1,
    )
    reference_boundary = torch.cat(
        (
            boundary_reference["u"],
            boundary_reference["v"],
            boundary_reference["pressure"],
        ),
        dim=1,
    )
    return {
        "continuity": residuals["continuity"].square().mean(),
        "x_momentum": residuals["x_momentum"].square().mean(),
        "y_momentum": residuals["y_momentum"].square().mean(),
        "boundary": (predicted_boundary - reference_boundary).square().mean(),
    }


def _run_poiseuille_vector_experiment(
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
    boundary = _channel_boundary_coordinates(config.grid_points)
    weights = {
        "continuity": 1.0,
        "x_momentum": 1.0,
        "y_momentum": 1.0,
        "boundary": 10.0,
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
            viscosity=config.viscosity,
            peak_velocity=config.peak_velocity,
        ),
        weights,
    )

    coordinates = _grid_coordinates(config.grid_points)
    autograd_coordinates = coordinates.detach().clone().requires_grad_(True)
    predicted = model(autograd_coordinates)
    reference_fields = poiseuille_channel_solution(
        autograd_coordinates,
        viscosity=config.viscosity,
        peak_velocity=config.peak_velocity,
    )
    residuals = residual_builder(predicted, autograd_coordinates)
    residual_np = np.sqrt(
        sum(value.detach().numpy() ** 2 for value in residuals.values())
    )
    predicted_velocity = torch.cat((predicted["u"], predicted["v"]), dim=1).detach().numpy()
    reference_velocity = torch.cat(
        (reference_fields["u"], reference_fields["v"]),
        dim=1,
    ).detach().numpy()
    predicted_pressure = predicted["pressure"].detach().numpy()
    reference_pressure = reference_fields["pressure"].detach().numpy()
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
        reference_u=reference_fields["u"].detach().numpy(),
        reference_v=reference_fields["v"].detach().numpy(),
        reference_pressure=reference_pressure,
        residual=residual_np,
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
    )


def run_poiseuille_navier_stokes_experiment(
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run the Poiseuille/Navier-Stokes PINN/reference vertical-slice experiment."""

    active = ExperimentConfig() if config is None else config
    return _run_poiseuille_vector_experiment(
        model_name="navier_stokes",
        reference="poiseuille_channel_reference",
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
    """Run the Stokes extension against the Poiseuille channel reference."""

    active = ExperimentConfig() if config is None else config
    return _run_poiseuille_vector_experiment(
        model_name="stokes",
        reference="poiseuille_channel_reference",
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
    """Run the Oseen extension against the Poiseuille channel reference."""

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

    return _run_poiseuille_vector_experiment(
        model_name="oseen",
        reference="poiseuille_channel_reference",
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
    "run_all_experiments",
    "run_darcy_experiment",
    "run_oseen_experiment",
    "run_poiseuille_navier_stokes_experiment",
    "run_stokes_experiment",
]
