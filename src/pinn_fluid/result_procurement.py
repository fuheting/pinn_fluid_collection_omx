"""Narrow result-procurement runner for saved experiment artifacts."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

from pinn_fluid.experiments import ExperimentConfig, ExperimentResult, run_all_experiments


def _config_payload(config: ExperimentConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        payload[field.name] = str(value) if isinstance(value, Path) else value
    return payload


def _git_commit(*, work_tree: Path | None = None, git_dir: Path | None = None) -> str:
    command = ["git"]
    if git_dir is not None:
        command.append(f"--git-dir={git_dir}")
    if work_tree is not None:
        command.append(f"--work-tree={work_tree}")
    command.extend(("rev-parse", "HEAD"))
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"could not resolve git commit: {detail}") from exc
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("could not resolve git commit: empty rev-parse output")
    return commit


def _result_payload(result: ExperimentResult, output_dir: Path) -> dict[str, Any]:
    metrics_finite = all(math.isfinite(value) for value in result.metrics.values())
    model_output_dir = output_dir / result.model
    return {
        "model": result.model,
        "reference": result.reference,
        "output_dir": str(model_output_dir),
        "grid_shape": list(result.grid_shape),
        "history_decreased": result.history.reduced,
        "metrics_finite": metrics_finite,
        "history": {
            "initial_total": result.history.total[0],
            "final_total": result.history.total[-1],
            "steps": result.history.steps,
            "component_names": list(result.history.component_names),
        },
        "metrics": dict(result.metrics),
        "artifacts": dict(result.artifacts),
    }


def _manifest_payload(
    *,
    config: ExperimentConfig,
    results: list[ExperimentResult],
    git_commit: str,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    result_entries = [_result_payload(result, output_dir) for result in results]
    return {
        "git_commit": git_commit,
        "output_dir": str(output_dir),
        "config": _config_payload(config),
        "all_metrics_finite": all(entry["metrics_finite"] for entry in result_entries),
        "all_histories_decreased": all(entry["history_decreased"] for entry in result_entries),
        "results": result_entries,
    }


def run_result_procurement(
    config: ExperimentConfig | None = None,
    *,
    git_commit: str | None = None,
    git_dir: Path | str | None = None,
    work_tree: Path | str | None = None,
) -> dict[str, Any]:
    """Run all current experiments and write a procurement manifest."""

    active = ExperimentConfig() if config is None else config
    output_dir = Path(active.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_commit = git_commit or _git_commit(
        git_dir=Path(git_dir) if git_dir is not None else None,
        work_tree=Path(work_tree) if work_tree is not None else None,
    )
    results = run_all_experiments(active)
    manifest = _manifest_payload(
        config=active,
        results=results,
        git_commit=resolved_commit,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Run PINN fluid experiments and write run_manifest.json.",
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--grid-points", type=int, default=defaults.grid_points)
    parser.add_argument("--training-steps", type=int, default=defaults.training_steps)
    parser.add_argument("--hidden-width", type=int, default=defaults.hidden_width)
    parser.add_argument("--hidden-layers", type=int, default=defaults.hidden_layers)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--viscosity", type=float, default=defaults.viscosity)
    parser.add_argument("--peak-velocity", type=float, default=defaults.peak_velocity)
    parser.add_argument(
        "--darcy-reference-iterations",
        type=int,
        default=defaults.darcy_reference_iterations,
    )
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--git-dir", default=None)
    parser.add_argument("--work-tree", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for Phase 11 result procurement."""

    args = _build_parser().parse_args(argv)
    config = ExperimentConfig(
        output_dir=args.output_dir,
        grid_points=args.grid_points,
        training_steps=args.training_steps,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        seed=args.seed,
        viscosity=args.viscosity,
        peak_velocity=args.peak_velocity,
        darcy_reference_iterations=args.darcy_reference_iterations,
    )
    manifest = run_result_procurement(
        config,
        git_commit=args.git_commit,
        git_dir=args.git_dir,
        work_tree=args.work_tree,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_result_procurement"]
