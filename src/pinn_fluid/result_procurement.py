"""Narrow result-procurement runner for saved experiment artifacts."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

from pinn_fluid.experiments import ExperimentConfig, ExperimentResult, run_all_experiments

FLOW_MODELS = ("darcy", "stokes", "oseen", "navier_stokes")


def _config_payload(config: ExperimentConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(config):
        if field.name in {
            "vector_reference_fields",
            "vector_reference_metadata",
            "model_reference_fields",
            "model_reference_metadata",
        }:
            continue
        value = getattr(config, field.name)
        if isinstance(value, Path):
            payload[field.name] = str(value)
        elif isinstance(value, dict):
            payload[field.name] = {
                str(key): str(item) if isinstance(item, Path) else item
                for key, item in value.items()
            }
        else:
            payload[field.name] = value
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
        "reference_metadata": dict(result.reference_metadata),
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
    reference_generators = {
        result.model: str(result.reference_metadata["reference_generator_name"])
        for result in results
    }
    return {
        "git_commit": git_commit,
        "output_dir": str(output_dir),
        "config": _config_payload(config),
        "reference_generators": reference_generators,
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
    active = _prepare_vector_reference_config(active)
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


def _run_openfoam_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{command[0]} was not found; install OpenFOAM before using OpenFOAM references"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"OpenFOAM command failed ({' '.join(command)}): {detail}") from exc


def _solve_openfoam_reference(active: ExperimentConfig, *, model_name: str) -> Path:
    from pinn_fluid.dedicated_solvers import OpenFOAMCaseConfig, write_openfoam_shared_domain_case

    if active.openfoam_case_dir is None:
        case_dir = Path(active.output_dir) / "openfoam_cases" / model_name
    else:
        case_dir = Path(active.openfoam_case_dir) / model_name
    write_openfoam_shared_domain_case(
        case_dir,
        OpenFOAMCaseConfig(
            grid_points=active.grid_points,
            viscosity=active.viscosity,
            inlet_velocity=active.peak_velocity,
            end_time=active.openfoam_end_time,
        ),
    )
    _run_openfoam_command(["blockMesh", "-case", str(case_dir)])
    _run_openfoam_command(["foamRun", "-solver", "incompressibleFluid", "-case", str(case_dir)])
    _run_openfoam_command(["postProcess", "-func", "sampleDict", "-latestTime", "-case", str(case_dir)])
    sample_path = (
        case_dir
        / "postProcessing"
        / "sampleDict"
        / str(active.openfoam_end_time)
        / "sharedDomainGrid.xy"
    )
    if not sample_path.is_file():
        candidates = sorted((case_dir / "postProcessing" / "sampleDict").glob("*/sharedDomainGrid.xy"))
        if not candidates:
            raise RuntimeError(f"OpenFOAM sample was not written under {case_dir}")
        sample_path = candidates[-1]
    return sample_path


def _solve_fenicsx_reference(active: ExperimentConfig, *, model_name: str) -> Path:
    from pinn_fluid.fenicsx_solvers import generate_fenicsx_reference

    return generate_fenicsx_reference(
        model_name=model_name,
        output_dir=Path(active.output_dir) / "fenicsx_references",
        grid_points=active.grid_points,
        mesh_cells=max(4, active.grid_points - 1),
        viscosity=active.viscosity,
        peak_velocity=active.peak_velocity,
    )


def _required_openfoam_sample_paths(active: ExperimentConfig) -> dict[str, Path]:
    if active.openfoam_reference_sample_path is not None:
        raise ValueError(
            "OpenFOAM ground truth must use per-model OpenFOAM sample paths; "
            "set openfoam_reference_sample_paths instead of openfoam_reference_sample_path"
        )
    if active.openfoam_reference_sample_paths is None:
        return {
            model_name: _solve_openfoam_reference(active, model_name=model_name)
            for model_name in FLOW_MODELS
        }

    paths = {
        model_name: Path(path)
        for model_name, path in active.openfoam_reference_sample_paths.items()
    }
    missing = sorted(set(FLOW_MODELS) - set(paths))
    extra = sorted(set(paths) - set(FLOW_MODELS))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("OpenFOAM sample paths must cover exactly the flow models: " + ", ".join(details))
    return paths


def _required_fenicsx_sample_paths(active: ExperimentConfig) -> dict[str, Path]:
    if active.openfoam_reference_sample_path is not None or active.openfoam_reference_sample_paths is not None:
        raise ValueError("FEniCSx references do not accept OpenFOAM sample paths")
    if active.fenicsx_reference_sample_paths is None:
        return {
            model_name: _solve_fenicsx_reference(active, model_name=model_name)
            for model_name in FLOW_MODELS
        }

    paths = {
        model_name: Path(path)
        for model_name, path in active.fenicsx_reference_sample_paths.items()
    }
    missing = sorted(set(FLOW_MODELS) - set(paths))
    extra = sorted(set(paths) - set(FLOW_MODELS))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("FEniCSx sample paths must cover exactly the flow models: " + ", ".join(details))
    return paths


def _prepare_vector_reference_config(active: ExperimentConfig) -> ExperimentConfig:
    if active.vector_reference_source != "fenicsx":
        raise ValueError("vector_reference_source must be 'fenicsx'")

    from pinn_fluid.fenicsx_solvers import (
        FENICSX_REFERENCE_METADATA,
        FLOW_MODEL_METADATA,
        import_fenicsx_sampled_fields,
    )

    sample_paths = _required_fenicsx_sample_paths(active)
    model_fields: dict[str, dict[str, Any]] = {}
    model_metadata: dict[str, dict[str, object]] = {}
    for model_name, sample_path in sample_paths.items():
        metadata = dict(FENICSX_REFERENCE_METADATA)
        metadata.update(FLOW_MODEL_METADATA[model_name])
        metadata["sample_path"] = str(sample_path)
        metadata["fenicsx_reference_model"] = model_name
        imported = import_fenicsx_sampled_fields(
            sample_path,
            grid_points=active.grid_points,
            metadata=metadata,
        )
        model_fields[model_name] = imported
        model_metadata[model_name] = metadata
    return replace(
        active,
        fenicsx_reference_sample_paths=sample_paths,
        model_reference_fields=model_fields,
        model_reference_metadata=model_metadata,
    )


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
    parser.add_argument(
        "--vector-reference-source",
        choices=("fenicsx",),
        default=defaults.vector_reference_source,
    )
    parser.add_argument("--openfoam-reference-sample-path", default=None)
    parser.add_argument("--darcy-openfoam-reference-sample-path", default=None)
    parser.add_argument("--stokes-openfoam-reference-sample-path", default=None)
    parser.add_argument("--oseen-openfoam-reference-sample-path", default=None)
    parser.add_argument("--navier-stokes-openfoam-reference-sample-path", default=None)
    parser.add_argument("--openfoam-case-dir", default=None)
    parser.add_argument("--openfoam-end-time", type=int, default=defaults.openfoam_end_time)
    parser.add_argument("--darcy-fenicsx-reference-sample-path", default=None)
    parser.add_argument("--stokes-fenicsx-reference-sample-path", default=None)
    parser.add_argument("--oseen-fenicsx-reference-sample-path", default=None)
    parser.add_argument("--navier-stokes-fenicsx-reference-sample-path", default=None)
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
        vector_reference_source=args.vector_reference_source,
        openfoam_reference_sample_path=args.openfoam_reference_sample_path,
        openfoam_reference_sample_paths={
            "darcy": args.darcy_openfoam_reference_sample_path,
            "stokes": args.stokes_openfoam_reference_sample_path,
            "oseen": args.oseen_openfoam_reference_sample_path,
            "navier_stokes": args.navier_stokes_openfoam_reference_sample_path,
        }
        if all(
            path is not None
            for path in (
                args.darcy_openfoam_reference_sample_path,
                args.stokes_openfoam_reference_sample_path,
                args.oseen_openfoam_reference_sample_path,
                args.navier_stokes_openfoam_reference_sample_path,
            )
        )
        else None,
        openfoam_case_dir=args.openfoam_case_dir,
        openfoam_end_time=args.openfoam_end_time,
        fenicsx_reference_sample_paths={
            "darcy": args.darcy_fenicsx_reference_sample_path,
            "stokes": args.stokes_fenicsx_reference_sample_path,
            "oseen": args.oseen_fenicsx_reference_sample_path,
            "navier_stokes": args.navier_stokes_fenicsx_reference_sample_path,
        }
        if all(
            path is not None
            for path in (
                args.darcy_fenicsx_reference_sample_path,
                args.stokes_fenicsx_reference_sample_path,
                args.oseen_fenicsx_reference_sample_path,
                args.navier_stokes_fenicsx_reference_sample_path,
            )
        )
        else None,
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
