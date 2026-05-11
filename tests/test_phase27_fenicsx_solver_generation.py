"""Phase 27 tests for generated FEniCSx model-specific references."""

from __future__ import annotations

import subprocess

import numpy as np

from pinn_fluid.experiments import ExperimentConfig
from pinn_fluid.fenicsx_solvers import FLOW_MODEL_METADATA, generate_fenicsx_reference
from pinn_fluid.result_procurement import run_result_procurement


def _write_solver_artifact(path, *, model_name: str, grid_points: int) -> None:
    offsets = {
        "darcy": 0.0,
        "stokes": 1.0,
        "oseen": 2.0,
        "navier_stokes": 3.0,
    }
    offset = offsets[model_name]
    values = np.linspace(0.0, 1.0, grid_points)
    coordinates = []
    pressure = []
    u = []
    v = []
    for x in values:
        for y in values:
            coordinates.append((x, y))
            pressure.append((offset + 2.0 - 1.5 * y + 0.25 * x,))
            u.append((0.05 * offset + x * (1.0 - x) * (0.1 + 0.03 * offset),))
            v.append((-0.1 * offset - y * (1.0 - 0.2 * x),))
    np.savez(
        path,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        reference_pressure=np.asarray(pressure, dtype=np.float64),
        reference_u=np.asarray(u, dtype=np.float64),
        reference_v=np.asarray(v, dtype=np.float64),
    )


def test_fenicsx_generator_writes_distinct_model_artifacts_with_nonconstant_pressure(tmp_path):
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        model_name = command[command.index("--model") + 1]
        output_path = command[command.index("--output-path") + 1]
        grid_points = int(command[command.index("--grid-points") + 1])
        _write_solver_artifact(output_path, model_name=model_name, grid_points=grid_points)
        return subprocess.CompletedProcess(command, 0, stdout="generated\n", stderr="")

    paths = {
        model_name: generate_fenicsx_reference(
            model_name=model_name,
            output_dir=tmp_path / "fenicsx_references",
            grid_points=5,
            mesh_cells=4,
            viscosity=0.25,
            peak_velocity=1.0,
            runner=fake_runner,
        )
        for model_name in ("darcy", "stokes", "oseen", "navier_stokes")
    }

    assert len(set(paths.values())) == 4
    assert [call[0][0] for call in calls] == ["/usr/bin/python3"] * 4
    for model_name, path in paths.items():
        assert path == tmp_path / "fenicsx_references" / model_name / "fields.npz"
        with np.load(path) as fields:
            assert fields["coordinates"].shape == (25, 2)
            assert np.ptp(fields["reference_pressure"]) > 1.0
        assert FLOW_MODEL_METADATA[model_name]["pde_model_represented"].startswith(model_name)


def test_procurement_generates_missing_fenicsx_references_per_model(tmp_path, monkeypatch):
    generated = {}

    def fake_generate_fenicsx_reference(*, model_name, output_dir, grid_points, **kwargs):
        path = output_dir / model_name / "fields.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_solver_artifact(path, model_name=model_name, grid_points=grid_points)
        generated[model_name] = path
        return path

    monkeypatch.setattr(
        "pinn_fluid.fenicsx_solvers.generate_fenicsx_reference",
        fake_generate_fenicsx_reference,
    )
    config = ExperimentConfig(
        output_dir=tmp_path / "comparison",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=27,
        viscosity=0.25,
        vector_reference_source="fenicsx",
        fenicsx_reference_sample_paths=None,
    )

    manifest = run_result_procurement(config, git_commit="phase27-test")

    assert set(generated) == {"darcy", "stokes", "oseen", "navier_stokes"}
    assert manifest["reference_generators"] == {
        "darcy": "fenicsx_darcy_shared_domain",
        "stokes": "fenicsx_stokes_shared_domain",
        "oseen": "fenicsx_oseen_shared_domain",
        "navier_stokes": "fenicsx_navier_stokes_shared_domain",
    }
    for entry in manifest["results"]:
        metadata = entry["reference_metadata"]
        model_name = entry["model"]
        assert metadata["sample_path"] == str(generated[model_name])
        assert metadata["fenicsx_reference_model"] == model_name
        assert metadata["pde_model_represented"] == FLOW_MODEL_METADATA[model_name]["pde_model_represented"]
        assert metadata["reference_kind"] == "dedicated-solver"
        assert "manufactured" not in str(metadata).lower()
        fields_path = tmp_path / "comparison" / entry["artifacts"]["fields_npz"]
        with np.load(fields_path) as fields:
            assert np.ptp(fields["reference_pressure"]) > 1.0
