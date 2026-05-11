"""Phase 26 tests for FEniCSx-backed model-specific references."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pinn_fluid.experiments import ExperimentConfig
from pinn_fluid.result_procurement import run_result_procurement


def _write_fenicsx_fields(path, grid_points: int, offset: float) -> None:
    values = np.linspace(0.0, 1.0, grid_points)
    coordinates = []
    pressure = []
    u = []
    v = []
    for x in values:
        for y in values:
            coordinates.append((x, y))
            pressure.append((offset + 1.0 - y + 0.1 * x,))
            u.append((offset * 0.01 + 0.2 * x * (1.0 - x),))
            v.append((-offset * 0.02 - y * (1.0 - 0.25 * x),))
    np.savez(
        path,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        reference_pressure=np.asarray(pressure, dtype=np.float64),
        reference_u=np.asarray(u, dtype=np.float64),
        reference_v=np.asarray(v, dtype=np.float64),
    )


def test_procurement_defaults_to_fenicsx_and_rejects_openfoam_source(tmp_path):
    assert ExperimentConfig().vector_reference_source == "fenicsx"
    config = ExperimentConfig(
        output_dir=tmp_path / "comparison",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        vector_reference_source="openfoam",
    )

    with pytest.raises(ValueError, match="vector_reference_source must be 'fenicsx'"):
        run_result_procurement(config, git_commit="phase26-test")


def test_procurement_uses_model_specific_fenicsx_field_artifacts(tmp_path):
    sample_paths = {}
    for index, model in enumerate(("darcy", "stokes", "oseen", "navier_stokes")):
        path = tmp_path / f"{model}_fenicsx_fields.npz"
        _write_fenicsx_fields(path, grid_points=5, offset=float(index))
        sample_paths[model] = path
    config = ExperimentConfig(
        output_dir=tmp_path / "comparison",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=26,
        viscosity=0.25,
        vector_reference_source="fenicsx",
        fenicsx_reference_sample_paths=sample_paths,
    )

    manifest = run_result_procurement(config, git_commit="phase26-test")

    assert manifest["reference_generators"] == {
        "darcy": "fenicsx_darcy_shared_domain",
        "stokes": "fenicsx_stokes_shared_domain",
        "oseen": "fenicsx_oseen_shared_domain",
        "navier_stokes": "fenicsx_navier_stokes_shared_domain",
    }
    assert manifest["all_metrics_finite"] is True
    for entry in manifest["results"]:
        metadata = entry["reference_metadata"]
        assert entry["reference"] == manifest["reference_generators"][entry["model"]]
        assert metadata["reference_kind"] == "dedicated-solver"
        assert metadata["solver_stack"] == "FEniCSx/dolfinx"
        assert metadata["compared_model"] == entry["model"]
        assert metadata["sample_path"] == str(sample_paths[entry["model"]])
        assert "openfoam" not in json.dumps(metadata).lower()
