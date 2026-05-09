"""Phase 22 tests for OpenFOAM-backed vector-model reference fields."""

from __future__ import annotations

import json

import numpy as np

import pinn_fluid.experiments as experiments
from pinn_fluid.experiments import ExperimentConfig
from pinn_fluid.result_procurement import run_result_procurement


def _write_openfoam_sample(path, grid_points: int) -> None:
    values = np.linspace(0.0, 1.0, grid_points)
    lines = ["# x y z p Ux Uy Uz"]
    for x in values:
        for y in values:
            pressure = 1.0 - y + 0.1 * x
            u = 0.2 * x * (1.0 - x)
            v = -y * (1.0 - 0.25 * x)
            lines.append(f"{x} {y} 0.0 {pressure} {u} {v} 0.0")
    path.write_text("\n".join(lines))


def test_procurement_can_use_openfoam_sample_for_all_vector_references(tmp_path):
    sample_path = tmp_path / "sharedDomainGrid.xy"
    _write_openfoam_sample(sample_path, grid_points=5)
    config = ExperimentConfig(
        output_dir=tmp_path / "comparison",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=22,
        viscosity=0.25,
        darcy_reference_iterations=20,
        vector_reference_source="openfoam",
        openfoam_reference_sample_path=sample_path,
    )

    manifest = run_result_procurement(config, git_commit="phase22-test")

    assert manifest["reference_generators"] == {
        "darcy": "fd_darcy_reference",
        "stokes": "openfoam_simplefoam_shared_domain",
        "oseen": "openfoam_simplefoam_shared_domain",
        "navier_stokes": "openfoam_simplefoam_shared_domain",
    }
    assert manifest["all_metrics_finite"] is True
    for entry in manifest["results"]:
        metadata = entry["reference_metadata"]
        if entry["model"] == "darcy":
            continue
        assert entry["reference"] == "openfoam_simplefoam_shared_domain"
        assert metadata["reference_kind"] == "dedicated-solver"
        assert metadata["solver_stack"] == "OpenFOAM simpleFoam"
        assert metadata["compared_model"] == entry["model"]
        assert "manufactured" not in json.dumps(metadata).lower()
        assert "reference_continuity_rms" in entry["metrics"]
        assert "reference_momentum_rms" in entry["metrics"]

        fields_path = tmp_path / "comparison" / entry["artifacts"]["fields_npz"]
        with np.load(fields_path) as fields:
            np.testing.assert_allclose(
                fields["reference_pressure"].reshape(5, 5)[:, -1],
                0.1 * np.linspace(0.0, 1.0, 5),
            )
            assert np.ptp(fields["reference_pressure"]) > 0.5
            assert {
                "reference_speed",
                "reference_continuity",
                "reference_x_momentum",
                "reference_y_momentum",
                "reference_residual",
            }.issubset(fields.files)


def test_vector_experiments_no_longer_expose_manufactured_streamfunction_references():
    assert ExperimentConfig().vector_reference_source == "openfoam"
    assert not hasattr(experiments, "_stokes_reference_fields")
    assert not hasattr(experiments, "_oseen_reference_fields")
    assert not hasattr(experiments, "_navier_stokes_reference_fields")
