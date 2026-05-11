"""Phase 22/26 tests for dedicated vector-model reference fields."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pinn_fluid.experiments as experiments
from pinn_fluid.experiments import ExperimentConfig
from pinn_fluid.result_procurement import run_result_procurement


def _write_fenicsx_sample(path, grid_points: int, offset: float = 0.0) -> None:
    values = np.linspace(0.0, 1.0, grid_points)
    coordinates = []
    pressure = []
    u_values = []
    v_values = []
    for x in values:
        for y in values:
            coordinates.append((x, y))
            pressure.append((offset + 1.0 - y + 0.1 * x,))
            u_values.append((0.01 * offset + 0.2 * x * (1.0 - x),))
            v_values.append((-0.02 * offset - y * (1.0 - 0.25 * x),))
    with path.open("wb") as handle:
        np.savez(
            handle,
            coordinates=np.asarray(coordinates, dtype=np.float64),
            reference_pressure=np.asarray(pressure, dtype=np.float64),
            reference_u=np.asarray(u_values, dtype=np.float64),
            reference_v=np.asarray(v_values, dtype=np.float64),
        )


def test_procurement_rejects_one_openfoam_sample_for_multiple_flow_models(tmp_path):
    sample_path = tmp_path / "sharedDomainGrid.xy"
    _write_fenicsx_sample(sample_path, grid_points=5)
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
        vector_reference_source="fenicsx",
        openfoam_reference_sample_path=sample_path,
    )

    with pytest.raises(ValueError, match="FEniCSx references do not accept OpenFOAM sample paths"):
        run_result_procurement(config, git_commit="phase22-test")


def test_procurement_uses_model_specific_openfoam_samples(tmp_path):
    sample_paths = {}
    for index, model in enumerate(("darcy", "stokes", "oseen", "navier_stokes")):
        sample_path = tmp_path / model / "sharedDomainGrid.xy"
        sample_path.parent.mkdir()
        _write_fenicsx_sample(sample_path, grid_points=5, offset=float(index))
        sample_paths[model] = sample_path
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
        vector_reference_source="fenicsx",
        fenicsx_reference_sample_paths=sample_paths,
    )

    manifest = run_result_procurement(config, git_commit="phase22-test")

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
        assert "manufactured" not in json.dumps(metadata).lower()
        if entry["model"] != "darcy":
            assert "reference_continuity_rms" in entry["metrics"]
            assert "reference_momentum_rms" in entry["metrics"]

        fields_path = tmp_path / "comparison" / entry["artifacts"]["fields_npz"]
        with np.load(fields_path) as fields:
            offset = list(sample_paths).index(entry["model"])
            np.testing.assert_allclose(
                fields["reference_pressure"].reshape(5, 5)[:, -1],
                offset + 0.1 * np.linspace(0.0, 1.0, 5),
            )
            assert np.ptp(fields["reference_pressure"]) > 0.5
            expected_fields = {"reference_speed", "reference_residual"}
            if entry["model"] == "darcy":
                expected_fields.update({"reference_velocity", "reference_u", "reference_v"})
            else:
                expected_fields.update(
                    {"reference_continuity", "reference_x_momentum", "reference_y_momentum"}
                )
            assert expected_fields.issubset(fields.files)


def test_vector_experiments_no_longer_expose_manufactured_streamfunction_references():
    assert ExperimentConfig().vector_reference_source == "fenicsx"
    assert not hasattr(experiments, "_stokes_reference_fields")
    assert not hasattr(experiments, "_oseen_reference_fields")
    assert not hasattr(experiments, "_navier_stokes_reference_fields")
