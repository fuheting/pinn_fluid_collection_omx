"""Phase 19 tests for Navier-Stokes actual field schema."""

import json

import numpy as np

from pinn_fluid.experiments import run_poiseuille_navier_stokes_experiment
from pinn_fluid.figures import generate_figure_bundle


def test_run_navier_stokes_experiment_uses_openfoam_reference_and_schema(
    tmp_path,
    openfoam_config_factory,
):
    config = openfoam_config_factory(
        output_dir=tmp_path,
        grid_points=9,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=11,
        viscosity=0.25,
        peak_velocity=1.0,
    )

    result = run_poiseuille_navier_stokes_experiment(config)

    assert result.model == "navier_stokes"
    assert result.reference == "fenicsx_navier_stokes_shared_domain"
    assert result.reference_metadata["reference_generator_name"] == "fenicsx_navier_stokes_shared_domain"
    assert result.reference_metadata["pde_model_represented"] == "FEniCSx finite-element flow solve"
    assert result.reference_metadata["reference_kind"] == "dedicated-solver"
    assert result.reference_metadata["compared_model"] == "navier_stokes"
    assert result.metrics["velocity_l2"] >= 0.0
    assert result.metrics["pressure_l2"] >= 0.0
    assert result.metrics["residual_rms"] >= 0.0
    assert result.metrics["reference_continuity_rms"] >= 0.0
    assert result.metrics["reference_momentum_rms"] >= 0.0

    with np.load(tmp_path / result.artifacts["fields_npz"]) as fields:
        assert {
            "predicted_u",
            "predicted_v",
            "predicted_pressure",
            "reference_u",
            "reference_v",
            "reference_pressure",
            "reference_speed",
            "reference_continuity",
            "reference_x_momentum",
            "reference_y_momentum",
            "reference_residual",
            "residual",
            "reference_metadata_json",
        }.issubset(fields.files)
        assert fields["coordinates"].shape == (81, 2)
        assert np.ptp(fields["reference_pressure"]) > 0.5
        metadata = json.loads(str(fields["reference_metadata_json"]))
        assert metadata == result.reference_metadata

    manifest = generate_figure_bundle(tmp_path, models=("navier_stokes",))
    assert manifest["models"][0]["model"] == "navier_stokes"
    assert (tmp_path / manifest["models"][0]["field_panel"]).is_file()
    assert (tmp_path / manifest["models"][0]["pressure_velocity_quiver"]).is_file()
