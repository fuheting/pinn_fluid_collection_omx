"""Phase 15 contracts for current reference-generation behavior."""

import json

import numpy as np

from pinn_fluid.experiments import (
    ExperimentConfig,
    _fd_darcy_reference,
    _grid_coordinates,
    _shared_patch_vector_reference,
    run_darcy_experiment,
    run_oseen_experiment,
    run_poiseuille_navier_stokes_experiment,
    run_stokes_experiment,
)
from pinn_fluid.figures import _reshape
from pinn_fluid.result_procurement import run_result_procurement


EXPECTED_COORDINATE_CONVENTION = {
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


def test_fd_darcy_reference_solves_shared_pressure_patch_contract():
    pressure, velocity = _fd_darcy_reference(grid_points=9, iterations=80)

    assert pressure.shape == (81, 1)
    assert velocity.shape == (81, 2)

    pressure_grid = pressure.reshape(9, 9)
    top_inlet_end = round(0.25 * (9 - 1))
    bottom_outlet_start = round(0.75 * (9 - 1))
    assert np.allclose(pressure_grid[: top_inlet_end + 1, -1], 1.0)
    assert np.allclose(pressure_grid[bottom_outlet_start:, 0], 0.0)
    assert np.all(np.isfinite(pressure))
    assert np.all(np.isfinite(velocity))


def test_shared_patch_vector_reference_is_darcy_pressure_velocity_wrapped_for_vector_models():
    pressure, velocity = _fd_darcy_reference(grid_points=7, iterations=60)
    reference = _shared_patch_vector_reference(grid_points=7, iterations=60)

    assert set(reference) == {"u", "v", "pressure", "velocity"}
    assert np.array_equal(reference["pressure"], pressure)
    assert np.array_equal(reference["velocity"], velocity)
    assert np.array_equal(reference["u"], velocity[:, :1])
    assert np.array_equal(reference["v"], velocity[:, 1:2])


def test_coordinate_flattening_and_figure_reshape_keep_y_up_orientation():
    coordinates = _grid_coordinates(3).numpy()

    assert coordinates.tolist() == [
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.5, 1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
    ]
    assert _reshape(np.arange(9), 3).tolist() == [
        [0.0, 3.0, 6.0],
        [1.0, 4.0, 7.0],
        [2.0, 5.0, 8.0],
    ]


def test_experiment_results_and_artifacts_record_reference_metadata(tmp_path, openfoam_config_factory):
    config = openfoam_config_factory(
        output_dir=tmp_path,
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=3,
        viscosity=0.25,
        darcy_reference_iterations=20,
    )

    results = [
        run_darcy_experiment(config),
        run_stokes_experiment(config),
        run_oseen_experiment(config),
        run_poiseuille_navier_stokes_experiment(config),
    ]

    metadata_by_model = {result.model: result.reference_metadata for result in results}
    assert metadata_by_model["darcy"] == {
        "reference_generator_name": "fd_darcy_reference",
        "pde_model_represented": "Darcy pressure Laplace equation",
        "boundary_condition_type": "pressure Dirichlet inlet/outlet with no-normal-flow walls",
        "coordinate_convention": EXPECTED_COORDINATE_CONVENTION,
        "reference_kind": "finite-difference",
    }
    assert metadata_by_model["navier_stokes"]["reference_generator_name"] == (
        "openfoam_simplefoam_shared_domain"
    )
    assert metadata_by_model["navier_stokes"]["pde_model_represented"] == (
        "OpenFOAM incompressible steady laminar flow"
    )
    assert metadata_by_model["navier_stokes"]["coordinate_convention"] == (
        EXPECTED_COORDINATE_CONVENTION
    )
    assert metadata_by_model["navier_stokes"]["reference_kind"] == "dedicated-solver"
    assert metadata_by_model["navier_stokes"]["compared_model"] == "navier_stokes"

    for result in results:
        with np.load(tmp_path / result.artifacts["fields_npz"]) as fields:
            assert json.loads(str(fields["reference_metadata_json"])) == result.reference_metadata


def test_legacy_shared_patch_vector_reference_remains_available_for_contract_audit():
    config = ExperimentConfig(
        grid_points=5,
        darcy_reference_iterations=20,
    )
    pressure, velocity = _fd_darcy_reference(
        grid_points=config.grid_points,
        iterations=config.darcy_reference_iterations,
    )
    expected = _shared_patch_vector_reference(
        grid_points=config.grid_points,
        iterations=config.darcy_reference_iterations,
    )

    assert np.array_equal(expected["pressure"], pressure)
    assert np.array_equal(expected["velocity"], velocity)
    assert np.array_equal(expected["u"], velocity[:, :1])
    assert np.array_equal(expected["v"], velocity[:, 1:2])


def test_procurement_manifest_records_reference_metadata(tmp_path, openfoam_sample_writer):
    sample_path = openfoam_sample_writer(tmp_path / "sharedDomainGrid.xy", grid_points=5)
    config = ExperimentConfig(
        output_dir=tmp_path / "procured",
        grid_points=5,
        training_steps=4,
        hidden_width=6,
        hidden_layers=1,
        learning_rate=0.03,
        seed=5,
        viscosity=0.25,
        darcy_reference_iterations=20,
        openfoam_reference_sample_path=sample_path,
    )

    manifest = run_result_procurement(config, git_commit="phase15-test")

    for entry in manifest["results"]:
        assert "reference_metadata" in entry
        assert entry["reference_metadata"]["coordinate_convention"] == EXPECTED_COORDINATE_CONVENTION
        if entry["model"] == "darcy":
            assert entry["reference_metadata"]["reference_kind"] == "finite-difference"
        elif entry["model"] in {"stokes", "oseen", "navier_stokes"}:
            assert entry["reference_metadata"]["reference_kind"] == "dedicated-solver"
