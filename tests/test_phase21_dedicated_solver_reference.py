import json

import numpy as np

from pinn_fluid.dedicated_solvers import (
    OPENFOAM_REFERENCE_METADATA,
    OpenFOAMCaseConfig,
    import_openfoam_sampled_fields,
    run_openfoam_reference_import,
    write_openfoam_shared_domain_case,
)
from pinn_fluid.experiments import COORDINATE_CONVENTION_METADATA


def test_openfoam_case_writer_records_shared_domain_geometry(tmp_path):
    case_dir = write_openfoam_shared_domain_case(
        tmp_path / "openfoam_case",
        OpenFOAMCaseConfig(grid_points=9, viscosity=0.25),
    )

    expected_files = {
        "0/U",
        "0/p",
        "constant/transportProperties",
        "system/blockMeshDict",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/sampleDict",
        "reference_metadata.json",
    }
    assert expected_files.issubset(
        {path.relative_to(case_dir).as_posix() for path in case_dir.rglob("*") if path.is_file()}
    )

    metadata = json.loads((case_dir / "reference_metadata.json").read_text())
    assert metadata["reference_generator_name"] == "openfoam_simplefoam_shared_domain"
    assert metadata["solver_stack"] == "OpenFOAM simpleFoam"
    assert metadata["reference_kind"] == "dedicated-solver"
    assert metadata["coordinate_convention"] == COORDINATE_CONVENTION_METADATA
    assert metadata["case_geometry"]["inlet"] == "top-left horizontal patch, x in [0.0, 0.25], y=1"
    assert metadata["case_geometry"]["outlet"] == "bottom-right horizontal patch, x in [0.75, 1.0], y=0"

    block_mesh = (case_dir / "system/blockMeshDict").read_text()
    assert "inlet" in block_mesh
    assert "outlet" in block_mesh
    assert "frontAndBack" in block_mesh
    assert "empty" in block_mesh


def test_openfoam_sample_import_preserves_orientation_and_schema(tmp_path):
    sample_path = tmp_path / "sample.xy"
    sample_path.write_text(
        "\n".join(
            [
                "# x y z p Ux Uy Uz",
                "0.0 0.0 0.0 0.0 0.00 0.00 0.0",
                "0.0 0.5 0.0 0.5 0.00 -0.50 0.0",
                "0.0 1.0 0.0 1.0 0.00 -1.00 0.0",
                "0.5 0.0 0.0 0.0 0.25 0.00 0.0",
                "0.5 0.5 0.0 0.5 0.25 -0.50 0.0",
                "0.5 1.0 0.0 1.0 0.25 -1.00 0.0",
                "1.0 0.0 0.0 0.0 0.00 0.00 0.0",
                "1.0 0.5 0.0 0.5 0.00 -0.50 0.0",
                "1.0 1.0 0.0 1.0 0.00 -1.00 0.0",
            ]
        )
    )

    fields = import_openfoam_sampled_fields(
        sample_path,
        grid_points=3,
        metadata=OPENFOAM_REFERENCE_METADATA,
    )

    assert fields["coordinates"].shape == (9, 2)
    np.testing.assert_allclose(fields["coordinates"][:3], [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(fields["reference_pressure"].reshape(3, 3)[:, 0], 0.0)
    np.testing.assert_allclose(fields["reference_pressure"].reshape(3, 3)[:, -1], 1.0)
    np.testing.assert_allclose(fields["reference_v"].reshape(3, 3)[:, -1], -1.0)
    assert np.isfinite(fields["reference_speed"]).all()
    assert json.loads(str(fields["reference_metadata_json"]))["solver_stack"] == "OpenFOAM simpleFoam"


def test_openfoam_import_writes_figure_compatible_artifacts(tmp_path):
    sample_path = tmp_path / "sample.xy"
    sample_path.write_text(
        "\n".join(
            [
                "0.0 0.0 0.0 0.0 0.0 0.0 0.0",
                "0.0 1.0 0.0 1.0 0.0 -1.0 0.0",
                "1.0 0.0 0.0 0.0 0.0 0.0 0.0",
                "1.0 1.0 0.0 1.0 0.0 -1.0 0.0",
            ]
        )
    )

    result = run_openfoam_reference_import(
        output_dir=tmp_path / "artifacts",
        sample_path=sample_path,
        grid_points=2,
    )

    assert result.model == "openfoam_darcy_reference"
    assert result.reference == "openfoam_simplefoam_shared_domain"
    assert result.reference_metadata["solver_stack"] == "OpenFOAM simpleFoam"
    assert np.isfinite(list(result.metrics.values())).all()

    fields_path = tmp_path / "artifacts" / result.artifacts["fields_npz"]
    manifest_path = tmp_path / "artifacts" / "dedicated_solver_manifest.json"
    with np.load(fields_path) as fields:
        expected = {
            "coordinates",
            "predicted_pressure",
            "reference_pressure",
            "predicted_velocity",
            "reference_velocity",
            "predicted_u",
            "predicted_v",
            "reference_u",
            "reference_v",
            "predicted_speed",
            "reference_speed",
            "reference_residual",
            "residual",
            "reference_metadata_json",
        }
        assert expected.issubset(fields.files)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["reference_generators"] == {
        "openfoam_darcy_reference": "openfoam_simplefoam_shared_domain"
    }
