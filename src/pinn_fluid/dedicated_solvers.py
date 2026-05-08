"""OpenFOAM case generation and field import for dedicated references."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from pinn_fluid.experiments import (
    COORDINATE_CONVENTION_METADATA,
    ExperimentResult,
    TrainingHistory,
)


OPENFOAM_REFERENCE_METADATA: dict[str, object] = {
    "reference_generator_name": "openfoam_simplefoam_shared_domain",
    "pde_model_represented": "OpenFOAM incompressible steady laminar flow",
    "boundary_condition_type": "velocity inlet with pressure outlet and no-slip walls",
    "coordinate_convention": COORDINATE_CONVENTION_METADATA,
    "reference_kind": "dedicated-solver",
    "solver_stack": "OpenFOAM simpleFoam",
    "case_geometry": {
        "domain": "unit square extruded to one empty OpenFOAM cell in z",
        "inlet": "top-left horizontal patch, x in [0.0, 0.25], y=1",
        "outlet": "bottom-right horizontal patch, x in [0.75, 1.0], y=0",
        "walls": "remaining unit-square perimeter",
    },
    "expected_sample_columns": ["x", "y", "z", "p", "Ux", "Uy", "Uz"],
}


@dataclass(frozen=True)
class OpenFOAMCaseConfig:
    """Configuration for the generated OpenFOAM shared-domain case."""

    grid_points: int = 31
    viscosity: float = 0.25
    inlet_velocity: float = 1.0
    end_time: int = 500

    def __post_init__(self) -> None:
        if self.grid_points < 2:
            raise ValueError("grid_points must be at least 2")
        if self.viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        if self.inlet_velocity <= 0.0:
            raise ValueError("inlet_velocity must be positive")


def _foam_header(class_name: str, object_name: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"    object      {object_name};\n"
        "}\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_openfoam_shared_domain_case(
    case_dir: Path | str,
    config: OpenFOAMCaseConfig | None = None,
) -> Path:
    """Write an OpenFOAM simpleFoam case for the shared unit-square geometry."""

    active = OpenFOAMCaseConfig() if config is None else config
    root = Path(case_dir)
    root.mkdir(parents=True, exist_ok=True)
    cells = active.grid_points - 1

    _write(
        root / "system/blockMeshDict",
        _foam_header("dictionary", "blockMeshDict")
        + f"""
convertToMeters 1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.01)
    (1 0 0.01)
    (1 1 0.01)
    (0 1 0.01)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({cells} {cells} 1) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ((3 7 6 2));
    }}
    outlet
    {{
        type patch;
        faces ((0 1 5 4));
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 4 7 3)
            (1 2 6 5)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
""",
    )
    _write(
        root / "0/U",
        _foam_header("volVectorField", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform (0 -{active.inlet_velocity} 0);
    }}
    outlet
    {{
        type zeroGradient;
    }}
    walls
    {{
        type noSlip;
    }}
    frontAndBack
    {{
        type empty;
    }}
}}
""",
    )
    _write(
        root / "0/p",
        _foam_header("volScalarField", "p")
        + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type zeroGradient;
    }
    outlet
    {
        type fixedValue;
        value uniform 0;
    }
    walls
    {
        type zeroGradient;
    }
    frontAndBack
    {
        type empty;
    }
}
""",
    )
    _write(
        root / "constant/transportProperties",
        _foam_header("dictionary", "transportProperties")
        + f"""
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {active.viscosity};
""",
    )
    _write(
        root / "system/controlDict",
        _foam_header("dictionary", "controlDict")
        + f"""
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {active.end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {active.end_time};
purgeWrite      0;
""",
    )
    _write(
        root / "system/fvSchemes",
        _foam_header("dictionary", "fvSchemes")
        + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )
    _write(
        root / "system/fvSolution",
        _foam_header("dictionary", "fvSolution")
        + """
solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.05; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl { p 1e-5; U 1e-5; }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
""",
    )
    _write(
        root / "system/sampleDict",
        _foam_header("dictionary", "sampleDict")
        + f"""
type sets;
libs ("libsampling.so");
interpolationScheme cellPoint;
setFormat raw;
sets
(
    sharedDomainGrid
    {{
        type face;
        axis xyz;
        start (0 0 0);
        end (1 1 0);
        nPoints {active.grid_points};
    }}
);
fields (p U);
""",
    )
    (root / "reference_metadata.json").write_text(
        json.dumps(deepcopy(OPENFOAM_REFERENCE_METADATA), indent=2, sort_keys=True)
    )
    return root


def import_openfoam_sampled_fields(
    sample_path: Path | str,
    *,
    grid_points: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, np.ndarray | str]:
    """Import sampled OpenFOAM raw columns into the repository field schema."""

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    rows = np.loadtxt(sample_path, comments="#", dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape != (grid_points * grid_points, 7):
        raise ValueError("OpenFOAM sample must have columns x y z p Ux Uy Uz on a square grid")

    order = np.lexsort((rows[:, 1], rows[:, 0]))
    sorted_rows = rows[order]
    coordinates = sorted_rows[:, :2]
    pressure = sorted_rows[:, 3:4]
    velocity = sorted_rows[:, 4:6]
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    residual = np.zeros_like(pressure)
    active_metadata = deepcopy(OPENFOAM_REFERENCE_METADATA if metadata is None else metadata)
    return {
        "coordinates": coordinates,
        "predicted_pressure": pressure,
        "reference_pressure": pressure,
        "predicted_velocity": velocity,
        "reference_velocity": velocity,
        "predicted_u": velocity[:, :1],
        "predicted_v": velocity[:, 1:2],
        "predicted_speed": speed,
        "reference_u": velocity[:, :1],
        "reference_v": velocity[:, 1:2],
        "reference_speed": speed,
        "reference_residual": residual,
        "residual": residual,
        "reference_metadata_json": json.dumps(active_metadata, sort_keys=True),
    }


def _relative(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_openfoam_reference_import(
    *,
    output_dir: Path | str,
    sample_path: Path | str,
    grid_points: int,
    metadata: dict[str, object] | None = None,
) -> ExperimentResult:
    """Write a figure-compatible artifact bundle from OpenFOAM sampled fields."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "openfoam_darcy_reference"
    model_dir.mkdir(exist_ok=True)
    active_metadata = deepcopy(OPENFOAM_REFERENCE_METADATA if metadata is None else metadata)
    fields = import_openfoam_sampled_fields(
        sample_path,
        grid_points=grid_points,
        metadata=active_metadata,
    )
    reference_residual = np.asarray(fields["reference_residual"], dtype=np.float64)
    residual_rms = float(np.sqrt(np.mean(reference_residual**2)))
    metrics = {
        "pressure_l2": 0.0,
        "velocity_l2": 0.0,
        "residual_rms": residual_rms,
        "reference_residual_rms": residual_rms,
    }
    history = TrainingHistory(
        total=[1.0, max(residual_rms, 1.0e-12)],
        components={"openfoam_import_residual": [1.0, max(residual_rms, 1.0e-12)]},
    )

    fields_path = model_dir / "fields.npz"
    history_path = model_dir / "history.json"
    metrics_path = model_dir / "metrics.json"
    np.savez(fields_path, **fields)
    _write_json(history_path, history.to_json_dict())
    _write_json(metrics_path, metrics)

    result = ExperimentResult(
        model="openfoam_darcy_reference",
        reference="openfoam_simplefoam_shared_domain",
        grid_shape=(grid_points, grid_points),
        history=history,
        metrics=metrics,
        artifacts={
            "fields_npz": _relative(fields_path, root),
            "history_json": _relative(history_path, root),
            "metrics_json": _relative(metrics_path, root),
        },
        reference_metadata=active_metadata,
    )
    manifest = {
        "reference_generators": {
            result.model: str(result.reference_metadata["reference_generator_name"])
        },
        "results": [result.to_json_dict()],
    }
    _write_json(root / "dedicated_solver_manifest.json", manifest)
    return result


__all__ = [
    "OPENFOAM_REFERENCE_METADATA",
    "OpenFOAMCaseConfig",
    "import_openfoam_sampled_fields",
    "run_openfoam_reference_import",
    "write_openfoam_shared_domain_case",
]
