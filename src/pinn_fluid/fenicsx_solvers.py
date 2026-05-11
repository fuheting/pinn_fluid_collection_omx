"""FEniCSx field generation, import, and dependency boundary for references."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

import numpy as np

COORDINATE_CONVENTION_METADATA: dict[str, object] = {
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

FLOW_MODEL_METADATA: dict[str, dict[str, object]] = {
    "darcy": {
        "pde_model_represented": "darcy porous-flow pressure equation",
        "fenicsx_formulation": "CG1 pressure solve with velocity recovered from -grad(p)",
    },
    "stokes": {
        "pde_model_represented": "stokes incompressible linear momentum and continuity",
        "fenicsx_formulation": "Taylor-Hood P2-P1 mixed finite-element solve",
    },
    "oseen": {
        "pde_model_represented": "oseen incompressible linearized momentum and continuity",
        "fenicsx_formulation": "Taylor-Hood P2-P1 mixed finite-element solve",
        "convection_velocity": [1.0, 0.0],
    },
    "navier_stokes": {
        "pde_model_represented": "navier_stokes steady incompressible nonlinear momentum and continuity",
        "fenicsx_formulation": "Taylor-Hood P2-P1 nonlinear finite-element solve",
    },
}

FENICSX_REFERENCE_METADATA: dict[str, object] = {
    "reference_generator_name": "fenicsx_shared_domain",
    "pde_model_represented": "FEniCSx finite-element flow solve",
    "boundary_condition_type": "velocity inlet with pressure outlet and no-slip walls",
    "coordinate_convention": COORDINATE_CONVENTION_METADATA,
    "reference_kind": "dedicated-solver",
    "solver_stack": "FEniCSx/dolfinx",
    "case_geometry": {
        "domain": "unit square",
        "inlet": "top-left horizontal patch, x in [0.0, 0.25], y=1",
        "outlet": "bottom-right horizontal patch, x in [0.75, 1.0], y=0",
        "walls": "remaining unit-square perimeter",
    },
    "expected_sample_arrays": ["coordinates", "reference_pressure", "reference_u", "reference_v"],
}

FLOW_MODELS = tuple(FLOW_MODEL_METADATA)
SYSTEM_FENICSX_PYTHON = "/usr/bin/python3"


def import_fenicsx_sampled_fields(
    sample_path: Path | str,
    *,
    grid_points: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, np.ndarray | str]:
    """Import a FEniCSx sampled field artifact into the repository field schema."""

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    with np.load(sample_path) as data:
        coordinates = np.asarray(data["coordinates"], dtype=np.float64)
        pressure = np.asarray(data["reference_pressure"], dtype=np.float64).reshape(-1, 1)
        u = np.asarray(data["reference_u"], dtype=np.float64).reshape(-1, 1)
        v = np.asarray(data["reference_v"], dtype=np.float64).reshape(-1, 1)
    expected_points = grid_points * grid_points
    if coordinates.shape != (expected_points, 2):
        raise ValueError("FEniCSx coordinates must describe a square grid")
    for name, values in {
        "reference_pressure": pressure,
        "reference_u": u,
        "reference_v": v,
    }.items():
        if values.shape != (expected_points, 1):
            raise ValueError(f"FEniCSx {name} must have shape (grid_points**2, 1)")

    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    coordinates = coordinates[order]
    pressure = pressure[order]
    u = u[order]
    v = v[order]
    velocity = np.column_stack((u.reshape(-1), v.reshape(-1)))
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    residual = np.zeros_like(pressure)
    active_metadata = deepcopy(FENICSX_REFERENCE_METADATA if metadata is None else metadata)
    return {
        "coordinates": coordinates,
        "predicted_pressure": pressure,
        "reference_pressure": pressure,
        "predicted_velocity": velocity,
        "reference_velocity": velocity,
        "predicted_u": u,
        "predicted_v": v,
        "predicted_speed": speed,
        "reference_u": u,
        "reference_v": v,
        "reference_speed": speed,
        "reference_residual": residual,
        "residual": residual,
        "reference_metadata_json": json.dumps(active_metadata, sort_keys=True),
    }


def require_fenicsx() -> None:
    """Raise a clear install-boundary error when FEniCSx is unavailable."""

    missing = [
        package
        for package in ("dolfinx", "ufl", "petsc4py", "mpi4py")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise RuntimeError(
            "FEniCSx ground-truth generation requires missing packages: "
            + ", ".join(missing)
            + ". Install FEniCSx/dolfinx before running generated references."
        )


def _pythonpath_with_src(env: dict[str, str] | None = None) -> dict[str, str]:
    active = dict(os.environ if env is None else env)
    src_root = Path(__file__).resolve().parents[1]
    existing = active.get("PYTHONPATH")
    active["PYTHONPATH"] = (
        str(src_root) if not existing else f"{src_root}{os.pathsep}{existing}"
    )
    return active


def generate_fenicsx_reference(
    *,
    model_name: str,
    output_dir: Path | str,
    grid_points: int,
    mesh_cells: int,
    viscosity: float,
    peak_velocity: float,
    python_executable: str = SYSTEM_FENICSX_PYTHON,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """Generate one model-specific FEniCSx sampled-field artifact."""

    if model_name not in FLOW_MODEL_METADATA:
        raise ValueError(f"unsupported FEniCSx model: {model_name}")
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    if mesh_cells < 1:
        raise ValueError("mesh_cells must be positive")
    output_path = Path(output_dir) / model_name / "fields.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-m",
        "pinn_fluid.fenicsx_solvers",
        "solve",
        "--model",
        model_name,
        "--output-path",
        str(output_path),
        "--grid-points",
        str(grid_points),
        "--mesh-cells",
        str(mesh_cells),
        "--viscosity",
        str(viscosity),
        "--peak-velocity",
        str(peak_velocity),
    ]
    try:
        runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=_pythonpath_with_src(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{python_executable} was not found; FEniCSx generation requires the system Python"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"FEniCSx {model_name} generation failed: {detail}") from exc
    if not output_path.is_file():
        raise RuntimeError(f"FEniCSx {model_name} generation did not write {output_path}")
    return output_path


def _inlet(x: np.ndarray) -> np.ndarray:
    return np.isclose(x[1], 1.0) & (x[0] >= -1.0e-12) & (x[0] <= 0.25 + 1.0e-12)


def _outlet(x: np.ndarray) -> np.ndarray:
    return np.isclose(x[1], 0.0) & (x[0] >= 0.75 - 1.0e-12) & (x[0] <= 1.0 + 1.0e-12)


def _walls(x: np.ndarray) -> np.ndarray:
    boundary = (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )
    return boundary & ~_inlet(x) & ~_outlet(x)


def _run_darcy_solve(
    *,
    grid_points: int,
    mesh_cells: int,
    viscosity: float,
    peak_velocity: float,
) -> tuple[Any, Any, Any]:
    from mpi4py import MPI
    from petsc4py.PETSc import ScalarType

    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem

    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_cells, mesh_cells)
    v_space = fem.functionspace(domain, ("Lagrange", 1))
    outlet_facets = mesh.locate_entities_boundary(domain, 1, _outlet)
    outlet_dofs = fem.locate_dofs_topological(v_space, 1, outlet_facets)
    bc = fem.dirichletbc(ScalarType(0.0), outlet_dofs, v_space)

    inlet_facets = mesh.locate_entities_boundary(domain, 1, _inlet)
    facet_tags = mesh.meshtags(
        domain,
        1,
        inlet_facets.astype(np.int32),
        np.full_like(inlet_facets, 1, dtype=np.int32),
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    pressure_trial = ufl.TrialFunction(v_space)
    test = ufl.TestFunction(v_space)
    a = ufl.inner(ufl.grad(pressure_trial), ufl.grad(test)) * ufl.dx
    L = peak_velocity * test * ds(1)
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="pinn_fluid_fenicsx_darcy_pressure_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    pressure = problem.solve()

    velocity_space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    velocity_trial = ufl.TrialFunction(velocity_space)
    velocity_test = ufl.TestFunction(velocity_space)
    projection = LinearProblem(
        ufl.inner(velocity_trial, velocity_test) * ufl.dx,
        ufl.inner(-ufl.grad(pressure), velocity_test) * ufl.dx,
        petsc_options_prefix="pinn_fluid_fenicsx_darcy_velocity_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    velocity = projection.solve()
    return domain, velocity, pressure


def _velocity_expression(*, peak_velocity: float) -> Callable[[np.ndarray], np.ndarray]:
    def expression(x: np.ndarray) -> np.ndarray:
        return np.vstack((np.zeros_like(x[0]), -peak_velocity * np.ones_like(x[0])))

    return expression


def _run_stokes_like_solve(
    *,
    model_name: str,
    grid_points: int,
    mesh_cells: int,
    viscosity: float,
    peak_velocity: float,
) -> tuple[Any, Any, Any]:
    from mpi4py import MPI
    from petsc4py import PETSc

    import ufl
    from basix.ufl import element, mixed_element
    from dolfinx import default_real_type, fem, mesh
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_cells, mesh_cells)
    p2 = element(
        "Lagrange",
        domain.basix_cell(),
        degree=2,
        shape=(domain.geometry.dim,),
        dtype=default_real_type,
    )
    p1 = element("Lagrange", domain.basix_cell(), degree=1, dtype=default_real_type)
    mixed_space = fem.functionspace(domain, mixed_element([p2, p1]))
    velocity_subspace = mixed_space.sub(0)
    pressure_subspace = mixed_space.sub(1)
    velocity_space, _ = velocity_subspace.collapse()
    pressure_space, _ = pressure_subspace.collapse()

    zero_velocity = fem.Function(velocity_space)
    inlet_velocity = fem.Function(velocity_space)
    inlet_velocity.interpolate(_velocity_expression(peak_velocity=peak_velocity))
    inlet_facets = mesh.locate_entities_boundary(domain, 1, _inlet)
    wall_facets = mesh.locate_entities_boundary(domain, 1, _walls)
    outlet_facets = mesh.locate_entities_boundary(domain, 1, _outlet)
    bcs = [
        fem.dirichletbc(
            inlet_velocity,
            fem.locate_dofs_topological((velocity_subspace, velocity_space), 1, inlet_facets),
            velocity_subspace,
        ),
        fem.dirichletbc(
            zero_velocity,
            fem.locate_dofs_topological((velocity_subspace, velocity_space), 1, wall_facets),
            velocity_subspace,
        ),
        fem.dirichletbc(
            fem.Function(pressure_space),
            fem.locate_dofs_topological((pressure_subspace, pressure_space), 1, outlet_facets),
            pressure_subspace,
        ),
    ]

    if model_name in {"stokes", "oseen"}:
        velocity, pressure = ufl.TrialFunctions(mixed_space)
        test_velocity, test_pressure = ufl.TestFunctions(mixed_space)
        momentum = viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(test_velocity))
        if model_name == "oseen":
            beta = fem.Constant(domain, PETSc.ScalarType((peak_velocity, 0.0)))
            momentum += ufl.inner(ufl.dot(beta, ufl.grad(velocity)), test_velocity)
        a = (
            momentum
            - pressure * ufl.div(test_velocity)
            + test_pressure * ufl.div(velocity)
        ) * ufl.dx
        L = ufl.inner(
            fem.Constant(domain, PETSc.ScalarType((0.0, 0.0))),
            test_velocity,
        ) * ufl.dx
        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=f"pinn_fluid_fenicsx_{model_name}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        solution = problem.solve()
    else:
        solution = fem.Function(mixed_space)
        velocity, pressure = ufl.split(solution)
        test_velocity, test_pressure = ufl.TestFunctions(mixed_space)
        F = (
            viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(test_velocity))
            + ufl.inner(ufl.dot(velocity, ufl.grad(velocity)), test_velocity)
            - pressure * ufl.div(test_velocity)
            + test_pressure * ufl.div(velocity)
        ) * ufl.dx
        problem = NonlinearProblem(
            F,
            solution,
            bcs=bcs,
            petsc_options_prefix="pinn_fluid_fenicsx_navier_stokes_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_rtol": 1.0e-8,
                "snes_atol": 1.0e-8,
                "snes_max_it": 30,
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        problem.solve()
        solution.x.scatter_forward()

    return domain, solution.sub(0).collapse(), solution.sub(1).collapse()


def _sample_function(function: Any, coordinates: np.ndarray) -> np.ndarray:
    from dolfinx import geometry

    domain = function.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim, padding=1.0e-10)
    points = np.column_stack(
        (coordinates[:, 0], coordinates[:, 1], np.zeros(coordinates.shape[0]))
    )
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    cells = []
    active_points = []
    active_indices = []
    for index, point in enumerate(points):
        links = colliding_cells.links(index)
        if len(links) == 0:
            clipped = np.array(
                [
                    min(max(point[0], 1.0e-10), 1.0 - 1.0e-10),
                    min(max(point[1], 1.0e-10), 1.0 - 1.0e-10),
                    0.0,
                ],
                dtype=np.float64,
            )
            retry_candidates = geometry.compute_collisions_points(tree, clipped.reshape(1, 3))
            retry_cells = geometry.compute_colliding_cells(domain, retry_candidates, clipped.reshape(1, 3))
            links = retry_cells.links(0)
            point = clipped
        if len(links) == 0:
            raise RuntimeError(f"could not locate FEniCSx cell for sample point {points[index]}")
        active_points.append(point)
        active_indices.append(index)
        cells.append(links[0])
    function.x.scatter_forward()
    values = np.asarray(function.eval(np.asarray(active_points), np.asarray(cells, dtype=np.int32)))
    output_shape = (coordinates.shape[0], values.shape[1] if values.ndim > 1 else 1)
    sampled = np.zeros(output_shape, dtype=np.float64)
    sampled[np.asarray(active_indices)] = values.reshape(len(active_indices), -1)
    return sampled


def solve_fenicsx_reference(
    *,
    model_name: str,
    output_path: Path | str,
    grid_points: int,
    mesh_cells: int,
    viscosity: float,
    peak_velocity: float,
) -> Path:
    """Solve one model with FEniCSx and write repository-compatible fields."""

    require_fenicsx()
    if model_name == "darcy":
        domain, velocity, pressure = _run_darcy_solve(
            grid_points=grid_points,
            mesh_cells=mesh_cells,
            viscosity=viscosity,
            peak_velocity=peak_velocity,
        )
    elif model_name in {"stokes", "oseen", "navier_stokes"}:
        domain, velocity, pressure = _run_stokes_like_solve(
            model_name=model_name,
            grid_points=grid_points,
            mesh_cells=mesh_cells,
            viscosity=viscosity,
            peak_velocity=peak_velocity,
        )
    else:
        raise ValueError(f"unsupported FEniCSx model: {model_name}")

    if domain.comm.rank != 0:
        return Path(output_path)
    values = np.linspace(0.0, 1.0, grid_points, dtype=np.float64)
    coordinates = np.asarray([(x, y) for x in values for y in values], dtype=np.float64)
    sampled_velocity = _sample_function(velocity, coordinates)
    sampled_pressure = _sample_function(pressure, coordinates)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coordinates=coordinates,
        reference_pressure=sampled_pressure[:, :1],
        reference_u=sampled_velocity[:, :1],
        reference_v=sampled_velocity[:, 1:2],
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FEniCSx reference fields.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve = subparsers.add_parser("solve")
    solve.add_argument("--model", choices=FLOW_MODELS, required=True)
    solve.add_argument("--output-path", required=True)
    solve.add_argument("--grid-points", type=int, required=True)
    solve.add_argument("--mesh-cells", type=int, required=True)
    solve.add_argument("--viscosity", type=float, required=True)
    solve.add_argument("--peak-velocity", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "solve":
        solve_fenicsx_reference(
            model_name=args.model,
            output_path=args.output_path,
            grid_points=args.grid_points,
            mesh_cells=args.mesh_cells,
            viscosity=args.viscosity,
            peak_velocity=args.peak_velocity,
        )
        return 0
    raise ValueError(f"unsupported command: {args.command}")


__all__ = [
    "COORDINATE_CONVENTION_METADATA",
    "FENICSX_REFERENCE_METADATA",
    "FLOW_MODEL_METADATA",
    "FLOW_MODELS",
    "generate_fenicsx_reference",
    "import_fenicsx_sampled_fields",
    "main",
    "require_fenicsx",
    "solve_fenicsx_reference",
]


if __name__ == "__main__":
    raise SystemExit(main())
