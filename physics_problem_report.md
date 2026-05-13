# Physics Problem Report

Generated: 2026-05-12

This report summarizes the current apples-to-apples PINN comparison problem in this repository. It is based on the source code and the latest regenerated FEniCSx comparison bundle:

`data/fenicsx_consistent_comparison_2026_05_11/`

## Objective

The repository compares Physics-Informed Neural Network surrogates against model-specific finite-element reference fields for four steady two-dimensional flow models:

| Model | PINN unknowns | Reference generator | Reference PDE represented |
| --- | --- | --- | --- |
| Darcy | `p` with velocity recovered as `-grad(p)` | `fenicsx_darcy_shared_domain` | Darcy porous-flow pressure equation |
| Stokes | `u`, `v`, `p` | `fenicsx_stokes_shared_domain` | Incompressible linear momentum and continuity |
| Oseen | `u`, `v`, `p` | `fenicsx_oseen_shared_domain` | Incompressible linearized momentum and continuity |
| Navier-Stokes | `u`, `v`, `p` | `fenicsx_navier_stokes_shared_domain` | Steady incompressible nonlinear momentum and continuity |

The comparison run uses dedicated FEniCSx ground truth for each model. The active run does not use OpenFOAM, manufactured vector fields, or a shared reference sample across models.

## Domain

All models use the same unit-square geometry:

- Domain: `(x, y) in [0, 1] x [0, 1]`
- Coordinate convention: mathematical coordinates
- `x` increases left-to-right
- `y = 0` is the bottom edge
- `y = 1` is the top edge
- Flattening order: x-major, with y varying fastest from bottom to top
- Plotting order: fields are reshaped with `reshape(grid_points, grid_points).T` and shown with `origin="lower"`

Boundary patches:

| Patch | Location | Geometry |
| --- | --- | --- |
| Inlet | Top-left horizontal segment | `x in [0.0, 0.25]`, `y = 1` |
| Outlet | Bottom-right horizontal segment | `x in [0.75, 1.0]`, `y = 0` |
| Walls | Remaining unit-square perimeter | Left wall, right wall, top edge outside inlet, bottom edge outside outlet |

## Boundary Conditions

The active FEniCSx references use a common boundary-condition convention for all four models:

- Inlet: prescribed velocity `(u, v) = (0, -1)`
- Outlet: prescribed pressure `p = 0`
- Walls: no-slip velocity `(u, v) = (0, 0)` for Stokes, Oseen, and Navier-Stokes
- Darcy reference: pressure solve with outlet `p = 0`; velocity is recovered from `-grad(p)` and the inlet loading is applied as a flux consistent with downward velocity

The PINN losses use matching boundary targets:

| Model family | Inlet term | Outlet term | Wall term |
| --- | --- | --- | --- |
| Darcy | Darcy velocity `-grad(p)` targets `(0, -peak_velocity)` | pressure target `p = 0` | no-normal-flow wall flux |
| Stokes/Oseen/Navier-Stokes | velocity targets `(0, -peak_velocity)` | pressure target `p = 0` | no-slip velocity target `(0, 0)` |

For the current comparison run, `peak_velocity = 1.0`.

## Governing Residuals Used By The PINNs

### Darcy

The Darcy PINN learns pressure `p_theta(x, y)`. The interior residual is the pressure Laplacian:

```text
p_xx + p_yy = 0
```

The Darcy velocity used for diagnostics and boundary terms is:

```text
u_D = -grad(p)
```

Loss components:

- `interior`: mean square Laplace residual
- `inlet`: mean square mismatch between Darcy velocity and `(0, -peak_velocity)`
- `outlet`: mean square outlet pressure
- `wall`: mean square normal pressure-gradient flux on walls

Loss weights:

| Component | Weight |
| --- | ---: |
| `interior` | 1.0 |
| `inlet` | 10.0 |
| `outlet` | 10.0 |
| `wall` | 1.0 |

### Stokes

The Stokes PINN learns `(u, v, p)`. Residuals:

```text
u_x + v_y = 0
-p_x + nu * Laplacian(u) = 0
-p_y + nu * Laplacian(v) = 0
```

Loss components:

- `continuity`
- `x_momentum`
- `y_momentum`
- `inlet`
- `outlet`
- `wall`

Loss weights:

| Component | Weight |
| --- | ---: |
| `continuity` | 1.0 |
| `x_momentum` | 1.0 |
| `y_momentum` | 1.0 |
| `inlet` | 10.0 |
| `outlet` | 10.0 |
| `wall` | 10.0 |

### Oseen

The Oseen PINN learns `(u, v, p)` and uses a fixed convection velocity. Residuals:

```text
u_x + v_y = 0
(beta . grad) u - p_x + nu * Laplacian(u) = 0
(beta . grad) v - p_y + nu * Laplacian(v) = 0
```

Current convection velocity:

```text
beta = (1.0, 0.0)
```

The Oseen loss uses the same component names and weights as Stokes.

### Navier-Stokes

The Navier-Stokes PINN learns `(u, v, p)`. Residuals:

```text
u_x + v_y = 0
(U . grad) u - p_x + nu * Laplacian(u) = 0
(U . grad) v - p_y + nu * Laplacian(v) = 0
```

where:

```text
U = (u, v)
```

The Navier-Stokes loss uses the same component names and weights as Stokes.

## PINN Parameters

The latest comparison run used one common PINN setup for every model:

| Parameter | Value |
| --- | ---: |
| `grid_points` | 17 |
| Prediction/sample grid | `17 x 17 = 289` points |
| Training steps | 300 |
| Hidden width | 32 |
| Hidden layers | 2 |
| Activation | `torch.nn.Tanh` |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Random seed | 20260511 |
| Viscosity `nu` | 0.25 |
| Peak inlet velocity | 1.0 |
| Reference source | `fenicsx` |

Network outputs:

| Model | Network architecture |
| --- | --- |
| Darcy | MLP input `(x, y)`, scalar output `p` |
| Stokes | MLP input `(x, y)`, vector output `(u, v, p)` |
| Oseen | MLP input `(x, y)`, vector output `(u, v, p)` |
| Navier-Stokes | MLP input `(x, y)`, vector output `(u, v, p)` |

Collocation sampling for the current run:

| Sample set | Rule | Current count |
| --- | --- | ---: |
| Interior | `max(2, grid_points - 2)` points per axis | `15 x 15 = 225` |
| Inlet | `max(2, grid_points // 2)` points on inlet patch | 8 |
| Outlet | `max(2, grid_points // 2)` points on outlet patch | 8 |
| Walls | same points per wall segment, four wall segments | 32 |

The wall set includes the left wall, right wall, bottom wall outside the outlet, and top wall outside the inlet.

## FEniCSx Reference Parameters

The latest run generated one model-specific FEniCSx artifact per model:

| Model | Reference artifact |
| --- | --- |
| Darcy | `data/fenicsx_consistent_comparison_2026_05_11/fenicsx_references/darcy/fields.npz` |
| Stokes | `data/fenicsx_consistent_comparison_2026_05_11/fenicsx_references/stokes/fields.npz` |
| Oseen | `data/fenicsx_consistent_comparison_2026_05_11/fenicsx_references/oseen/fields.npz` |
| Navier-Stokes | `data/fenicsx_consistent_comparison_2026_05_11/fenicsx_references/navier_stokes/fields.npz` |

Common FEniCSx generation parameters:

| Parameter | Value |
| --- | ---: |
| System Python used for FEniCSx generation | `/usr/bin/python3` |
| Output sample grid | `17 x 17` |
| Mesh cells per direction | 16 |
| Mesh type | unit-square triangular mesh from FEniCSx `mesh.create_unit_square` |
| Viscosity `nu` | 0.25 |
| Peak inlet velocity | 1.0 |
| Required sample arrays | `coordinates`, `reference_pressure`, `reference_u`, `reference_v` |

Model-specific FEniCSx formulations:

| Model | Formulation |
| --- | --- |
| Darcy | CG1 pressure solve; velocity projected from `-grad(p)` |
| Stokes | Taylor-Hood P2-P1 mixed finite-element solve |
| Oseen | Taylor-Hood P2-P1 mixed finite-element solve with `beta = (1.0, 0.0)` |
| Navier-Stokes | Taylor-Hood P2-P1 nonlinear finite-element solve |

Linear/nonlinear solver settings in the current implementation:

| Model group | Solver configuration |
| --- | --- |
| Darcy pressure and velocity projection | PETSc `ksp_type=preonly`, `pc_type=lu` |
| Stokes and Oseen | PETSc `ksp_type=preonly`, `pc_type=lu` |
| Navier-Stokes | PETSc SNES Newton line-search, `snes_rtol=1e-8`, `snes_atol=1e-8`, `snes_max_it=30`, direct LU linear solve |

## Run Artifacts

The current comparison bundle reports:

| Item | Status |
| --- | --- |
| All histories decreased | `true` |
| All metrics finite | `true` |
| Figure bundle | `data/fenicsx_consistent_comparison_2026_05_11/figures/` |
| Manifest | `data/fenicsx_consistent_comparison_2026_05_11/run_manifest.json` |

Each model output directory contains:

- `fields.npz`
- `history.json`
- `metrics.json`
- `loss_history.png`
- a quick predicted-field PNG

The figure bundle contains comparison panels, separate predicted/actual/residual scalar images, pressure-velocity quiver diagnostics, and convergence histories.

## Current Metrics

| Model | Final loss | Initial loss | Pressure L2 | Velocity L2 | PINN residual RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Darcy | 0.1792228818 | 5.7702226639 | 0.1652421232 | 0.2759495997 | 0.1395752877 |
| Stokes | 0.6091469526 | 5.5509572029 | 3.9031886273 | 0.1897401529 | 0.4435284734 |
| Oseen | 0.5200714469 | 5.5662965775 | 3.9120795754 | 0.1939750868 | 0.6595875025 |
| Navier-Stokes | 0.6155582666 | 5.5520763397 | 4.1132287190 | 0.1853255782 | 0.5439361334 |

Reference residual diagnostics are post-processed on sampled fields and should not be interpreted as native variational residual norms.

## Reproduction Commands

Run the current comparison:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pinn_fluid.result_procurement \
  --output-dir data/fenicsx_consistent_comparison_2026_05_11 \
  --grid-points 17 \
  --training-steps 300 \
  --hidden-width 32 \
  --hidden-layers 2 \
  --learning-rate 0.01 \
  --seed 20260511 \
  --viscosity 0.25 \
  --peak-velocity 1.0 \
  --git-commit fenicsx-consistent-comparison-2026-05-11
```

Regenerate figures:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-figures python -m pinn_fluid.figures data/fenicsx_consistent_comparison_2026_05_11
```

Run the verification suite:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-figures python -m pytest -q
git diff --check
```

## Limitations

- The run is a controlled comparison, not a publication-scale convergence study.
- FEniCSx references are generated on a modest `16 x 16` cell mesh and sampled onto a `17 x 17` grid.
- The current Oseen reference uses `beta = (1.0, 0.0)`, while the inlet velocity is vertical `(0, -1)`. This is documented in metadata and should be revisited if the intended Oseen benchmark should linearize around the inlet direction.
- Darcy and incompressible vector models share geometry and loading conventions, but Darcy remains a porous-flow pressure model while Stokes/Oseen/Navier-Stokes are velocity-pressure models.
- Figure residual panels compare sampled predicted and reference fields; they are visualization diagnostics, not formal error estimates.
- Generated artifacts under `data/` are ignored by git and should not be committed unless explicitly requested.
