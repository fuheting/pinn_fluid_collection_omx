# PINN Fluid Mechanics

Research scaffold for studying Physics-Informed Neural Networks applied to fluid mechanics. The project is organized as a phased repository: begin with a reproducible PyTorch-oriented foundation, then add governing-equation examples from simple porous-media flow through laminar incompressible flow.

## Research Objective

The long-term objective is to evaluate how PINNs converge on fluid mechanics problems when conservation laws and constitutive relations are enforced through loss terms. Each phase should add one problem family, document the governing equations, define verification targets, and preserve a clear connection between the learned field and the physical residuals.

## Current Status

The current phase hardens the Darcy finite-difference actual field before model-specific vector-flow references are introduced. The repository defines model-agnostic inlet/outlet/wall patches, shared deterministic collocation sampling, Darcy residual helpers, Stokes residual helpers, Oseen residual helpers, Navier-Stokes residual helpers, minimal Darcy and Stokes neural fields, lightweight Darcy, Stokes, Oseen, and Navier-Stokes training smoke loops, phase-ordered smoke summaries, a closed-form laminar channel reference, a CFD-style experiment layer that saves predicted fields, reference fields, residual fields, objective histories, component-wise histories, plots, summary metrics, and explicit reference metadata, a manifest-writing procurement surface for staged local runs, standalone figure panels generated from saved `fields.npz` and `history.json` artifacts, boundary-mask diagnostics, pressure-velocity quiver diagnostics, and a documented inventory of the locally procured paper-demo bundle.

## Planned Phases

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Repository scaffold, documentation, PyTorch-oriented dependency direction, importable placeholders, import tests | Complete for scaffold pass |
| 2 | Shared unit-square flow domain and first Darcy-flow residual instance | Residual foundation complete |
| 3 | Stokes flow at low Reynolds number | Residual foundation complete |
| shared foundation | Collocation sampling and Stokes boundary-target mapping | Foundation complete |
| shared foundation | Minimal Darcy and Stokes neural fields | Foundation complete |
| shared foundation | Darcy loss assembly and training smoke loop | Foundation complete |
| shared foundation | Stokes loss assembly and training smoke loop | Foundation complete |
| 4 | Reduced-order Navier-Stokes through Oseen equations | Training smoke foundation complete |
| 5 | Laminar Navier-Stokes examples such as channel or Poiseuille flow | Training smoke foundation complete |
| 6 | Documentation, cleanup, and consolidated test coverage | Smoke benchmark summary complete |
| 7 | Analytic Poiseuille Navier-Stokes benchmark | Residual validation complete |
| 8 | Experiment data schema and component-history recording | Complete |
| 9 | CFD-backed Darcy and shared-patch Navier-Stokes vertical slice | Complete |
| 10 | Stokes/Oseen experiment extension and consolidated report | Complete |
| 11 | Narrow result-procurement runner and manifest | Complete |
| 12 | Figure-generation utilities for saved experiment artifacts | Complete |
| 13 | Local sanity and paper-demo result runs | Complete |
| 14 | Result-procurement documentation, figure inventory, and limitations | Complete |
| 15 | Reference-generation audit and metadata contracts | Complete |
| 16 | Darcy ground truth hardening | Complete |

## Repository Layout

```text
.
├── data/
│   └── .gitkeep
├── src/
│   └── pinn_fluid/
│       ├── __init__.py
│       ├── models/
│       ├── solvers/
│       └── utils/
├── tests/
│   └── test_package_structure.py
├── progress.md
├── pyproject.toml
└── requirements.txt
```

The `src/pinn_fluid` package now contains the shared domain abstraction, Darcy-flow residual utilities, Stokes-flow residual utilities, Oseen residual utilities, Navier-Stokes residual utilities, minimal neural field modules, smoke-training utilities for Darcy, Stokes, Oseen, and Navier-Stokes flow, a consolidated smoke benchmark summary module, and deterministic experiment orchestration in `pinn_fluid.experiments`.

## Shared Example Domain

All flow models should start from the same unit-square example unless a later phase explicitly introduces a new benchmark:

```text
Omega = [0, 1] x [0, 1]
```

The coordinate convention is Cartesian: `x` increases left-to-right, `y=0` is the bottom boundary, and `y=1` is the top boundary. The shared inlet is therefore the top-left segment (`x in [0, 0.25]`, `y=1`), and the shared outlet is the bottom-right segment (`x in [0.75, 1]`, `y=0`). Plotting code uses this same convention with `origin="lower"`.

Boundary patches are model-agnostic dictionaries:

```python
boundary_patches = {
    "inlet": {
        "location": "top",
        "x_range": [0.0, 0.25],
        "type": "dirichlet",
        "variable": "pressure",
        "value": 1.0,
    },
    "outlet": {
        "location": "bottom",
        "x_range": [0.75, 1.0],
        "type": "dirichlet",
        "variable": "pressure",
        "value": 0.0,
    },
    "walls": {
        "type": "neumann",
        "condition": "no_normal_flow",
    },
}
```

For Darcy flow with constant permeability `K = 1`, the governing equation reduces to:

```text
Delta p = 0
```

The Phase 2 implementation includes:

- pressure field module `p_theta(x, y)`
- interior residual `p_xx + p_yy`
- Darcy velocity `u = -grad(p)`
- inlet residual `p - 1`
- outlet residual `p`
- wall residual `grad(p) dot n`
- default loss weights: interior `1`, inlet `10`, outlet `10`, wall `1`
- deterministic interior and boundary collocation samples from the shared unit-square patches
- weighted loss assembly and a small deterministic optimization smoke loop

## Stokes Residual Foundation

For steady incompressible Stokes flow with velocity `(u, v)`, pressure `p`, and constant viscosity `mu`, Phase 3 defines:

```text
continuity = u_x + v_y
x_momentum = -p_x + mu * (u_xx + u_yy)
y_momentum = -p_y + mu * (v_xx + v_yy)
```

The current Stokes implementation includes:

- velocity-pressure field module for `(u, v, p)`
- continuity residual
- x- and y-momentum residuals
- no-slip wall residual helper
- Stokes boundary targets derived from the shared inlet, outlet, and wall patches
- default loss weights: continuity `1`, x-momentum `1`, y-momentum `1`, inlet `10`, outlet `10`, wall `10`
- weighted loss assembly and a small deterministic optimization smoke loop

## Oseen Residual Foundation

For steady incompressible Oseen flow with velocity `(u, v)`, pressure `p`, viscosity `mu`, and prescribed convection velocity `beta = (beta_x, beta_y)`, Phase 4 defines:

```text
continuity = u_x + v_y
x_momentum = beta dot grad(u) - p_x + mu * (u_xx + u_yy)
y_momentum = beta dot grad(v) - p_y + mu * (v_xx + v_yy)
```

The Oseen residual helpers reduce to the Stokes residual helpers when `beta = 0`.

The current Oseen training implementation includes:

- velocity-pressure field reuse through `StokesVelocityPressureField`
- continuity and momentum residual loss assembly with a prescribed convection velocity
- inlet velocity, outlet pressure, and no-slip wall boundary losses from the shared patches
- default loss weights: continuity `1`, x-momentum `1`, y-momentum `1`, inlet `10`, outlet `10`, wall `10`
- weighted loss assembly and a small deterministic optimization smoke loop

## Collocation Sampling Foundation

Shared sampling utilities now provide deterministic PyTorch tensors for the existing unit-square benchmark:

- `interior_collocation_points(points_per_axis)` returns an evenly spaced interior grid that excludes the boundary.
- `boundary_collocation_points(points_per_patch)` returns inlet and outlet coordinates plus wall coordinates and outward normals.

The wall samples reuse the existing inlet/outlet patch geometry to leave the inlet and outlet openings available for model-specific boundary conditions.

## Neural Field Foundation

Minimal PyTorch neural fields now provide the learnable surfaces that later training code will optimize:

- `MLPField` maps two-dimensional coordinates to a configurable output width.
- `DarcyPressureField` maps `(x, y)` to scalar pressure.
- `StokesVelocityPressureField` maps `(x, y)` to named `u`, `v`, and `pressure` tensors.

These modules intentionally do not assemble losses, run optimizers, or claim convergence behavior.

## Darcy Training Smoke Foundation

Darcy training utilities now connect the shared collocation samplers, pressure field, residual helpers, and default loss weights:

- `darcy_loss_components(...)` returns interior, inlet, outlet, and wall mean-squared residual losses.
- `darcy_total_loss(...)` applies the Phase 2 default weights.
- `train_darcy_smoke(...)` runs a short local optimizer loop and returns loss history for regression tests.

The smoke loop only checks that the training path is wired and can reduce its own residual loss on a tiny collocation set. It is not a convergence study or a validated numerical benchmark.

## Stokes Training Smoke Foundation

Stokes training utilities now connect the shared collocation samplers, velocity-pressure field, residual helpers, no-slip wall helper, and default loss weights:

- `stokes_loss_components(...)` returns continuity, momentum, inlet, outlet, and wall mean-squared residual losses.
- `stokes_total_loss(...)` applies the Phase 3 default weights.
- `train_stokes_smoke(...)` runs a short local optimizer loop and returns loss history for regression tests.

The smoke loop checks training-path wiring and loss reduction on a tiny collocation set. It is not a validated Stokes benchmark or convergence study.

## Oseen Training Smoke Foundation

Oseen training utilities now connect the shared collocation samplers, velocity-pressure field, Oseen residual helpers, no-slip wall helper, and default loss weights:

- `oseen_loss_components(...)` returns continuity, momentum, inlet, outlet, and wall mean-squared residual losses.
- `oseen_total_loss(...)` applies the Phase 4 default weights.
- `train_oseen_smoke(...)` runs a short local optimizer loop and returns loss history for regression tests.

The smoke loop checks training-path wiring and loss reduction on a tiny collocation set. It is not a validated Oseen benchmark or convergence study.

## Navier-Stokes Residual Foundation

For steady incompressible Navier-Stokes flow with velocity `(u, v)`, pressure `p`, and constant viscosity `mu`, Phase 5 defines:

```text
continuity = u_x + v_y
x_momentum = u * u_x + v * u_y - p_x + mu * (u_xx + u_yy)
y_momentum = u * v_x + v * v_y - p_y + mu * (v_xx + v_yy)
```

The current Navier-Stokes implementation includes:

- continuity residual
- x- and y-momentum residuals with nonlinear self-advection
- tests showing zero velocity with constant pressure has zero residuals
- tests showing the residuals match Oseen residuals when the Oseen convection velocity is the current velocity
- default loss weights matching the current Stokes/Oseen boundary and residual weighting shape
- weighted loss assembly from shared interior and boundary collocation samples
- a small deterministic optimization smoke loop that verifies local loss reduction

The smoke loop checks training-path wiring and loss reduction on a tiny collocation set. It is not a validated Navier-Stokes benchmark or convergence study.

## Consolidated Smoke Benchmark Summary

Phase 6 adds `pinn_fluid.benchmarks` as a lightweight reporting surface across the implemented model families:

- `SmokeBenchmarkConfig` defines a tiny deterministic run shape shared by Darcy, Stokes, Oseen, and Navier-Stokes smoke loops.
- `run_smoke_benchmarks(...)` returns phase-ordered `SmokeBenchmarkResult` entries for Darcy, Stokes, Oseen, and Navier-Stokes.
- `summarize_loss_history(...)` converts a loss history into initial loss, final loss, step count, absolute reduction, fractional reduction, and reduction status.

These summaries are only wiring checks for local regression tests. They do not replace validated physical benchmark cases or convergence studies.

## Analytic Poiseuille Benchmark

The Navier-Stokes model utilities now include a closed-form horizontal channel reference on the unit square:

```text
u(x, y) = 4 U y (1 - y)
v(x, y) = 0
p(x, y) = p0 - 8 mu U x
```

`poiseuille_channel_solution(...)` returns differentiable `u`, `v`, and `pressure` tensors for this profile. `poiseuille_pressure_drop(...)` returns the corresponding pressure drop over a channel length. Tests verify that the profile satisfies the steady incompressible Navier-Stokes residuals and horizontal wall no-slip behavior.

This is an analytic residual benchmark, not a trained PINN comparison against the analytic solution.

## CFD-Backed Experiment Layer

Phases 8-10 add `pinn_fluid.experiments` for lightweight local comparisons as physics complexity increases:

- `TrainingHistory` records total objective values and per-loss-component histories for every optimizer iteration.
- `ExperimentResult` stores model name, reference source, grid shape, metrics, and artifact paths.
- `ExperimentResult` stores reference metadata with the generator name, PDE model represented by the reference, boundary condition type, coordinate convention, and reference kind.
- `run_darcy_experiment(...)` trains a small Darcy pressure field and compares it to an in-repo finite-difference Laplace reference on a deterministic grid. The Darcy actual field now exposes pressure, `u`, `v`, speed, and a finite-difference Laplace residual diagnostic.
- `run_poiseuille_navier_stokes_experiment(...)` keeps the historical API name but now trains a small velocity-pressure field on the same shared unit-square patch geometry as Darcy: top-left inlet, bottom-right outlet, and walls.
- `run_stokes_experiment(...)` and `run_oseen_experiment(...)` use the same shared-patch boundary setup so the vector-flow comparisons are no longer horizontal channel-flow artifacts.
- The vector experiments save reference `u`, `v`, and pressure fields from the shared-patch finite-difference pressure/velocity reference used to make the Darcy geometry visible in all model panels.
- Current reference metadata is:

| Model | Reference generator | PDE represented | Reference kind |
| --- | --- | --- | --- |
| Darcy | `fd_darcy_reference` | Darcy pressure Laplace equation | finite-difference |
| Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Oseen | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Navier-Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |

All four references use the Cartesian unit-square convention: `x` increases left-to-right, `y=0` is bottom, `y=1` is top, the inlet is the top-left horizontal patch, and the outlet is the bottom-right horizontal patch. Grid coordinates flatten in x-major order with y varying fastest, while plotting reshapes fields with `reshape(grid_points, grid_points).T` and `origin="lower"`.
- `run_all_experiments(...)` runs Darcy, Stokes, Oseen, and Navier-Stokes in order and writes a consolidated JSON and Markdown report.

Each experiment writes reproducible numeric artifacts under the configured output directory:

- `fields.npz` with grid coordinates, PINN-predicted fields, reference fields, and residual fields
- `fields.npz` also includes `reference_metadata_json` so each field bundle records its reference source and coordinate convention
- Darcy `fields.npz` includes `reference_pressure`, `reference_velocity`, `reference_u`, `reference_v`, `reference_speed`, `reference_residual`, `predicted_u`, `predicted_v`, `predicted_speed`, and the PINN residual field `residual`
- `history.json` with total and component-wise objective histories
- `metrics.json` with finite L2/RMS comparison metrics
- loss-history and field plot artifacts

The first-pass success criterion is intentionally modest: histories should decrease over the tiny deterministic run and reported metrics should be finite. These runs are reproducible local experiment slices, not final accuracy claims.

## Result Procurement Handoff

Phase 11 adds `pinn_fluid.result_procurement.run_result_procurement(...)` and a `python -m pinn_fluid.result_procurement` entry point. The runner wraps the existing all-model experiment API, writes results under the configured output directory, and saves `run_manifest.json` with the git commit, config values, per-model output directories, reference metadata, artifact paths, metric summaries, finite-metric flags, and history-reduction flags.

Example sanity procurement command:

```bash
PYTHONPATH=src python -m pinn_fluid.result_procurement \
  --output-dir data/experiments_sanity \
  --grid-points 9 \
  --training-steps 80 \
  --hidden-width 12 \
  --hidden-layers 1 \
  --learning-rate 0.02 \
  --git-dir /tmp/pinn_fluid_collection_omx.gitdir \
  --work-tree /home/hfu_nestle/projects/pinn_fluid_omx
```

Use the existing experiment API and this procurement runner to reproduce the Phase 13 paper-draft figures and metrics. Keep generated run outputs local unless the artifact policy changes explicitly.

## Figure Generation Handoff

Phase 12 adds `pinn_fluid.figures.generate_figure_bundle(...)` and a `python -m pinn_fluid.figures` entry point for generating paper-draft panels from saved experiment artifacts. It reads each available model directory under an experiment output root, expects `fields.npz` and `history.json`, writes panel PNGs under `figures/`, and records the generated paths in `figures/figure_manifest.json`.

Example command after a procurement run:

```bash
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

Generated Phase 12 figures:

- Darcy field panel: row-labeled pressure, `u`, and `v` comparisons with predicted, actual, and residual columns.
- Stokes/Oseen/Navier-Stokes field panels: row-labeled `u`, `v`, and pressure comparisons with predicted, actual, and residual columns.
- Field comparison panels place bold row labels on the left and bold predicted/actual/residual column labels below the columns.
- Separate scalar field images for Stokes/Oseen/Navier-Stokes predicted, actual, and residual/error `u`, `v`, pressure, speed, and residual-magnitude fields.
- Per-model convergence panels: total objective and every recorded component loss versus iteration.
- Flow-field figures use the `turbo` colormap, include value colorbars with min/mid/max tick labels, and mark the shared inlet in red just above the top-left horizontal opening and the shared outlet in blue just below the bottom-right horizontal opening so the field values remain unobscured.
- Predicted and actual fields for the same variable share the same color range; residual/error panels keep their own residual range.
- Per-model pressure-velocity quiver diagnostics overlay velocity arrows on pressure contours so the plotted flow direction can be checked against the red inlet and blue outlet.
- Convergence figures include axis labels, log-scaled objective values, and a titled legend for loss components.
- Figure generation imports `matplotlib` directly; install `matplotlib>=3.8` with `python -m pip install -r requirements.txt` if the import is missing.
- Figure manifests include field-panel row/column layout, shared predicted/actual color limits, inlet/outlet marker metadata, pressure-velocity quiver paths, panel titles, convergence legend entries, convergence axes, colormap names, and scalar colorbar tick values.

## Boundary And Flow Diagnostics

Use `pinn_fluid.diagnostics` to verify the boundary masks before interpreting a result bundle:

```bash
PYTHONPATH=src python -m pinn_fluid.diagnostics \
  data/boundary_diagnostics.png \
  --report data/boundary_diagnostics.json \
  --interior-points-per-axis 9 \
  --boundary-points-per-patch 9
```

The JSON report prints min/max coordinates and counts for `inlet`, `outlet`, `walls`, and `interior`. Expected values for the default unit square are top-left inlet coordinates with `x` in `[0, 0.25]` and `y=1`, plus bottom-right outlet coordinates with `x` in `[0.75, 1]` and `y=0`.

After generating a figure bundle, inspect:

- `figures/{model}_pressure_velocity_quiver.png` for pressure contours with velocity arrows.
- `figures/figure_manifest.json` for `field_panel_color_limits`, `separate_field_images[*].color_limits`, and `boundary_markers`.

If extrema still appear physically suspicious after these checks, treat the `matplotlib` figures as Cartesian `y` up: `origin="lower"` places `y=1` at the top of the physical domain and `y=0` at the bottom.

## Phase 13 Local Runs

Phase 13 procured two ignored local output tiers with the Phase 11 runner and Phase 12 figure bundle utility. The manifests record git commit `6c92946`, which was the local Phase 12 commit available when the temporary gitdir was unavailable in this shell.

Sanity tier command:

```bash
PYTHONPATH=src python -m pinn_fluid.result_procurement \
  --output-dir data/experiments_sanity \
  --grid-points 9 \
  --training-steps 80 \
  --hidden-width 12 \
  --hidden-layers 1 \
  --learning-rate 0.02 \
  --seed 0 \
  --viscosity 0.25 \
  --peak-velocity 1.0 \
  --darcy-reference-iterations 400 \
  --git-commit 6c92946
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

Paper-demo tier command:

```bash
PYTHONPATH=src python -m pinn_fluid.result_procurement \
  --output-dir data/experiments_paper_demo \
  --grid-points 31 \
  --training-steps 800 \
  --hidden-width 24 \
  --hidden-layers 2 \
  --learning-rate 0.01 \
  --seed 0 \
  --viscosity 0.25 \
  --peak-velocity 1.0 \
  --darcy-reference-iterations 1200 \
  --git-commit 6c92946
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_paper_demo
```

The first paper-demo run reduced every model history, so the fallback `learning_rate=0.005` and `training_steps=1200` rerun was not used.

Paper-demo metrics:

| Model | Final objective | Velocity L2 | Pressure L2 | Residual RMS |
| --- | ---: | ---: | ---: | ---: |
| Darcy | 0.119401 | 0.319252 | 0.211299 | 0.118235 |
| Stokes | 0.000218439 | 0.000880394 | 0.00243016 | 0.0304391 |
| Oseen | 0.000276085 | 0.00149186 | 0.00306 | 0.0350645 |
| Navier-Stokes | 0.000277937 | 0.00111077 | 0.00200808 | 0.0370115 |

Generated local artifact roots:

- `data/experiments_sanity/`
- `data/experiments_paper_demo/`

Each root contains `run_manifest.json`, per-model `fields.npz`, `history.json`, `metrics.json`, raw single-field plots, `summary/cross_model_report.json`, `summary/cross_model_report.md`, and Phase 12 panels under `figures/`.

## Phase 14 Figure Inventory And Limits

The Phase 13 paper-demo bundle in `data/experiments_paper_demo/` satisfies the minimum paper-demo figure inventory without committing generated files. The same file pattern exists for `data/experiments_sanity/`. Regenerate the bundle with `PYTHONPATH=src python -m pinn_fluid.figures data/experiments_paper_demo` after changing figure code. Flow-field comparison panels use predicted/actual/residual columns with bold bottom column labels and bold left row labels; Darcy now uses pressure, `u`, and `v` rows rather than velocity magnitude. Flow-field figures use `turbo` with numeric colorbar values, shared predicted/actual color limits per variable, and red/blue inlet/outlet boundary markers placed just outside the top and bottom horizontal openings. Convergence figures label the iteration and log-objective axes and include a titled loss-component legend. Figure generation requires `matplotlib>=3.8`; without it, Python raises the normal import error. Pressure-velocity quiver panels are generated for every model to show whether velocity arrows move from the red inlet toward the blue outlet.

Paper-demo figure inventory:

| Model | Field panel | Convergence panel |
| --- | --- | --- |
| Darcy | `data/experiments_paper_demo/figures/darcy_fields.png` | `data/experiments_paper_demo/figures/darcy_convergence.png` |
| Stokes | `data/experiments_paper_demo/figures/stokes_fields.png` | `data/experiments_paper_demo/figures/stokes_convergence.png` |
| Oseen | `data/experiments_paper_demo/figures/oseen_fields.png` | `data/experiments_paper_demo/figures/oseen_convergence.png` |
| Navier-Stokes | `data/experiments_paper_demo/figures/navier_stokes_fields.png` | `data/experiments_paper_demo/figures/navier_stokes_convergence.png` |

Pressure-velocity diagnostic panels:

- `data/experiments_paper_demo/figures/{darcy,stokes,oseen,navier_stokes}_pressure_velocity_quiver.png`

Paper-demo numeric and report artifacts:

- `data/experiments_paper_demo/run_manifest.json`
- `data/experiments_paper_demo/figures/figure_manifest.json`
- `data/experiments_paper_demo/summary/cross_model_report.json`
- `data/experiments_paper_demo/summary/cross_model_report.md`
- `data/experiments_paper_demo/{darcy,stokes,oseen,navier_stokes}/fields.npz`
- `data/experiments_paper_demo/{darcy,stokes,oseen,navier_stokes}/history.json`
- `data/experiments_paper_demo/{darcy,stokes,oseen,navier_stokes}/metrics.json`

Separate paper-demo scalar images:

- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_predicted_u.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_actual_u.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_residual_u.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_predicted_v.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_actual_v.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_residual_v.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_predicted_p.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_actual_p.png`
- `data/experiments_paper_demo/figures/{stokes,oseen,navier_stokes}_residual_p.png`

Minimum paper-demo figure bundle contents:

- Darcy panel: predicted pressure, reference pressure, pressure error, predicted and reference `u`, predicted and reference `v`, and component-wise velocity errors.
- Stokes/Oseen/Navier-Stokes panels: predicted `u`, `v`, speed, and pressure; reference `u`, `v`, speed, and pressure; scalar residual magnitude.
- Convergence panels: total objective and every recorded loss component versus iteration for each model.
- Cross-model table: final objective, velocity L2, pressure L2, and residual RMS.

Known limitations:

- The paper-demo tier is laptop-moderate procurement evidence, not a final convergence study.
- The references are lightweight in-repo shared-patch finite-difference fields, not external CFD validation data.
- Stokes, Oseen, and Navier-Stokes now use the same top-left inlet and bottom-right outlet geometry as Darcy, but their shared-patch vector reference remains a paper-demo procurement reference rather than a high-fidelity CFD solve.
- Cross-model velocity-pressure metrics should not be read as a general model ranking.
- The run used one seed and one training budget; no uncertainty, sensitivity, or hyperparameter sweep is reported.
- Generated `.npz`, `.json`, and `.png` artifacts are intentionally ignored under `data/` and are not part of the committed source tree.

Keep generated outputs under ignored local paths such as `data/experiments_sanity/` and `data/experiments_paper_demo/`. Commit source, tests, and documentation only; do not commit generated `.npz`, `.json`, or `.png` experiment artifacts unless a future phase explicitly changes that policy.

## Phase 15 Reference-Generation Contracts

Phase 15 makes the current reference behavior explicit without changing the physics. The new contract tests document `_fd_darcy_reference(...)`, the shared Darcy-derived vector reference, coordinate flattening and plotting orientation, and the reference metadata saved by experiments and procurement manifests.

Reference metadata is now saved in:

- `ExperimentResult.reference_metadata`
- `fields.npz` as `reference_metadata_json`
- `run_manifest.json` entries as `reference_metadata`
- `summary/cross_model_report.json` through `ExperimentResult.to_json_dict()`

The current limitation remains intentional: Darcy uses a finite-difference Laplace reference, while Stokes, Oseen, and Navier-Stokes still use a demo-only vector field derived from that Darcy reference. Phase 15 does not make Stokes, Oseen, or Navier-Stokes ground truth physically model-specific; it locks the existing behavior so Phases 17-19 can replace those references deliberately.

Verification commands for this phase:

```bash
python -m pytest tests/test_phase15_reference_contracts.py
python -m pytest
git diff --check
```

Artifact and figure regeneration still use the existing commands:

```bash
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --git-commit <commit>
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

## Phase 16 Darcy Ground Truth Hardening

Phase 16 keeps the shared unit-square top-left inlet and bottom-right outlet geometry and hardens only the Darcy actual field. The finite-difference Darcy reference remains `fd_darcy_reference`, representing the Darcy pressure Laplace equation with pressure Dirichlet inlet/outlet patches and no-normal-flow walls.

The hardened Darcy contracts verify:

- pressure extrema occur on the inlet and outlet pressure patches
- Darcy velocity points from the high-pressure top-left inlet toward the low-pressure bottom-right outlet
- wall-normal velocity is zero away from inlet/outlet openings
- saved Darcy actual fields include pressure, `u`, `v`, speed, and finite-difference residual diagnostics
- Darcy figure generation remains compatible with the explicit component fields

Reference generators after this phase remain:

| Model | Reference generator | PDE represented | Reference kind |
| --- | --- | --- | --- |
| Darcy | `fd_darcy_reference` | Darcy pressure Laplace equation | finite-difference |
| Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Oseen | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Navier-Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |

Limitations remain narrow: Phase 16 does not introduce Stokes, Oseen, or Navier-Stokes model-specific actual fields, and it does not claim external CFD validation for the Darcy finite-difference solve.

Verification and reproduction commands:

```bash
python -m pytest tests/test_phase16_darcy_ground_truth.py
python -m pytest
git diff --check
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --git-commit <commit>
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

## Environment Direction

PyTorch is the intended machine-learning framework for future phases. Phase 1 tests only check package structure and import behavior; they do not require importing PyTorch.

Install dependencies in a virtual environment when implementation work begins:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the current tests:

```bash
python -m pytest
```

## Continuing The Model Phases

The remaining research work is to run broader, longer experiments with selected grid sizes and training budgets, then analyze how the reported metrics change with the enforced physics model. Keep those studies separate from the lightweight regression-oriented defaults in `pinn_fluid.experiments`.

New agents should start with this `README.md` and `progress.md`. Continue the established style: tests first, narrow implementation, docs/progress update, fresh verification, Lore commit, then push `origin main` with the temp gitdir/worktree command when needed. Split the result-procurement work into staged phases rather than trying to produce the full study in one pass.
