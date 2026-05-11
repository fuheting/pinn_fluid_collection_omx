# Progress

## Phase 1: Repository Scaffold

Status: complete for the Phase 1 scaffold pass.

Completed in this pass:

- Established research-oriented project documentation.
- Declared PyTorch as the future machine-learning framework direction.
- Added a minimal `src/` package layout with inert placeholders for future model, solver, and utility work.
- Added a retained `data/` directory placeholder.
- Added lightweight import tests for the placeholder package.
- Added project metadata and ignore rules for a Python research repository.

Explicitly not completed:

- No fluid mechanics model has been implemented.
- No neural network, residual loss, solver, training loop, or numerical validation exists.
- No CI setup exists.

## Remaining Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 2 | Darcy flow domain, residual formulation, training script, and validation tests | Residual foundation complete |
| 3 | Stokes flow PINN formulation and tests | Residual foundation complete |
| shared foundation | Collocation sampling and Stokes boundary-target mapping | Foundation complete |
| shared foundation | Minimal Darcy and Stokes neural fields | Foundation complete |
| shared foundation | Darcy loss assembly and training smoke loop | Foundation complete |
| shared foundation | Stokes loss assembly and training smoke loop | Foundation complete |
| 4 | Oseen equation formulation and convergence checks | Training smoke foundation complete |
| 5 | Laminar Navier-Stokes formulation and common channel-flow cases | Training smoke foundation complete |
| 6 | Documentation consolidation and cleanup across implemented models | Smoke benchmark summary complete |
| 7 | Analytic Poiseuille Navier-Stokes benchmark | Residual validation complete |
| 8 | Experiment data schema and component-history recording | Complete |
| 9 | CFD-backed Darcy and shared-patch Navier-Stokes vertical slice | Complete |
| 10 | Stokes/Oseen experiment extension and consolidated report | Complete |
| 11 | Result-procurement runner and manifest | Complete |
| 12 | Figure-generation utilities for saved experiment artifacts | Complete |
| 13 | Local sanity and paper-demo result runs | Complete |
| 14 | Result-procurement documentation, figure inventory, and limitations | Complete |
| 15 | Reference-generation audit and metadata contracts | Complete |
| 16 | Darcy ground truth hardening | Complete |
| 17 | Stokes model-specific manufactured ground truth | Superseded by OpenFOAM-only vector references |
| 18 | Oseen model-specific ground truth | Complete |
| 19 | Navier-Stokes model-specific ground truth | Complete |
| 20 | True performance comparison run with model-specific actual fields | Complete |
| 21 | Dedicated OpenFOAM reference case generation and field import | Complete |
| 22 | OpenFOAM-backed vector-model actual fields and PINN rerun | Complete |
| 23 | Remove manufactured streamfunction vector references from active experiment flow | Complete |
| 24 | Align Darcy boundary conditions with Navier-Stokes and use OpenFOAM Darcy ground truth | Complete |
| 25 | Reject shared OpenFOAM ground truth across PDE models | Complete |
| 26 | Switch active dedicated reference path to FEniCSx | Complete |
| 27 | FEniCSx model-specific solver generation | Complete |

## Phase 2: Darcy Flow

Status: training smoke foundation complete.

Completed in this pass:

- Defined the shared unit-square flow domain.
- Added model-agnostic inlet, outlet, and wall boundary patches.
- Added Darcy-flow loss weights for interior, inlet, outlet, and wall residuals.
- Added PyTorch autograd helpers for pressure gradients, Laplace residuals, Darcy velocity, and boundary residuals.
- Added tests for the patch definition, loss weights, harmonic pressure residual, velocity derivation, and boundary residual behavior.
- Added weighted Darcy loss assembly from shared interior and boundary collocation samples.
- Added a lightweight deterministic Darcy training smoke loop that verifies loss reduction.

Remaining:

- Add example output documentation once training exists.

## Continuation Notes

The next model-training step should build on `pinn_fluid.domains.unit_square_flow_patches`, `pinn_fluid.domains.interior_collocation_points`, `pinn_fluid.domains.boundary_collocation_points`, `pinn_fluid.models.darcy.DarcyPressureField`, `pinn_fluid.models.stokes.StokesVelocityPressureField`, and the solver pattern in `pinn_fluid.solvers.darcy` rather than redefining the domain or adding a second field abstraction. Keep training tests small enough for local execution.

## Phase 3: Stokes Flow

Status: training smoke foundation complete.

Completed in this pass:

- Added Stokes-flow residual helpers for steady incompressible low-Reynolds-number flow.
- Added continuity, x-momentum, and y-momentum residuals for `(u, v, p)` fields.
- Added a no-slip wall residual helper.
- Added Stokes loss weights for continuity, momentum, inlet, outlet, and wall residuals.
- Added tests for zero residuals under constant pressure and zero velocity, linear-field momentum behavior, and no-slip residual behavior.
- Added weighted Stokes loss assembly from shared interior and boundary collocation samples.
- Added a lightweight deterministic Stokes training smoke loop that verifies loss reduction.

Remaining:

- Add fuller Stokes example output documentation after a validated benchmark exists.

## Shared Collocation Foundation

Status: complete for the sampling foundation pass.

Completed in this pass:

- Added deterministic interior collocation sampling for the shared unit-square domain.
- Added deterministic inlet, outlet, and wall boundary sampling from the existing shared patch abstraction.
- Added outward wall normals for Darcy no-normal-flow and Stokes no-slip boundary residual construction.
- Added Stokes boundary targets derived from the shared inlet, outlet, and wall patch definitions without mutating the shared patch API.
- Added tests for sample placement, wall normals, input validation, and Stokes target mapping.

Remaining:

- Begin the Oseen residual foundation on the shared benchmark.

## Shared Neural Field Foundation

Status: complete for the neural-field foundation pass.

Completed in this pass:

- Added a reusable `MLPField` for coordinate-to-field modules.
- Added `DarcyPressureField` for scalar pressure predictions on `(x, y)`.
- Added `StokesVelocityPressureField` for named `u`, `v`, and `pressure` predictions on `(x, y)`.
- Added tests for output shapes, dtype preservation, invalid architecture settings, and named Stokes outputs.

Remaining:

- Begin Phase 4 Oseen residual utilities after the Stokes smoke-training path is stable.

## Darcy Training Smoke Foundation

Status: complete for the Darcy training smoke pass.

Completed in this pass:

- Added `DarcyTrainingConfig` for tiny local training runs.
- Added Darcy loss components for interior, inlet, outlet, and wall residuals.
- Added weighted total loss using the Phase 2 default Darcy loss weights.
- Added a deterministic smoke-training loop around `DarcyPressureField`.
- Added tests for loss components, default weighting, and loss reduction over a short optimizer run.

Remaining:

- Add documented example outputs after a fuller Darcy benchmark exists.
- Add fuller Stokes example output documentation after a validated benchmark exists.

## Stokes Training Smoke Foundation

Status: complete for the Stokes training smoke pass.

Completed in this pass:

- Added `StokesTrainingConfig` for tiny local training runs.
- Added Stokes loss components for continuity, x-momentum, y-momentum, inlet, outlet, and wall residuals.
- Added weighted total loss using the Phase 3 default Stokes loss weights.
- Added a deterministic smoke-training loop around `StokesVelocityPressureField`.
- Added tests for loss components, default weighting, and loss reduction over a short optimizer run.

Remaining:

- Begin Phase 4 with Oseen residual formulation and tests.

## Phase 4: Oseen Flow

Status: training smoke foundation complete.

Completed in this pass:

- Added Oseen residual helpers for steady incompressible reduced-order Navier-Stokes flow.
- Added continuity, x-momentum, and y-momentum residuals for `(u, v, p)` fields with a prescribed convection velocity.
- Added Oseen loss weights matching the current Stokes boundary and residual weighting shape.
- Added tests showing zero-convection Oseen residuals reduce to Stokes residuals.
- Added tests for explicit convection terms under a linear velocity field.
- Added weighted Oseen loss assembly from shared interior and boundary collocation samples.
- Added a lightweight deterministic Oseen training smoke loop that verifies loss reduction.

Remaining:

- Add example output documentation once a fuller Oseen benchmark exists.

## Phase 5: Navier-Stokes Flow

Status: training smoke foundation complete.

Completed in this pass:

- Added Navier-Stokes residual helpers for steady incompressible laminar flow.
- Added continuity, x-momentum, and y-momentum residuals for `(u, v, p)` fields with nonlinear self-advection.
- Added Navier-Stokes loss weights matching the current Stokes/Oseen boundary and residual weighting shape.
- Added tests for zero residuals under zero velocity and constant pressure.
- Added tests showing Navier-Stokes residuals match Oseen residuals when the Oseen convection velocity is the current velocity.
- Added tests for explicit nonlinear advection terms under a linear velocity field.
- Added weighted Navier-Stokes loss assembly from shared interior and boundary collocation samples.
- Added a lightweight deterministic Navier-Stokes training smoke loop that verifies loss reduction.

Remaining:

- Add example output documentation once a fuller Navier-Stokes benchmark exists.

## Phase 6: Consolidated Smoke Benchmark Summary

Status: smoke benchmark summary complete.

Completed in this pass:

- Added a phase-ordered smoke benchmark summary surface across Darcy, Stokes, Oseen, and Navier-Stokes.
- Reused the existing shared unit-square collocation samplers, neural fields, and training smoke loops without redefining geometry.
- Added structured result metadata for initial loss, final loss, optimizer step count, absolute reduction, fractional reduction, and reduction status.
- Added tests for loss-history summarization, empty-history validation, model ordering, and deterministic loss reduction across the tiny smoke benchmark set.

Remaining:

- Add documented physical example outputs after a fuller validated benchmark exists.

## Phase 7: Analytic Poiseuille Navier-Stokes Benchmark

Status: residual validation complete.

Completed in this pass:

- Added a closed-form horizontal Poiseuille channel solution on the existing unit-square geometry.
- Added pressure-drop metadata for the parabolic channel profile.
- Added tests showing the analytic field has the expected velocity profile and pressure drop.
- Added tests showing the analytic field has zero steady incompressible Navier-Stokes residual under the current autograd helpers.
- Added tests showing the analytic field satisfies horizontal wall no-slip behavior.

Remaining:

- Compare a trained Navier-Stokes field against this analytic reference in a lightweight documented example.

## Phase 8: Experiment Schema And Component Histories

Status: complete for the first experiment-data pass.

Completed in this pass:

- Added `TrainingHistory` for total objective values and component-wise histories by optimizer iteration.
- Added `ExperimentResult` for model metadata, reference metadata, grid shape, metrics, and artifact paths.
- Added JSON serialization helpers for reproducible history and summary output.
- Added tests for history length validation, stable component ordering, and result serialization.

Remaining:

- Keep the schema small until broader research runs show a concrete need for richer metadata.

## Phase 9: Darcy And Shared-Patch Navier-Stokes Vertical Slice

Status: complete for the lightweight deterministic vertical slice.

Completed in this pass:

- Added a deterministic in-repo finite-difference Laplace reference for the shared Darcy pressure setup.
- Added Darcy experiment orchestration that saves predicted pressure, reference pressure, predicted/reference velocities, residual fields, objective history, component histories, plots, and finite metrics.
- Added Navier-Stokes experiment orchestration that trains a small velocity-pressure field on the shared Darcy unit-square patch geometry while recording continuity, x-momentum, y-momentum, inlet, outlet, and wall objectives.
- Added tests that assert decreasing histories, finite metrics, and saved numeric/plot artifacts for both vertical-slice models.

Remaining:

- Run longer local studies outside the regression defaults before making accuracy or physics-ranking claims.

## Phase 10: Stokes/Oseen Extension And Consolidated Report

Status: complete for the first consolidated experiment pass.

Completed in this pass:

- Added Stokes and Oseen experiment runs against the same shared-patch unit-square reference used by the Navier-Stokes vertical slice.
- Added `run_all_experiments(...)` to run Darcy, Stokes, Oseen, and Navier-Stokes in phase order.
- Added consolidated JSON and Markdown cross-model reports with objective and field/residual metrics.
- Added tests for ordered cross-model execution, decreasing histories, finite metrics, and report artifacts.

Remaining:

- Increase grid sizes, training budgets, and reporting detail in future research runs once the lightweight pipeline is stable.

## Handoff For Follow-Up Experiment Studies

Status: lightweight experiment pipeline complete and ready for broader local studies.

Current research objective:

- Demonstrate how PINN effectiveness changes as the enforced physics grows more complex from Darcy to Stokes, Oseen, and Navier-Stokes.
- Produce predicted PINN fields, true/reference fields from lightweight in-repo CFD-style solvers, residual fields, total objective histories, per-component loss histories, and summary metrics.

Deep-interview decisions to preserve:

- Use CFD-backed references, but keep the first pass lightweight, deterministic, and in-repo.
- Build a vertical slice first for Darcy and Navier-Stokes rather than all models at once.
- Save reproducible numeric artifacts plus plots.
- Use simple deterministic grids and lightweight local runs.
- Let the implementing agent choose grid sizes, training steps, file formats, and plot layouts when documented and lightweight.
- Treat first-pass success as decreasing histories plus finite reported metrics, not strict accuracy thresholds or final claims about physics-complexity ranking.

Completed phase order:

1. Phase 8: experiment result schema and component-wise training history capture.
2. Phase 9: Darcy and shared-patch Navier-Stokes CFD-backed vertical-slice experiments.
3. Phase 10: Stokes/Oseen extension and consolidated cross-model report.
4. Phase 11: result-procurement runner and manifest for staged local runs.

## Phase 11: Result-Procurement Runner

Status: complete.

Completed in this pass:

- Added `pinn_fluid.result_procurement.run_result_procurement(...)` as a narrow wrapper around `run_all_experiments(...)`.
- Added `python -m pinn_fluid.result_procurement` for configured local sanity and paper-demo tiers.
- Wrote `run_manifest.json` under the configured output directory with git commit, config values, per-model output directories, artifact paths, metrics, finite-metric flags, and history-reduction flags.
- Added tests that first failed on the missing runner module, then verified configured output directories, manifest creation, finite metrics, decreasing histories, and artifact references on tiny deterministic settings.

Remaining:

- Phase 13 should use the runner for sanity and paper-demo tiers without committing generated data artifacts.

## Phase 12: Figure Generation From Saved Artifacts

Status: complete.

Completed in this pass:

- Added `pinn_fluid.figures.generate_figure_bundle(...)` for standalone figure generation from saved experiment output directories.
- Added `python -m pinn_fluid.figures <output_dir>` as a narrow CLI surface for local figure procurement.
- Generated per-model field panels from `fields.npz`: Darcy pressure, `u`, and `v` rows, and Stokes/Oseen/Navier-Stokes `u`, `v`, and pressure rows, each arranged under predicted, actual, and residual columns.
- Generated per-model convergence panels from `history.json` with the total objective and all recorded component losses.
- Added titled separate scalar images for Stokes/Oseen/Navier-Stokes predicted, actual, and residual/error `u`, `v`, pressure, speed, and residual-magnitude fields.
- Added the `turbo` colormap, numeric min/mid/max colorbar values, field-panel layout metadata, convergence axes, titled convergence legends, and matching manifest metadata.
- Added red shared-inlet and blue shared-outlet markers to flow-field panels and separate scalar field images, with marker metadata in `figure_manifest.json`.
- Moved the red/blue inlet/outlet marker lines just outside the top/bottom horizontal unit-square openings so they no longer overlay the rendered flow field while matching the actual boundary orientation.
- Replaced Darcy speed comparison panels and separate speed images with Darcy `u` and `v` velocity-component comparison panels and separate component images.
- Moved field-panel predicted/actual/residual column labels to the bottom, increased and bolded row/column labels, and separated colorbar labels from tick values.
- Standardized field-panel and separate-image color limits so predicted and actual fields for the same variable share the same value range while residual/error fields keep a residual-specific range.
- Added per-model pressure-velocity quiver diagnostics that overlay velocity arrows on pressure contours and preserve red/blue inlet/outlet markers.
- Made figure generation depend directly on `matplotlib>=3.8` instead of maintaining a dependency-free fallback renderer.
- Wrote `figures/figure_manifest.json` with the generated panel paths.
- Added tests that first failed on the missing `pinn_fluid.figures` module, then verified Darcy and velocity-pressure panels against small synthetic `fields.npz` and `history.json` fixtures.

Remaining:

- Phase 13 should use the result-procurement runner and this figure bundle utility on actual sanity and paper-demo tiers without committing generated output artifacts.
- Phase 14 should document the actual commands, figure inventory, artifact locations, and limitations from those completed local runs.

## Phase 13: Local Sanity And Paper-Demo Runs

Status: complete.

Completed in this pass:

- Ran the sanity tier at `grid_points=9`, `training_steps=80`, `hidden_width=12`, `hidden_layers=1`, `learning_rate=0.02`, `seed=0`, `viscosity=0.25`, `peak_velocity=1.0`, and `darcy_reference_iterations=400`.
- Wrote sanity artifacts under ignored `data/experiments_sanity/`.
- Generated sanity Phase 12 panels under `data/experiments_sanity/figures/`.
- Ran the paper-demo tier at `grid_points=31`, `training_steps=800`, `hidden_width=24`, `hidden_layers=2`, `learning_rate=0.01`, `seed=0`, `viscosity=0.25`, `peak_velocity=1.0`, and `darcy_reference_iterations=1200`.
- Wrote paper-demo artifacts under ignored `data/experiments_paper_demo/`.
- Generated paper-demo Phase 12 panels under `data/experiments_paper_demo/figures/`.
- Confirmed both run manifests reported `all_metrics_finite=true` and `all_histories_decreased=true`.
- Skipped the fallback `learning_rate=0.005`, `training_steps=1200` paper-demo rerun because all first-pass paper-demo histories decreased.

Paper-demo metrics:

| Model | Initial objective | Final objective | Velocity L2 | Pressure L2 | Residual RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Darcy | 10.4557 | 0.119401 | 0.319252 | 0.211299 | 0.118235 |
| Stokes | 6.43499 | 0.000218439 | 0.000880394 | 0.00243016 | 0.0304391 |
| Oseen | 6.41629 | 0.000276085 | 0.00149186 | 0.00306 | 0.0350645 |
| Navier-Stokes | 6.43666 | 0.000277937 | 0.00111077 | 0.00200808 | 0.0370115 |

Generated local outputs:

- `data/experiments_sanity/run_manifest.json`
- `data/experiments_sanity/figures/figure_manifest.json`
- `data/experiments_paper_demo/run_manifest.json`
- `data/experiments_paper_demo/figures/figure_manifest.json`
- Per-model `fields.npz`, `history.json`, `metrics.json`, raw plots, and figure panels under each ignored output root.

Notes:

- The manifests record git commit `6c92946`, the local Phase 12 commit available when the temporary gitdir was unavailable in this shell.
- These moderate results are procurement evidence only. They should not be used to rank physical accuracy or PINN effectiveness across models.

## Phase 14: Result Documentation And Figure Inventory

Status: complete.

Completed in this pass:

- Updated `README.md` to mark Phase 14 complete and make the result-procurement handoff current.
- Documented the actual Phase 13 sanity and paper-demo commands already used for local artifact procurement.
- Documented the paper-demo cross-model metrics table from `data/experiments_paper_demo/summary/cross_model_report.md`.
- Added a concrete paper-demo figure inventory for every required field and convergence panel.
- Listed the numeric/report artifact locations for `run_manifest.json`, `figure_manifest.json`, cross-model reports, `fields.npz`, `history.json`, and `metrics.json`.
- Documented the separate predicted, actual, and residual/error scalar images for `u`, `v`, and pressure.
- Documented that flow figures use `turbo` colorbars with values, highlight the shared inlet/outlet patches, and that convergence figures include labeled axes plus a loss-component legend.
- Recorded limitations that keep the moderate demo tier from being interpreted as final physical accuracy or model-ranking evidence.
- Preserved the policy that generated outputs under `data/` remain ignored and uncommitted.

Final paper-demo figure inventory:

| Model | Field panel | Convergence panel |
| --- | --- | --- |
| Darcy | `data/experiments_paper_demo/figures/darcy_fields.png` | `data/experiments_paper_demo/figures/darcy_convergence.png` |
| Stokes | `data/experiments_paper_demo/figures/stokes_fields.png` | `data/experiments_paper_demo/figures/stokes_convergence.png` |
| Oseen | `data/experiments_paper_demo/figures/oseen_fields.png` | `data/experiments_paper_demo/figures/oseen_convergence.png` |
| Navier-Stokes | `data/experiments_paper_demo/figures/navier_stokes_fields.png` | `data/experiments_paper_demo/figures/navier_stokes_convergence.png` |

Known limitations from the actual runs:

- The paper-demo run used one seed, one grid size, and one training budget.
- The Darcy and vector-flow shared-patch references are lightweight in-repo references, not external CFD validation datasets.
- Stokes, Oseen, and Navier-Stokes now use the same top-left inlet and bottom-right outlet geometry as Darcy; the vector-flow reference remains a paper-demo procurement reference rather than a high-fidelity CFD solve.
- The moderate demo tier is suitable for paper-draft artifact procurement and workflow validation, not final accuracy claims.
- Cross-model values should not be used to rank PINN effectiveness because the references are lightweight paper-demo procurement targets.
- Generated `.npz`, `.json`, and `.png` outputs remain local under ignored `data/` directories.

## Follow-Up: Boundary And Visualization Diagnostics

Status: complete.

Completed in this pass:

- Confirmed the shared coordinate convention remains Cartesian unit-square coordinates: `x` increases left-to-right, `y=0` is bottom, and `y=1` is top.
- Added `pinn_fluid.diagnostics.boundary_diagnostic_report(...)` to print counts and min/max coordinates for inlet, outlet, wall, and interior samples.
- Added `python -m pinn_fluid.diagnostics <png> --report <json>` to plot the boundary masks and write a JSON coordinate report.
- Confirmed the expected default masks are top-left inlet (`x in [0, 0.25]`, `y=1`) and bottom-right outlet (`x in [0.75, 1]`, `y=0`).
- Added pressure-velocity quiver figures for Darcy, Stokes, Oseen, and Navier-Stokes figure bundles.
- Updated figure manifests with pressure-velocity quiver paths and field-panel color-limit metadata.
- Updated separate scalar image metadata with explicit `color_limits`; predicted and actual images now share limits for `u`, `v`, pressure, and speed.
- Updated `README.md` with boundary diagnostic commands, coordinate-convention notes, quiver diagnostic outputs, and the color-scale policy.

Diagnosis:

- The boundary point generation and mask definitions match the intended shared domain.
- The solver boundary mask is correct for the intended geometry.
- The `matplotlib` plotting path uses `origin="lower"` with the same `x, y` coordinate convention.
- Fallback rendering has been removed; missing `matplotlib` now surfaces as Python's normal import error.
- Remaining physically suspicious extrema or structured residuals after regenerating figures should be treated as reference/training/model-form limitations until the boundary-mask and quiver diagnostics show another coordinate mismatch.

## Phase 15: Reference-Generation Audit And Contracts

Status: complete.

Completed in this pass:

- Added contract tests for the current Darcy finite-difference reference, including shape, finite values, top-left inlet pressure, and bottom-right outlet pressure.
- Added contract tests proving Stokes, Oseen, and Navier-Stokes currently save the same Darcy-derived shared-patch vector reference.
- Added coordinate-orientation tests for x-major grid flattening, y-up mathematical coordinates, and figure reshaping with `reshape(grid_points, grid_points).T`.
- Added reference metadata to `ExperimentResult`, `fields.npz` through `reference_metadata_json`, result-procurement manifest entries, and cross-model JSON reports.
- Documented the active reference generator for each model in `README.md`.

Current reference generators:

| Model | Reference generator | PDE represented | Reference kind |
| --- | --- | --- | --- |
| Darcy | `fd_darcy_reference` | Darcy pressure Laplace equation | finite-difference |
| Stokes | `stokes_streamfunction_reference` | Stokes incompressible momentum and continuity | manufactured |
| Oseen | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Navier-Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |

Artifacts produced by current experiments:

- `fields.npz` with predicted fields, reference fields, residual fields, coordinates, and `reference_metadata_json`.
- `history.json` with total and component-wise training histories.
- `metrics.json` with finite comparison metrics.
- `run_manifest.json` with per-model reference metadata when using result procurement.
- Figure bundles generated from `fields.npz` and `history.json` remain compatible with the metadata addition.

Limitations:

- Phase 15 does not introduce model-specific Stokes, Oseen, or Navier-Stokes actual fields.
- The vector-flow references remain demo-only Darcy-derived shared-patch fields until later phases replace them.
- The metadata records the current behavior so future phases can distinguish a true model-specific reference from the current procurement target.

Verification:

- `python -m pytest tests/test_phase15_reference_contracts.py` first failed on missing `reference_metadata` in `ExperimentResult` and missing manifest metadata.
- After the narrow implementation, `python -m pytest tests/test_phase15_reference_contracts.py` passed.
- Final verification for the phase is `python -m pytest tests/test_phase15_reference_contracts.py`, `python -m pytest`, and `git diff --check`.

Reproduction commands:

```bash
python -m pytest tests/test_phase15_reference_contracts.py
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --git-commit <commit>
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

## Phase 16: Darcy Ground Truth Hardening

Status: complete.

Completed in this pass:

- Reviewed `_fd_darcy_reference(...)` against the required Cartesian unit-square convention, top-left pressure inlet, bottom-right pressure outlet, and no-normal-flow walls.
- Added tests for Darcy pressure extrema on the inlet/outlet patches.
- Added tests showing the Darcy velocity points from the high-pressure inlet toward the low-pressure outlet through the interior.
- Added tests for no-normal-flow wall behavior away from inlet/outlet openings.
- Added `_fd_darcy_reference_fields(...)` to expose named pressure, velocity, `u`, `v`, speed, and finite-difference residual diagnostics.
- Updated Darcy experiment artifacts to save `reference_u`, `reference_v`, `reference_speed`, `reference_residual`, `predicted_u`, `predicted_v`, and `predicted_speed` alongside the existing pressure, velocity, metadata, and residual arrays.
- Added `reference_residual_rms` to Darcy metrics.
- Updated Darcy figure generation to consume explicit component arrays when present while preserving compatibility with older `reference_velocity`/`predicted_velocity` artifacts.
- Updated `README.md` with the Phase 16 artifact schema, active reference generators, limitations, and reproduction commands.

Current reference generators:

| Model | Reference generator | PDE represented | Reference kind |
| --- | --- | --- | --- |
| Darcy | `fd_darcy_reference` | Darcy pressure Laplace equation | finite-difference |
| Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Oseen | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Navier-Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |

Artifacts produced by current Darcy experiments:

- `fields.npz` with coordinates, predicted/reference pressure, predicted/reference velocity, explicit predicted/reference `u` and `v`, predicted/reference speed, PINN residual, finite-difference reference residual, and `reference_metadata_json`.
- `history.json` with total and component-wise training histories.
- `metrics.json` with `pressure_l2`, `velocity_l2`, `residual_rms`, and `reference_residual_rms`.
- Figure bundles continue to read Darcy `fields.npz` and generate pressure, `u`, `v`, residual, convergence, and quiver diagnostics.

Limitations:

- Phase 16 hardens only Darcy actual fields.
- Stokes, Oseen, and Navier-Stokes still use the Phase 15 demo-only Darcy-derived vector reference.
- The Darcy reference is a lightweight in-repo finite-difference solve, not external CFD validation data.

Verification:

- `python -m pytest tests/test_phase16_darcy_ground_truth.py` first failed on missing `_fd_darcy_reference_fields` and missing explicit Darcy artifact fields.
- After implementation, `python -m pytest tests/test_phase16_darcy_ground_truth.py` passed.
- Final verification for the phase is `python -m pytest tests/test_phase16_darcy_ground_truth.py`, `python -m pytest`, and `git diff --check`.

Reproduction commands:

```bash
python -m pytest tests/test_phase16_darcy_ground_truth.py
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --git-commit <commit>
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

## Phase 17: Stokes Model-Specific Ground Truth

Status: complete.

Completed in this pass:

- Added `_stokes_reference_fields(...)`, a deterministic manufactured Stokes streamfunction reference on the shared unit-square geometry.
- Built the reference from a streamfunction so continuity is zero up to numerical/autograd precision.
- Added Stokes reference diagnostics for continuity, x-momentum, y-momentum, residual magnitude, velocity components, speed, and pressure.
- Wired `run_stokes_experiment(...)` to use `manufactured_stokes_streamfunction` instead of the Darcy-derived shared-patch vector reference.
- Updated Stokes reference metadata to `stokes_streamfunction_reference`, PDE model `Stokes incompressible momentum and continuity`, and kind `manufactured`.
- Preserved Oseen and Navier-Stokes on the existing Darcy-derived demo-only vector reference at the close of Phase 17.
- Added tests for Stokes incompressibility behavior, momentum residual diagnostics, inlet/outlet/horizontal-wall boundary behavior, finite metrics, artifact schema compatibility, and figure-generation compatibility.
- Updated `README.md` with the Phase 17 artifact schema, active reference generators, limitations, and reproduction commands.

Current reference generators:

| Model | Reference generator | PDE represented | Reference kind |
| --- | --- | --- | --- |
| Darcy | `fd_darcy_reference` | Darcy pressure Laplace equation | finite-difference |
| Stokes | `stokes_streamfunction_reference` | Stokes incompressible momentum and continuity | manufactured |
| Oseen | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
| Navier-Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |

Artifacts produced by current Stokes experiments:

- `fields.npz` with coordinates, predicted/reference `u`, `v`, pressure, reference speed, reference continuity, reference x/y momentum residuals, reference residual magnitude, PINN residual magnitude, and `reference_metadata_json`.
- `history.json` with total and component-wise training histories.
- `metrics.json` with `pressure_l2`, `velocity_l2`, `residual_rms`, `reference_continuity_rms`, and `reference_momentum_rms`.
- Figure bundles continue to generate Stokes field panels, scalar fields, convergence panels, and pressure-velocity quiver diagnostics from the saved artifact schema.

Limitations:

- The Stokes reference is manufactured and deterministic; it is not external CFD validation data.
- The manufactured field is intended as a model-specific comparison target and diagnostic reference, not a final physical benchmark for publication-grade accuracy claims.
- Phase 18 subsequently replaces Oseen with a manufactured model-specific reference; Navier-Stokes remains on the Phase 15 demo-only Darcy-derived vector reference until Phase 19.

Verification:

- `python -m pytest tests/test_phase17_stokes_reference.py tests/test_phase15_reference_contracts.py tests/test_phase10_consolidated_experiments.py` first failed because `_stokes_reference_fields` did not exist and Stokes still reported `shared_patch_unit_square_reference`.
- After implementation and contract updates, targeted tests passed.
- Final verification for the phase is `python -m pytest tests/test_phase17_stokes_reference.py`, `python -m pytest`, and `git diff --check`.

Reproduction commands:

```bash
python -m pytest tests/test_phase17_stokes_reference.py
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --git-commit <commit>
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

## Phase 18: Oseen Model-Specific Ground Truth

Status: complete.

Completed in this pass:

- Added `_oseen_reference_fields(...)`, a deterministic manufactured Oseen streamfunction reference on the shared unit-square geometry.
- Built the reference from a streamfunction so continuity is zero up to numerical/autograd precision.
- Evaluated Oseen momentum residual diagnostics with a documented prescribed convection velocity `(peak_velocity, 0.0)`.
- Wired `run_oseen_experiment(...)` to use `manufactured_oseen_streamfunction` instead of the Darcy-derived shared-patch vector reference.
- Updated Oseen reference metadata to `oseen_streamfunction_reference`, PDE model `Oseen incompressible momentum and continuity`, kind `manufactured`, and the per-run convection velocity.
- Preserved Navier-Stokes on the existing Darcy-derived demo-only vector reference at the close of Phase 18.
- Added tests for Oseen continuity behavior, momentum residual diagnostics, inlet/outlet/horizontal-wall boundary behavior, finite metrics, artifact schema compatibility, and figure-generation compatibility.
- Updated `README.md` with the Phase 18 artifact schema, active reference generators, limitations, and reproduction commands.

Current reference generators:

| Model | Reference generator | PDE represented | Reference kind |
| --- | --- | --- | --- |
| Darcy | `fd_darcy_reference` | Darcy pressure Laplace equation | finite-difference |
| Stokes | `stokes_streamfunction_reference` | Stokes incompressible momentum and continuity | manufactured |
| Oseen | `oseen_streamfunction_reference` | Oseen incompressible momentum and continuity | manufactured |
| Navier-Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |

Artifacts produced by current Oseen experiments:

- `fields.npz` with coordinates, predicted/reference `u`, `v`, pressure, reference speed, reference continuity, reference x/y momentum residuals, reference residual magnitude, reference convection velocity, PINN residual magnitude, and `reference_metadata_json`.
- `history.json` with total and component-wise training histories.
- `metrics.json` with `pressure_l2`, `velocity_l2`, `residual_rms`, `reference_continuity_rms`, and `reference_momentum_rms`.
- Figure bundles continue to generate Oseen field panels, scalar fields, convergence panels, and pressure-velocity quiver diagnostics from the saved artifact schema.

Limitations:

- The Oseen reference is manufactured and deterministic; it is not external CFD validation data.
- The prescribed convection velocity is fixed to `(peak_velocity, 0.0)` for this reference generator.
- Phase 19 subsequently replaces Navier-Stokes with a manufactured model-specific reference.

Verification:

- `python -m pytest tests/test_phase18_oseen_reference.py tests/test_phase17_stokes_reference.py tests/test_phase15_reference_contracts.py tests/test_phase10_consolidated_experiments.py tests/test_phase11_result_procurement.py tests/test_phase12_figure_generation.py` first failed because `_oseen_reference_fields` did not exist and Oseen still reported `shared_patch_unit_square_reference`.
- After implementation and contract updates, the same targeted test set passed.
- Final verification for the phase is `python -m pytest tests/test_phase18_oseen_reference.py`, `python -m pytest`, and `git diff --check`.

Reproduction commands:

```bash
python -m pytest tests/test_phase18_oseen_reference.py
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --git-commit <commit>
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

## Phase 19: Navier-Stokes Model-Specific Ground Truth

Status: complete.

Completed in this pass:

- Added `_navier_stokes_reference_fields(...)`, a deterministic manufactured Navier-Stokes streamfunction reference on the shared unit-square geometry.
- Built the reference from a streamfunction so continuity is zero up to numerical/autograd precision.
- Evaluated nonlinear Navier-Stokes momentum residual diagnostics with the existing `navier_stokes_residuals(...)` helper.
- Wired the existing backward-compatible `run_poiseuille_navier_stokes_experiment(...)` API to use `manufactured_navier_stokes_streamfunction` instead of the Darcy-derived shared-patch vector reference.
- Updated Navier-Stokes reference metadata to `navier_stokes_streamfunction_reference`, PDE model `Navier-Stokes incompressible momentum and continuity`, and kind `manufactured`.
- Added tests for Navier-Stokes continuity behavior, nonlinear momentum residual diagnostics, inlet/outlet/horizontal-wall boundary behavior, finite metrics, artifact schema compatibility, and figure-generation compatibility.
- Updated `README.md` with the Phase 19 artifact schema, active reference generators, limitations, and reproduction commands.

Current reference generators:

| Model | Reference generator | PDE represented | Reference kind |
| --- | --- | --- | --- |
| Darcy | `fd_darcy_reference` | Darcy pressure Laplace equation | finite-difference |
| Stokes | `stokes_streamfunction_reference` | Stokes incompressible momentum and continuity | manufactured |
| Oseen | `oseen_streamfunction_reference` | Oseen incompressible momentum and continuity | manufactured |
| Navier-Stokes | `navier_stokes_streamfunction_reference` | Navier-Stokes incompressible momentum and continuity | manufactured |

Artifacts produced by current Navier-Stokes experiments:

- `fields.npz` with coordinates, predicted/reference `u`, `v`, pressure, reference speed, reference continuity, reference x/y momentum residuals, reference residual magnitude, PINN residual magnitude, and `reference_metadata_json`.
- `history.json` with total and component-wise training histories.
- `metrics.json` with `pressure_l2`, `velocity_l2`, `residual_rms`, `reference_continuity_rms`, and `reference_momentum_rms`.
- Figure bundles continue to generate Navier-Stokes field panels, scalar fields, convergence panels, and pressure-velocity quiver diagnostics from the saved artifact schema.

Limitations:

- The Navier-Stokes reference is manufactured and deterministic; it is not external CFD validation data.
- The manufactured field is intended as a model-specific comparison target and diagnostic reference, not a final physical benchmark for publication-grade accuracy claims.
- Phase 21 remains reserved for a dedicated OpenFOAM, FEniCS, FiPy, SciPy, or similar reference-solver integration.

Verification:

- `python -m pytest tests/test_phase19_navier_stokes_reference.py tests/test_phase18_oseen_reference.py tests/test_phase15_reference_contracts.py tests/test_phase10_consolidated_experiments.py tests/test_phase9_vertical_slice_experiments.py` first failed because `_navier_stokes_reference_fields` did not exist and Navier-Stokes still reported `shared_patch_unit_square_reference`.
- After implementation and contract updates, the same targeted test set passed.
- Final verification for the phase is `python -m pytest tests/test_phase19_navier_stokes_reference.py`, `python -m pytest`, and `git diff --check`.

Reproduction commands:

```bash
python -m pytest tests/test_phase19_navier_stokes_reference.py
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --git-commit <commit>
PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity
```

## Phase 20: True Performance Comparison Run

Status: complete.

Completed in this pass:

- Added a top-level `reference_generators` summary to result-procurement manifests so each run directly records which reference generator each model used.
- Added Phase 20 tests for model-specific reference-generator recording, finite/decreasing procurement manifests, required field schemas, and figure-bundle artifact compatibility.
- Ran the sanity tier under `data/experiments_sanity/` with model-specific references.
- Regenerated the sanity figure bundle under `data/experiments_sanity/figures/`.
- Ran the paper-demo tier under `data/experiments_paper_demo/` with model-specific references.
- Regenerated the paper-demo figure bundle under `data/experiments_paper_demo/figures/`.
- Confirmed each tier produced `run_manifest.json`, per-model `fields.npz`, `metrics.json`, `history.json`, `summary/cross_model_report.json`, `summary/cross_model_report.md`, `figures/figure_manifest.json`, field panels, convergence panels, scalar field images, and pressure-velocity quiver diagnostics.
- Left generated `.npz`, `.json`, and `.png` artifacts under ignored `data/` paths; they are not committed.

Reference generators recorded in both Phase 20 manifests:

| Model | Reference generator | Reference kind |
| --- | --- | --- |
| Darcy | `fd_darcy_reference` | finite-difference |
| Stokes | `stokes_streamfunction_reference` | manufactured |
| Oseen | `oseen_streamfunction_reference` | manufactured |
| Navier-Stokes | `navier_stokes_streamfunction_reference` | manufactured |

Sanity tier:

- Command: `PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_sanity --grid-points 9 --training-steps 80 --hidden-width 12 --hidden-layers 1 --learning-rate 0.02 --seed 0 --viscosity 0.25 --peak-velocity 1.0 --darcy-reference-iterations 400 --git-commit phase20-local-model-specific-references`
- Figure command: `PYTHONPATH=src python -m pinn_fluid.figures data/experiments_sanity`
- Manifest result: `all_metrics_finite=true`, `all_histories_decreased=true`

Paper-demo tier:

- Command: `PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/experiments_paper_demo --grid-points 31 --training-steps 800 --hidden-width 24 --hidden-layers 2 --learning-rate 0.01 --seed 0 --viscosity 0.25 --peak-velocity 1.0 --darcy-reference-iterations 1200 --git-commit phase20-local-model-specific-references`
- Figure command: `PYTHONPATH=src python -m pinn_fluid.figures data/experiments_paper_demo`
- Manifest result: `all_metrics_finite=true`, `all_histories_decreased=true`

Paper-demo metrics:

| Model | Initial objective | Final objective | Velocity L2 | Pressure L2 | Residual RMS | Reference continuity RMS | Reference momentum RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Darcy | 10.4557 | 0.119401 | 0.319252 | 0.211299 | 0.118235 | n/a | n/a |
| Stokes | 7.20939 | 0.35744 | 0.224723 | 0.531488 | 0.442768 | 1.91367e-16 | 11.7339 |
| Oseen | 7.19069 | 0.374738 | 0.227178 | 0.526716 | 0.467904 | 1.91367e-16 | 12.0635 |
| Navier-Stokes | 7.21106 | 0.375367 | 0.223985 | 0.556291 | 0.481473 | 1.91367e-16 | 11.7356 |

Generated local outputs:

- `data/experiments_sanity/run_manifest.json`
- `data/experiments_sanity/figures/figure_manifest.json`
- `data/experiments_paper_demo/run_manifest.json`
- `data/experiments_paper_demo/figures/figure_manifest.json`
- per-model `fields.npz`, `history.json`, `metrics.json`, raw plots, field panels, scalar images, convergence plots, and quiver diagnostics under each ignored output root

Limitations:

- Phase 20 is a clean local comparison against lightweight in-repo finite-difference/manufactured references, not external CFD validation.
- Vector-flow references are deterministic manufactured fields for model-specific comparison and residual diagnostics, not final physical ground truth.
- The run used one seed and one training budget per tier; no uncertainty, mesh, grid-size, hyperparameter, or sensitivity study is reported.
- The `matplotlib` commands emitted the existing `tight_layout` warning from `src/pinn_fluid/figures.py:500`; generated PNGs and figure manifests were still written.

Verification:

- `python -m pytest tests/test_phase20_comparison_run_contracts.py` first failed because `run_manifest.json` did not include a top-level `reference_generators` summary.
- After implementation, `python -m pytest tests/test_phase20_comparison_run_contracts.py tests/test_phase11_result_procurement.py` passed.
- Final verification for the phase is `python -m pytest tests/test_phase20_comparison_run_contracts.py`, `python -m pytest`, and `git diff --check`.

## Phase 21: Dedicated Reference Solver Integration

Status: complete for OpenFOAM case generation and sampled-field import.

Solver-selection decision:

- Selected OpenFOAM 13 for the dedicated-solver phase, per the user direction to use OpenFOAM.
- Did not add Python dependencies.
- Initial local environment check found no `blockMesh`, `simpleFoam`, or `foamVersion` executables on `PATH`; after user-installed OpenFOAM 13, `blockMesh`, `simpleFoam`, and `foamRun` were available.
- Implemented the OpenFOAM case writer and sampled-field importer so the phase is testable without committing generated solver outputs and CI can still run without OpenFOAM.

Completed:

- Added `pinn_fluid.dedicated_solvers`.
- Added `OpenFOAMCaseConfig`, `write_openfoam_shared_domain_case(...)`, `import_openfoam_sampled_fields(...)`, and `run_openfoam_reference_import(...)`.
- Generated an OpenFOAM `simpleFoam` case skeleton with:
  - `0/U`
  - `0/p`
  - `constant/momentumTransport`
  - `constant/physicalProperties`
  - `constant/transportProperties`
  - `system/blockMeshDict`
  - `system/controlDict`
  - `system/fvSchemes`
  - `system/fvSolution`
  - `system/sampleDict`
  - `reference_metadata.json`
- Preserved the repository coordinate convention in metadata: unit square, `y=0` bottom, `y=1` top, top-left inlet, bottom-right outlet, and existing flattening/plotting order.
- Imported OpenFOAM raw sampled fields with columns `x y z p Ux Uy Uz` into the existing figure-compatible artifact schema.
- Wrote artifact bundles for imported OpenFOAM samples:
  - `dedicated_solver_manifest.json`
  - `openfoam_darcy_reference/fields.npz`
  - `openfoam_darcy_reference/history.json`
  - `openfoam_darcy_reference/metrics.json`
- Recorded reference metadata with:
  - `reference_generator_name="openfoam_simplefoam_shared_domain"`
  - `reference_kind="dedicated-solver"`
  - `solver_stack="OpenFOAM simpleFoam"`
  - expected sampled columns
  - the shared coordinate convention
- Added tests for generated OpenFOAM case files, metadata, coordinate orientation, flattening order, finite imported fields, artifact schema compatibility, and manifest reference-generator recording.
- After OpenFOAM installation, patched the writer for OpenFOAM 13 compatibility:
  - added `constant/momentumTransport` and `constant/physicalProperties`
  - changed sampling from a diagonal `lineFace` set to an explicit `points` set over the full square grid
- Verified `blockMesh`, `foamRun -solver incompressibleFluid`, `postProcess -func sampleDict`, artifact import, and figure generation on an ignored local case.

Reference generators after Phase 21:

- Darcy PINN experiment: `fd_darcy_reference`
- Stokes PINN experiment: `stokes_streamfunction_reference`
- Oseen PINN experiment: `oseen_streamfunction_reference`
- Navier-Stokes PINN experiment: `navier_stokes_streamfunction_reference`
- OpenFOAM dedicated bundle: `openfoam_simplefoam_shared_domain`

Reproduction commands:

```bash
PYTHONPATH=src python -c "from pinn_fluid.dedicated_solvers import OpenFOAMCaseConfig, write_openfoam_shared_domain_case; write_openfoam_shared_domain_case('data/openfoam_phase21/case', OpenFOAMCaseConfig(grid_points=31, viscosity=0.25))"
blockMesh -case data/openfoam_phase21/case
foamRun -solver incompressibleFluid -case data/openfoam_phase21/case
postProcess -func sampleDict -latestTime -case data/openfoam_phase21/case
PYTHONPATH=src python -c "from pinn_fluid.dedicated_solvers import run_openfoam_reference_import; run_openfoam_reference_import(output_dir='data/openfoam_phase21/imported', sample_path='data/openfoam_phase21/case/postProcessing/sampleDict/50/sharedDomainGrid.xy', grid_points=31)"
PYTHONPATH=src python -c "from pinn_fluid.figures import generate_figure_bundle; generate_figure_bundle('data/openfoam_phase21/imported', models=('openfoam_darcy_reference',))"
```

Local Phase 21 artifact evidence:

- `data/openfoam_phase21/case/postProcessing/sampleDict/50/sharedDomainGrid.xy`
- `data/openfoam_phase21/imported/dedicated_solver_manifest.json`
- `data/openfoam_phase21/imported/openfoam_darcy_reference/fields.npz`
- `data/openfoam_phase21/imported/openfoam_darcy_reference/history.json`
- `data/openfoam_phase21/imported/openfoam_darcy_reference/metrics.json`
- `data/openfoam_phase21/imported/figures/figure_manifest.json`

These generated OpenFOAM case outputs, meshes, logs, `.npz`, `.json`, and `.png` outputs remain ignored under `data/openfoam_phase21/` unless explicitly approved for commit.

Limitations:

- Phase 21 adds OpenFOAM case-generation and sampled-field import contracts; it does not commit the OpenFOAM solve result.
- `simpleFoam` is superseded in OpenFOAM 13; the verified solver command is `foamRun -solver incompressibleFluid`.
- It does not replace `run_all_experiments(...)` references or the regression-friendly finite-difference/manufactured references.
- The generated OpenFOAM mesh uses whole top and bottom patches as an executable approximation while metadata preserves the intended top-left inlet and bottom-right outlet convention; a later phase should add a segmented OpenFOAM mesh before making physical validation claims.
- The imported bundle mirrors OpenFOAM sample fields into `predicted_*` keys only to reuse existing figure-generation panels for a reference self-comparison.

Verification:

- `python -m pytest tests/test_phase21_dedicated_solver_reference.py` first failed because the interrupted SciPy module did not expose the OpenFOAM metadata/API.
- After implementation, `python -m pytest tests/test_phase21_dedicated_solver_reference.py` passed.
- After OpenFOAM installation, `foamRun -solver incompressibleFluid` first failed because OpenFOAM 13 required `constant/momentumTransport`; a failing test was added and the writer was patched.
- `postProcess -func sampleDict` then failed because `type face` had been renamed; a failing test was added and the writer was patched.
- Sampling initially produced a diagonal set, so the writer was changed to an explicit `points` set that produced 961 samples for a 31 by 31 grid.
- Verified local OpenFOAM commands: `blockMesh -case data/openfoam_phase21/case`, `foamRun -solver incompressibleFluid -case data/openfoam_phase21/case`, and `postProcess -func sampleDict -latestTime -case data/openfoam_phase21/case`.
- Verified import and figure commands for `data/openfoam_phase21/imported`.
- Final verification for the phase is `python -m pytest tests/test_phase21_dedicated_solver_reference.py`, `python -m pytest`, and `git diff --check`.

## Continuing Guidance

Required workflow for future phases or follow-up studies:

- Read `README.md` and this file.
- Inspect current model, solver, benchmark, and test patterns before editing.
- Write failing tests first, then implement narrowly.
- Update docs and progress in the same phase.
- Verify with targeted tests, `python -m pytest`, `git diff --check`, and diagnostics or `py_compile` where useful.
- Commit with the Lore protocol and push `origin main`.

## Vector Reference Pressure Solver Review

Status: follow-up bug fix complete.

Finding:

- The Stokes, Oseen, and Navier-Stokes manufactured reference builders were still using `pressure = 0` everywhere.
- That made the "actual" pressure panels constant and removed `grad(p)` from the model-specific residual diagnostics for the vector references.
- The issue was in `pinn_fluid.experiments`, not the plotting pipeline; saved `reference_pressure` arrays for Stokes/Oseen/Navier-Stokes had zero range.

Changes:

- Added regression tests that require nonconstant vector-model reference pressure.
- Replaced the zero-pressure placeholder with a differentiable smooth pressure potential anchored to the shared top-left inlet and bottom-right outlet.
- Added model-specific dynamic pressure corrections for Stokes, Oseen, and Navier-Stokes manufactured references.
- Recorded the manufactured pressure model in reference metadata.

Reference generators after the pressure fix:

- Darcy PINN experiment: `fd_darcy_reference`
- Stokes PINN experiment: `stokes_streamfunction_reference`
- Oseen PINN experiment: `oseen_streamfunction_reference`
- Navier-Stokes PINN experiment: `navier_stokes_streamfunction_reference`
- OpenFOAM dedicated bundle: `openfoam_simplefoam_shared_domain`

Limitations:

- The vector reference pressures are still manufactured fields for regression-friendly comparison and residual diagnostics.
- They are no longer constant placeholders, but they are not a substitute for segmented-mesh OpenFOAM or another dedicated CFD pressure solve.

Verification:

- Failing test first: `python -m pytest tests/test_phase17_stokes_reference.py tests/test_phase18_oseen_reference.py tests/test_phase19_navier_stokes_reference.py` failed on zero pressure range for Stokes, Oseen, and Navier-Stokes.
- After implementation, the same targeted command passed.

## Phase 22: OpenFOAM-Backed Vector Actual Fields

Status: complete.

What changed:

- Added `ExperimentConfig.vector_reference_source`.
- Added OpenFOAM reference preparation in `pinn_fluid.result_procurement`.
- Added `--vector-reference-source openfoam`, `--openfoam-reference-sample-path`, `--openfoam-case-dir`, and `--openfoam-end-time` CLI options.
- Stokes, Oseen, and Navier-Stokes can now use OpenFOAM-sampled pressure and velocity fields as their actual/reference fields.
- The OpenFOAM case writer now segments the top and bottom mesh boundaries so the executable case matches the shared geometry:
  - inlet: top-left horizontal segment, `x in [0.0, 0.25]`, `y=1`
  - outlet: bottom-right horizontal segment, `x in [0.75, 1.0]`, `y=0`
  - walls: remaining perimeter
- Added finite-difference post-processing of sampled OpenFOAM fields to produce per-model continuity, x-momentum, y-momentum, and residual-magnitude diagnostics for Stokes, Oseen, and Navier-Stokes.
- Added regression tests proving the OpenFOAM vector-reference path does not use manufactured metadata or manufactured reference generators.

Reference generators after Phase 22:

- Darcy PINN experiment: `fd_darcy_reference`
- Stokes PINN experiment with `--vector-reference-source openfoam`: `openfoam_simplefoam_shared_domain`
- Oseen PINN experiment with `--vector-reference-source openfoam`: `openfoam_simplefoam_shared_domain`
- Navier-Stokes PINN experiment with `--vector-reference-source openfoam`: `openfoam_simplefoam_shared_domain`

Artifacts produced by the OpenFOAM-backed comparison:

- `run_manifest.json`
- per-model `fields.npz`, `history.json`, `metrics.json`, and raw field/loss plots
- `summary/cross_model_report.json`
- `summary/cross_model_report.md`
- OpenFOAM case files and sampled output under `openfoam_case/`
- figure manifest, field panels, scalar field images, convergence panels, and quiver diagnostics after running `pinn_fluid.figures`

Local run:

```bash
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/openfoam_vector_predictions_2026_05_09 --grid-points 41 --training-steps 800 --hidden-width 24 --hidden-layers 2 --learning-rate 0.01 --seed 0 --viscosity 0.25 --peak-velocity 1.0 --darcy-reference-iterations 1200 --vector-reference-source openfoam --openfoam-end-time 50 --git-commit openfoam-vector-predictions-2026-05-09
PYTHONPATH=src python -m pinn_fluid.figures data/openfoam_vector_predictions_2026_05_09
```

Local artifact evidence:

- `data/openfoam_vector_predictions_2026_05_09/openfoam_case/postProcessing/sampleDict/50/sharedDomainGrid.xy`
- `data/openfoam_vector_predictions_2026_05_09/run_manifest.json`
- `data/openfoam_vector_predictions_2026_05_09/{darcy,stokes,oseen,navier_stokes}/fields.npz`
- `data/openfoam_vector_predictions_2026_05_09/figures/figure_manifest.json`

Run result summary:

- `all_metrics_finite=true`
- `all_histories_decreased=true`
- grid shape: `41 x 41`
- vector reference generator: `openfoam_simplefoam_shared_domain`

Limitations:

- At Phase 22 close, Darcy remained on the finite-difference Darcy reference; Phase 24 later replaced the active Darcy experiment reference with OpenFOAM-sampled fields.
- Stokes, Oseen, and Navier-Stokes share one OpenFOAM laminar-flow sample as external pressure/velocity ground truth for this phase.
- Reference residual diagnostics are finite-difference diagnostics computed from sampled OpenFOAM fields; they are not native OpenFOAM solver residual histories.
- Generated artifacts under `data/` remain ignored and should not be committed unless explicitly requested.

Verification:

- Red test: `python -m pytest tests/test_phase22_openfoam_experiment_references.py` failed before the OpenFOAM vector-reference config existed.
- Red geometry test: `python -m pytest tests/test_phase21_dedicated_solver_reference.py` failed before the blockMesh writer had segmented inlet/outlet coordinates.
- Targeted green tests: `python -m pytest tests/test_phase21_dedicated_solver_reference.py tests/test_phase22_openfoam_experiment_references.py`.
- Real OpenFOAM smoke run: `PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/openfoam_vector_smoke_2026_05_09 --grid-points 9 --training-steps 4 --hidden-width 6 --hidden-layers 1 --learning-rate 0.03 --seed 22 --viscosity 0.25 --peak-velocity 1.0 --darcy-reference-iterations 20 --vector-reference-source openfoam --openfoam-end-time 50 --git-commit openfoam-vector-smoke-2026-05-09`.
- Full OpenFOAM-backed PINN run and figure generation commands are listed above.

## Phase 23: Remove Manufactured Streamfunction Vector References

Status: complete.

What changed:

- Removed the manufactured Stokes, Oseen, and Navier-Stokes streamfunction reference builders from `pinn_fluid.experiments`.
- Removed the active manufactured vector-reference branch and metadata constants from the vector experiment APIs.
- Changed `ExperimentConfig.vector_reference_source` to default to `openfoam`.
- Changed result procurement so `--vector-reference-source` accepts only `openfoam`.
- Updated Stokes, Oseen, Navier-Stokes, consolidated experiment, and procurement tests to use OpenFOAM-style sample fields.
- Added shared test fixtures for deterministic OpenFOAM-style sample fields without adding generated artifacts under `data/`.

Reference generators at Phase 23 close:

- Darcy PINN experiment: `fd_darcy_reference`
- Stokes PINN experiment: `openfoam_simplefoam_shared_domain`
- Oseen PINN experiment: `openfoam_simplefoam_shared_domain`
- Navier-Stokes PINN experiment: `openfoam_simplefoam_shared_domain`

Artifacts produced:

- Same experiment artifact schema as Phase 22: per-model `fields.npz`, `history.json`, `metrics.json`, plots, summary reports, and figure bundles.
- Vector-model `fields.npz` files continue to include reference `u`, `v`, pressure, speed, continuity, x/y momentum diagnostics, residual magnitude, and `reference_metadata_json`.
- Generated OpenFOAM/PINN outputs remain ignored under `data/` unless explicitly approved for commit.

Commands:

```bash
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/openfoam_vector_predictions_2026_05_09 --grid-points 41 --training-steps 800 --hidden-width 24 --hidden-layers 2 --learning-rate 0.01 --seed 0 --viscosity 0.25 --peak-velocity 1.0 --darcy-reference-iterations 1200 --vector-reference-source openfoam --openfoam-end-time 50 --git-commit openfoam-vector-predictions-2026-05-09
PYTHONPATH=src python -m pinn_fluid.figures data/openfoam_vector_predictions_2026_05_09
```

Limitations:

- At Phase 23 close, Darcy remained a finite-difference pressure Laplace reference; Phase 24 later replaced the active Darcy experiment reference with OpenFOAM-sampled fields.
- Stokes, Oseen, and Navier-Stokes share one OpenFOAM incompressible laminar-flow sample as the pressure/velocity reference for this benchmark geometry.
- Reference residual diagnostics are still finite-difference post-processing of sampled fields, not native OpenFOAM residual histories.
- The analytic Poiseuille helper remains as a residual benchmark, not as active vector ground truth.
- The legacy Darcy-derived shared-patch vector helper remains only for the Phase 15 contract audit and is not used by the active vector experiment flow.

Verification:

- Red test: `python -m pytest tests/test_phase22_openfoam_experiment_references.py -q` failed while `ExperimentConfig.vector_reference_source` still defaulted to `manufactured`.
- Targeted tests: `python -m pytest tests/test_phase9_vertical_slice_experiments.py tests/test_phase10_consolidated_experiments.py tests/test_phase15_reference_contracts.py tests/test_phase17_stokes_reference.py tests/test_phase18_oseen_reference.py tests/test_phase19_navier_stokes_reference.py tests/test_phase20_comparison_run_contracts.py tests/test_phase22_openfoam_experiment_references.py -q`.
- Full suite: `python -m pytest`.
- Whitespace check: `git diff --check`.

## Phase 24: Align Darcy Boundary Conditions And OpenFOAM Ground Truth

Status: complete.

What changed:

- Changed active Darcy boundary residuals from pressure inlet/outlet to a velocity-inlet, pressure-outlet setup consistent with the Navier-Stokes experiment family.
- Darcy training now enforces inlet Darcy velocity `(0, -peak_velocity)`, outlet pressure `p=0`, and no-normal-flow walls.
- Changed `run_darcy_experiment(...)` to use the prepared OpenFOAM pressure/velocity sample as its active ground truth instead of the in-repo finite-difference Laplace reference.
- Added Darcy OpenFOAM reference fields to the normal `fields.npz` schema: reference pressure, velocity, `u`, `v`, speed, continuity, residual, and metadata.
- Kept the older `_fd_darcy_reference(...)` helpers as historical/contract-audit utilities; they are no longer the active Darcy experiment reference.

Reference generators after Phase 24:

- Darcy PINN experiment: `openfoam_simplefoam_shared_domain`
- Stokes PINN experiment: `openfoam_simplefoam_shared_domain`
- Oseen PINN experiment: `openfoam_simplefoam_shared_domain`
- Navier-Stokes PINN experiment: `openfoam_simplefoam_shared_domain`

Artifacts produced:

- Per-model `fields.npz`, `history.json`, `metrics.json`, loss plots, field plots, summary reports, and figure bundles remain the active artifact schema.
- Darcy `fields.npz` now records OpenFOAM-backed reference pressure and velocity components instead of finite-difference Darcy pressure/velocity components.
- Generated OpenFOAM/PINN outputs remain ignored under `data/` unless explicitly approved for commit.

Commands:

```bash
PYTHONPATH=src python -m pinn_fluid.result_procurement --output-dir data/openfoam_vector_predictions_2026_05_09 --grid-points 41 --training-steps 800 --hidden-width 24 --hidden-layers 2 --learning-rate 0.01 --seed 0 --viscosity 0.25 --peak-velocity 1.0 --darcy-reference-iterations 1200 --vector-reference-source openfoam --openfoam-end-time 50 --git-commit openfoam-vector-predictions-2026-05-09
PYTHONPATH=src python -m pinn_fluid.figures data/openfoam_vector_predictions_2026_05_09
```

Limitations:

- This aligns Darcy's active experiment boundary targets with the velocity-inlet/pressure-outlet benchmark used by Navier-Stokes, but Darcy still represents velocity as `-grad(p)`.
- The OpenFOAM sample is an incompressible laminar velocity-pressure solve, not a porous-media Darcy solver.
- Reference residual diagnostics are finite-difference post-processing of sampled OpenFOAM fields, not native OpenFOAM solver residual histories.
- The legacy finite-difference Darcy helpers remain in source for older audit tests and should not be treated as the active ground truth path.

Verification:

- Red tests: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-darcy-openfoam python -m pytest tests/test_phase9_vertical_slice_experiments.py tests/test_phase15_reference_contracts.py tests/test_phase16_darcy_ground_truth.py tests/test_phase20_comparison_run_contracts.py tests/test_phase22_openfoam_experiment_references.py -q` failed while Darcy still reported `fd_darcy_reference`.
- Targeted tests: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-darcy-openfoam python -m pytest tests/test_phase2_darcy_domain.py tests/test_phase4_darcy_training.py tests/test_phase9_vertical_slice_experiments.py tests/test_phase15_reference_contracts.py tests/test_phase16_darcy_ground_truth.py tests/test_phase20_comparison_run_contracts.py tests/test_phase22_openfoam_experiment_references.py -q`.
- Full suite: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-darcy-openfoam python -m pytest -q`.
- Real OpenFOAM smoke: wrote `data/openfoam_darcy_aligned_smoke_2026_05_11/openfoam_case`, ran `blockMesh`, `foamRun -solver incompressibleFluid`, and `postProcess -func sampleDict -latestTime`; OpenFOAM converged at time `41`, producing `postProcessing/sampleDict/41/sharedDomainGrid.xy` with shape `(81, 7)` and nonconstant pressure range `10.5543`.
- Smoke procurement with explicit sample path: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-darcy-openfoam python -m pinn_fluid.result_procurement --output-dir data/openfoam_darcy_aligned_smoke_2026_05_11 --grid-points 9 --training-steps 4 --hidden-width 6 --hidden-layers 1 --learning-rate 0.03 --seed 24 --viscosity 0.25 --peak-velocity 1.0 --darcy-reference-iterations 20 --vector-reference-source openfoam --openfoam-reference-sample-path data/openfoam_darcy_aligned_smoke_2026_05_11/openfoam_case/postProcessing/sampleDict/41/sharedDomainGrid.xy --openfoam-end-time 50 --git-commit darcy-openfoam-aligned-smoke-2026-05-11`.
- Smoke result: `reference_generators.darcy="openfoam_simplefoam_shared_domain"`, `all_metrics_finite=true`, and `all_histories_decreased=true`.

## Phase 25: Reject Shared OpenFOAM Ground Truth Across PDE Models

Status: complete.

What changed:

- Confirmed the previous comparison run reused one OpenFOAM pressure/velocity sample for Darcy, Stokes, Oseen, and Navier-Stokes. That is not an apples-to-apples PINN surrogate comparison across PDE models.
- Added a procurement contract that rejects a single `openfoam_reference_sample_path` for all flow models.
- Added `ExperimentConfig.openfoam_reference_sample_paths` so Darcy, Stokes, Oseen, and Navier-Stokes can each carry a distinct OpenFOAM sample path, imported field bundle, and metadata payload.
- Updated experiment reference names to model-specific generators: `openfoam_darcy_shared_domain`, `openfoam_stokes_shared_domain`, `openfoam_oseen_shared_domain`, and `openfoam_navier_stokes_shared_domain`.
- The CLI now accepts `--darcy-openfoam-reference-sample-path`, `--stokes-openfoam-reference-sample-path`, `--oseen-openfoam-reference-sample-path`, and `--navier-stokes-openfoam-reference-sample-path`.

Reference generators after this change:

- Darcy PINN experiment: `openfoam_darcy_shared_domain`
- Stokes PINN experiment: `openfoam_stokes_shared_domain`
- Oseen PINN experiment: `openfoam_oseen_shared_domain`
- Navier-Stokes PINN experiment: `openfoam_navier_stokes_shared_domain`

Artifacts produced:

- The artifact schema remains unchanged: per-model `fields.npz`, `history.json`, `metrics.json`, loss plots, field plots, summary reports, and figure bundles.
- Each model's `reference_metadata_json` now records its own sample path and model-specific reference generator.
- Generated OpenFOAM/PINN outputs remain ignored under `data/` unless explicitly approved for commit.

Limitations:

- This phase fixes the repository contract that allowed one OpenFOAM sample to be reused across PDE models.
- A scientifically complete apples-to-apples run still requires true model-specific OpenFOAM cases or solvers for Darcy, Stokes, Oseen, and Navier-Stokes before new comparison figures should be treated as valid ground truth.
- Stock OpenFOAM has Navier-Stokes-oriented incompressible solvers and `porousSimpleFoam`; a true Oseen-specific solve may require a custom OpenFOAM solver or another dedicated scientific-computing stack.

Verification so far:

- Red test: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-model-specific-openfoam python -m pytest tests/test_phase22_openfoam_experiment_references.py -q` failed because procurement accepted one shared sample and `ExperimentConfig.openfoam_reference_sample_paths` did not exist.
- Targeted tests: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-model-specific-openfoam python -m pytest tests/test_phase9_vertical_slice_experiments.py tests/test_phase10_consolidated_experiments.py tests/test_phase15_reference_contracts.py tests/test_phase16_darcy_ground_truth.py tests/test_phase17_stokes_reference.py tests/test_phase18_oseen_reference.py tests/test_phase19_navier_stokes_reference.py tests/test_phase20_comparison_run_contracts.py tests/test_phase22_openfoam_experiment_references.py -q`.

## Phase 26: Switch Active Dedicated Reference Path To FEniCSx

Status: in progress.

What changed:

- Switched `ExperimentConfig.vector_reference_source` from `openfoam` to `fenicsx`.
- Added `pinn_fluid.fenicsx_solvers` with FEniCSx metadata, field-artifact import, and a clear dependency boundary for missing `dolfinx`, `ufl`, `petsc4py`, and `mpi4py`.
- Added `ExperimentConfig.fenicsx_reference_sample_paths` for per-model FEniCSx field artifacts.
- Changed result procurement so active reference preparation accepts only `vector_reference_source="fenicsx"`.
- Added per-model FEniCSx CLI artifact paths: `--darcy-fenicsx-reference-sample-path`, `--stokes-fenicsx-reference-sample-path`, `--oseen-fenicsx-reference-sample-path`, and `--navier-stokes-fenicsx-reference-sample-path`.
- Left the OpenFOAM case-generation/import module in place as historical helper coverage, but it is no longer the active procurement reference source.

Reference generators after this change:

- Darcy PINN experiment: `fenicsx_darcy_shared_domain`
- Stokes PINN experiment: `fenicsx_stokes_shared_domain`
- Oseen PINN experiment: `fenicsx_oseen_shared_domain`
- Navier-Stokes PINN experiment: `fenicsx_navier_stokes_shared_domain`

Artifacts produced:

- Per-model `fields.npz`, `history.json`, `metrics.json`, loss plots, field plots, summary reports, and figure bundles remain the active artifact schema.
- FEniCSx field artifacts are imported from `.npz` files containing `coordinates`, `reference_pressure`, `reference_u`, and `reference_v`.
- Generated solver/PINN outputs remain ignored under `data/` unless explicitly approved for commit.

Limitations at Phase 26 close:

- Phase 26 only switched the active reference contract to FEniCSx artifact imports. Phase 27 subsequently implemented automatic per-model FEniCSx solve generation.
- The normal user Python used by tests could not import `dolfinx`, `ufl`, `petsc4py`, or `mpi4py`; the working FEniCSx stack is available through `/usr/bin/python3`.
- Residual diagnostics remained finite-difference post-processing on sampled fields.

Verification so far:

- Red test: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pytest tests/test_phase26_fenicsx_reference.py -q` failed while the default source was still `openfoam` and `fenicsx_reference_sample_paths` did not exist.
- Targeted tests: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pytest tests/test_phase9_vertical_slice_experiments.py tests/test_phase10_consolidated_experiments.py tests/test_phase15_reference_contracts.py tests/test_phase16_darcy_ground_truth.py tests/test_phase17_stokes_reference.py tests/test_phase18_oseen_reference.py tests/test_phase19_navier_stokes_reference.py tests/test_phase20_comparison_run_contracts.py tests/test_phase22_openfoam_experiment_references.py tests/test_phase26_fenicsx_reference.py -q`.

## Phase 27: FEniCSx Model-Specific Solver Generation

Status: complete.

What changed:

- Added model-specific FEniCSx solver generation in `pinn_fluid.fenicsx_solvers`.
- Added `generate_fenicsx_reference(...)`, which invokes `/usr/bin/python3 -m pinn_fluid.fenicsx_solvers solve ...` so the normal conda experiment process can use the system FEniCSx installation.
- Implemented deterministic per-model finite-element solves on the shared unit-square geometry:
  - Darcy: CG1 pressure solve with velocity recovered from `-grad(p)`.
  - Stokes: Taylor-Hood P2-P1 linear incompressible mixed solve.
  - Oseen: Taylor-Hood P2-P1 linearized mixed solve with convection velocity `(peak_velocity, 0.0)`.
  - Navier-Stokes: steady nonlinear Taylor-Hood P2-P1 mixed solve.
- Changed procurement so missing `fenicsx_reference_sample_paths` triggers per-model FEniCSx artifact generation instead of stopping at the previous not-implemented boundary.
- Added model-specific FEniCSx metadata for PDE model, formulation, solver stack, sample path, coordinate convention, and reference generator.
- Added Phase 27 tests for generated artifact paths, nonconstant reference pressures, model-specific metadata, and procurement integration without manufactured/OpenFOAM/shared fallback calls.

Reference generators after Phase 27:

- Darcy PINN experiment: `fenicsx_darcy_shared_domain`
- Stokes PINN experiment: `fenicsx_stokes_shared_domain`
- Oseen PINN experiment: `fenicsx_oseen_shared_domain`
- Navier-Stokes PINN experiment: `fenicsx_navier_stokes_shared_domain`

Artifacts produced:

- Generated FEniCSx artifacts are written under ignored `data/<run>/fenicsx_references/{darcy,stokes,oseen,navier_stokes}/fields.npz`.
- Each generated FEniCSx artifact contains `coordinates`, `reference_pressure`, `reference_u`, and `reference_v`.
- Procurement imports those artifacts and writes the established per-model `fields.npz`, `history.json`, `metrics.json`, raw loss/field plots, `run_manifest.json`, and summary reports.
- Figure generation writes `figures/figure_manifest.json`, field panels, convergence panels, scalar field images, and pressure-velocity quiver diagnostics.
- Generated `.npz`, `.json`, and `.png` outputs under `data/` remain ignored and were not committed.

Limitations:

- FEniCSx is available through `/usr/bin/python3`; the normal conda Python used by the test suite still does not import `dolfinx`, `ufl`, `petsc4py`, or `mpi4py`.
- The container sandbox blocks MPI socket initialization, so real FEniCSx smoke commands required unsandboxed execution.
- Reference residual diagnostics are still finite-difference post-processing on sampled fields, not native FEniCSx variational residual exports.
- The smoke run used a 5 by 5 sampling grid and 4 training steps only; it verifies the generation and artifact path, not final surrogate accuracy.
- The Oseen reference preserves the repository's existing convection velocity convention `(peak_velocity, 0.0)`.

Reproduction commands:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pinn_fluid.result_procurement --output-dir data/fenicsx_phase27_smoke_2026_05_11 --grid-points 5 --training-steps 4 --hidden-width 6 --hidden-layers 1 --learning-rate 0.03 --seed 27 --viscosity 0.25 --peak-velocity 1.0 --git-commit fenicsx-phase27-smoke
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pinn_fluid.figures data/fenicsx_phase27_smoke_2026_05_11
```

Verification:

- Red test: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pytest tests/test_phase27_fenicsx_solver_generation.py -q` failed because `FLOW_MODEL_METADATA` and `generate_fenicsx_reference(...)` did not exist.
- Targeted green tests: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pytest tests/test_phase27_fenicsx_solver_generation.py -q` passed with `2 passed`.
- Real FEniCSx generator smoke:
  - `PYTHONPATH=src /usr/bin/python3 -m pinn_fluid.fenicsx_solvers solve --model darcy --output-path /tmp/pinn_fenicsx_smoke/darcy/fields.npz --grid-points 5 --mesh-cells 4 --viscosity 0.25 --peak-velocity 1.0`
  - `PYTHONPATH=src /usr/bin/python3 -m pinn_fluid.fenicsx_solvers solve --model stokes --output-path /tmp/pinn_fenicsx_smoke/stokes/fields.npz --grid-points 5 --mesh-cells 4 --viscosity 0.25 --peak-velocity 1.0`
  - `PYTHONPATH=src /usr/bin/python3 -m pinn_fluid.fenicsx_solvers solve --model oseen --output-path /tmp/pinn_fenicsx_smoke/oseen/fields.npz --grid-points 5 --mesh-cells 4 --viscosity 0.25 --peak-velocity 1.0`
  - `PYTHONPATH=src /usr/bin/python3 -m pinn_fluid.fenicsx_solvers solve --model navier_stokes --output-path /tmp/pinn_fenicsx_smoke/navier_stokes/fields.npz --grid-points 5 --mesh-cells 4 --viscosity 0.25 --peak-velocity 1.0`
- Real smoke result: all four generated artifacts had shape `(25, 2)` for coordinates and nonconstant pressure ranges.
- End-to-end smoke: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pinn_fluid.result_procurement --output-dir data/fenicsx_phase27_smoke_2026_05_11 --grid-points 5 --training-steps 4 --hidden-width 6 --hidden-layers 1 --learning-rate 0.03 --seed 27 --viscosity 0.25 --peak-velocity 1.0 --git-commit fenicsx-phase27-smoke` completed with `all_metrics_finite=true`, `all_histories_decreased=true`, and distinct FEniCSx sample paths for all models.
- Figure smoke: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pinn_fluid.figures data/fenicsx_phase27_smoke_2026_05_11` completed and wrote `figures/figure_manifest.json`; it emitted the existing Matplotlib `tight_layout` warning.
- Targeted regression set: `PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-pinn-fenicsx python -m pytest tests/test_phase26_fenicsx_reference.py tests/test_phase27_fenicsx_solver_generation.py tests/test_phase9_vertical_slice_experiments.py tests/test_phase10_consolidated_experiments.py tests/test_phase15_reference_contracts.py tests/test_phase16_darcy_ground_truth.py tests/test_phase17_stokes_reference.py tests/test_phase18_oseen_reference.py tests/test_phase19_navier_stokes_reference.py tests/test_phase20_comparison_run_contracts.py tests/test_phase22_openfoam_experiment_references.py -q` passed with `26 passed, 7 warnings`.
