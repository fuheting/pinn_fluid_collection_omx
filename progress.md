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
| Stokes | `shared_patch_vector_reference_from_fd_darcy` | Darcy pressure Laplace equation with velocity from negative pressure gradient | demo-only |
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

## Continuing Guidance

Required workflow for future phases or follow-up studies:

- Read `README.md` and this file.
- Inspect current model, solver, benchmark, and test patterns before editing.
- Write failing tests first, then implement narrowly.
- Update docs and progress in the same phase.
- Verify with targeted tests, `python -m pytest`, `git diff --check`, and diagnostics or `py_compile` where useful.
- Commit with the Lore protocol and push `origin main`.
