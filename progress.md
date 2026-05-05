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
| 9 | CFD-backed Darcy and Poiseuille/Navier-Stokes vertical slice | Complete |
| 10 | Stokes/Oseen experiment extension and consolidated report | Complete |
| 11 | Result-procurement runner and manifest | Complete |

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

## Phase 9: Darcy And Poiseuille/Navier-Stokes Vertical Slice

Status: complete for the lightweight deterministic vertical slice.

Completed in this pass:

- Added a deterministic in-repo finite-difference Laplace reference for the shared Darcy pressure setup.
- Added Darcy experiment orchestration that saves predicted pressure, reference pressure, predicted/reference velocities, residual fields, objective history, component histories, plots, and finite metrics.
- Added Poiseuille/Navier-Stokes experiment orchestration that trains a small velocity-pressure field against the channel reference while recording continuity, x-momentum, y-momentum, and boundary objectives.
- Added tests that assert decreasing histories, finite metrics, and saved numeric/plot artifacts for both vertical-slice models.

Remaining:

- Run longer local studies outside the regression defaults before making accuracy or physics-ranking claims.

## Phase 10: Stokes/Oseen Extension And Consolidated Report

Status: complete for the first consolidated experiment pass.

Completed in this pass:

- Added Stokes and Oseen experiment runs against the same Poiseuille channel reference used by the Navier-Stokes vertical slice.
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
- Build a vertical slice first for Darcy and Poiseuille/Navier-Stokes rather than all models at once.
- Save reproducible numeric artifacts plus plots.
- Use simple deterministic grids and lightweight local runs.
- Let the implementing agent choose grid sizes, training steps, file formats, and plot layouts when documented and lightweight.
- Treat first-pass success as decreasing histories plus finite reported metrics, not strict accuracy thresholds or final claims about physics-complexity ranking.

Completed phase order:

1. Phase 8: experiment result schema and component-wise training history capture.
2. Phase 9: Darcy and Poiseuille/Navier-Stokes CFD-backed vertical-slice experiments.
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

- Phase 12 should add standalone figure-generation utilities from saved artifacts rather than expanding the runner.
- Phase 13 should use the runner for sanity and paper-demo tiers without committing generated data artifacts.

## Next Result-Procurement Phases

Do not ask the next agent to complete the entire result study in one uninterrupted run. Preserve the phased style and make each phase testable.

Recommended next phases:

1. Phase 12: Add figure-generation utilities for saved experiment artifacts with tests against small synthetic `fields.npz` and `history.json` inputs.
2. Phase 13: Run the sanity tier and laptop-moderate paper-demo tier locally, record the manifest, and summarize generated artifact paths without committing bulky outputs.
3. Phase 14: Update documentation with result-procurement commands, figure inventory, and known limitations from the actual run.

Suggested initial run tiers:

- Sanity tier: `grid_points=9`, `training_steps=80`, `hidden_width=12`, `hidden_layers=1`, `learning_rate=0.02`.
- Paper-demo tier: `grid_points=31`, `training_steps=800`, `hidden_width=24`, `hidden_layers=2`, `learning_rate=0.01`, `seed=0`, `viscosity=0.25`, `peak_velocity=1.0`, `darcy_reference_iterations=1200`.
- If a paper-demo model does not reduce total history, rerun that tier with `learning_rate=0.005` and `training_steps=1200`.

Minimum paper-demo outputs:

- Per-model field panels for Darcy, Stokes, Oseen, and Navier-Stokes.
- Per-model convergence panels showing total and component losses.
- A cross-model metrics table derived from `summary/cross_model_report.json`.
- A `run_manifest.json` that records git commit, config values, output directories, metric summary, and whether each history decreased.

Non-goals for the next pass:

- Do not redefine the unit-square domain or residual APIs.
- Do not claim final physical accuracy or rank PINN effectiveness from the moderate demo tier.
- Do not commit generated experiment artifacts unless a later phase explicitly approves that policy change.
- Do not add dependencies unless the existing declared dependencies are insufficient and the change is covered by tests.

Required workflow for future phases:

- Read `README.md` and this file.
- Inspect current model, solver, benchmark, and test patterns before editing.
- Write failing tests first, then implement narrowly.
- Update docs and progress in the same phase.
- Verify with targeted tests, `python -m pytest`, `git diff --check`, and diagnostics or `py_compile` where useful.
- Commit with the Lore protocol and push `origin main`.
