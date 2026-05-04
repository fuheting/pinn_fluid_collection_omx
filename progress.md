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
| 5 | Laminar Navier-Stokes formulation and common channel-flow cases | Pending |
| 6 | Documentation consolidation and cleanup across implemented models | Pending |

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
