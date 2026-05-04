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
| 4 | Oseen equation formulation and convergence checks | Pending |
| 5 | Laminar Navier-Stokes formulation and common channel-flow cases | Pending |
| 6 | Documentation consolidation and cleanup across implemented models | Pending |

## Phase 2: Darcy Flow

Status: domain and residual foundation complete.

Completed in this pass:

- Defined the shared unit-square flow domain.
- Added model-agnostic inlet, outlet, and wall boundary patches.
- Added Darcy-flow loss weights for interior, inlet, outlet, and wall residuals.
- Added PyTorch autograd helpers for pressure gradients, Laplace residuals, Darcy velocity, and boundary residuals.
- Added tests for the patch definition, loss weights, harmonic pressure residual, velocity derivation, and boundary residual behavior.

Remaining:

- Add a lightweight training loop and convergence smoke test.
- Add example output documentation once training exists.

## Continuation Notes

The next model-training step should build on `pinn_fluid.domains.unit_square_flow_patches`, `pinn_fluid.domains.interior_collocation_points`, `pinn_fluid.domains.boundary_collocation_points`, `pinn_fluid.models.darcy.DarcyPressureField`, and `pinn_fluid.models.stokes.StokesVelocityPressureField` rather than redefining the domain or adding a second field abstraction. Keep training tests small enough for local execution.

## Phase 3: Stokes Flow

Status: residual foundation complete.

Completed in this pass:

- Added Stokes-flow residual helpers for steady incompressible low-Reynolds-number flow.
- Added continuity, x-momentum, and y-momentum residuals for `(u, v, p)` fields.
- Added a no-slip wall residual helper.
- Added Stokes loss weights for continuity, momentum, inlet, outlet, and wall residuals.
- Added tests for zero residuals under constant pressure and zero velocity, linear-field momentum behavior, and no-slip residual behavior.

Remaining:

- Add a training smoke test around the minimal `(u, v, p)` neural field.

## Shared Collocation Foundation

Status: complete for the sampling foundation pass.

Completed in this pass:

- Added deterministic interior collocation sampling for the shared unit-square domain.
- Added deterministic inlet, outlet, and wall boundary sampling from the existing shared patch abstraction.
- Added outward wall normals for Darcy no-normal-flow and Stokes no-slip boundary residual construction.
- Added Stokes boundary targets derived from the shared inlet, outlet, and wall patch definitions without mutating the shared patch API.
- Added tests for sample placement, wall normals, input validation, and Stokes target mapping.

Remaining:

- Add lightweight training loops and convergence smoke tests.

## Shared Neural Field Foundation

Status: complete for the neural-field foundation pass.

Completed in this pass:

- Added a reusable `MLPField` for coordinate-to-field modules.
- Added `DarcyPressureField` for scalar pressure predictions on `(x, y)`.
- Added `StokesVelocityPressureField` for named `u`, `v`, and `pressure` predictions on `(x, y)`.
- Added tests for output shapes, dtype preservation, invalid architecture settings, and named Stokes outputs.

Remaining:

- Add Darcy loss assembly and a lightweight training smoke test.
- Add Stokes loss assembly and a lightweight training smoke test after the Darcy training path is stable.
