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
| 2 | Darcy flow domain, residual formulation, training script, and validation tests | In progress |
| 3 | Stokes flow PINN formulation and tests | Pending |
| 4 | Oseen equation formulation and convergence checks | Pending |
| 5 | Laminar Navier-Stokes formulation and common channel-flow cases | Pending |
| 6 | Documentation consolidation and cleanup across implemented models | Pending |

## Phase 2: Darcy Flow

Status: domain and residual foundation implemented.

Completed in this pass:

- Defined the shared unit-square flow domain.
- Added model-agnostic inlet, outlet, and wall boundary patches.
- Added Darcy-flow loss weights for interior, inlet, outlet, and wall residuals.
- Added PyTorch autograd helpers for pressure gradients, Laplace residuals, Darcy velocity, and boundary residuals.
- Added tests for the patch definition, loss weights, harmonic pressure residual, velocity derivation, and boundary residual behavior.

Remaining:

- Add collocation-point sampling for the interior and boundary patches.
- Add a minimal pressure network for `p_theta(x, y)`.
- Add a lightweight training loop and convergence smoke test.
- Add example output documentation once training exists.

## Continuation Notes

The next Phase 2 step should build on `pinn_fluid.domains.unit_square_flow_patches` and `pinn_fluid.models.darcy` rather than redefining the domain. Keep training tests small enough for local execution.
