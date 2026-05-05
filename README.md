# PINN Fluid Mechanics

Research scaffold for studying Physics-Informed Neural Networks applied to fluid mechanics. The project is organized as a phased repository: begin with a reproducible PyTorch-oriented foundation, then add governing-equation examples from simple porous-media flow through laminar incompressible flow.

## Research Objective

The long-term objective is to evaluate how PINNs converge on fluid mechanics problems when conservation laws and constitutive relations are enforced through loss terms. Each phase should add one problem family, document the governing equations, define verification targets, and preserve a clear connection between the learned field and the physical residuals.

## Current Status

The current phase adds an analytic Poiseuille-style Navier-Stokes benchmark on the same shared unit-square domain. The repository defines model-agnostic inlet/outlet/wall patches, shared deterministic collocation sampling, Darcy residual helpers, Stokes residual helpers, Oseen residual helpers, Navier-Stokes residual helpers, minimal Darcy and Stokes neural fields, lightweight Darcy, Stokes, Oseen, and Navier-Stokes training smoke loops, phase-ordered loss summaries for those smoke loops, and a closed-form laminar channel reference that has zero steady Navier-Stokes residual under the existing autograd helpers.

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

The `src/pinn_fluid` package now contains the shared domain abstraction, Darcy-flow residual utilities, Stokes-flow residual utilities, Oseen residual utilities, Navier-Stokes residual utilities, minimal neural field modules, smoke-training utilities for Darcy, Stokes, Oseen, and Navier-Stokes flow, and a consolidated smoke benchmark summary module.

## Shared Example Domain

All flow models should start from the same unit-square example unless a later phase explicitly introduces a new benchmark:

```text
Omega = [0, 1] x [0, 1]
```

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

Next, compare a trained Navier-Stokes field against the analytic Poiseuille reference on a lightweight documented example.
