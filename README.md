# PINN Fluid Mechanics

Research scaffold for studying Physics-Informed Neural Networks applied to fluid mechanics. The project is organized as a phased repository: begin with a reproducible PyTorch-oriented foundation, then add governing-equation examples from simple porous-media flow through laminar incompressible flow.

## Research Objective

The long-term objective is to evaluate how PINNs converge on fluid mechanics problems when conservation laws and constitutive relations are enforced through loss terms. Each phase should add one problem family, document the governing equations, define verification targets, and preserve a clear connection between the learned field and the physical residuals.

## Current Status

Phase 3 has a Stokes-flow residual foundation on the same shared unit-square domain. The repository now defines model-agnostic inlet/outlet/wall patches, shared deterministic collocation sampling, Darcy residual helpers, Stokes residual helpers, minimal Darcy and Stokes neural fields, and a lightweight Darcy training smoke loop. Stokes training remains pending.

## Planned Phases

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Repository scaffold, documentation, PyTorch-oriented dependency direction, importable placeholders, import tests | Complete for scaffold pass |
| 2 | Shared unit-square flow domain and first Darcy-flow residual instance | Residual foundation complete |
| 3 | Stokes flow at low Reynolds number | Residual foundation complete |
| shared foundation | Collocation sampling and Stokes boundary-target mapping | Foundation complete |
| shared foundation | Minimal Darcy and Stokes neural fields | Foundation complete |
| shared foundation | Darcy loss assembly and training smoke loop | Foundation complete |
| 4 | Reduced-order Navier-Stokes through Oseen equations | Pending |
| 5 | Laminar Navier-Stokes examples such as channel or Poiseuille flow | Pending |
| 6 | Documentation, cleanup, and consolidated test coverage | Pending |

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

The `src/pinn_fluid` package now contains the shared domain abstraction, Darcy-flow residual utilities, Stokes-flow residual utilities, minimal neural field modules, and Darcy smoke-training utilities. Stokes training orchestration remains pending.

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

Next, add the Stokes training smoke loop around the existing Stokes residual APIs.
