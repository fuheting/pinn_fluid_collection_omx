# PINN Fluid Mechanics

Research scaffold for studying Physics-Informed Neural Networks applied to fluid mechanics. The project is organized as a phased repository: begin with a reproducible PyTorch-oriented foundation, then add governing-equation examples from simple porous-media flow through laminar incompressible flow.

## Research Objective

The long-term objective is to evaluate how PINNs converge on fluid mechanics problems when conservation laws and constitutive relations are enforced through loss terms. Each phase should add one problem family, document the governing equations, define verification targets, and preserve a clear connection between the learned field and the physical residuals.

## Current Status

Phase 2 has started with the shared flow-domain definition and a Darcy-flow residual instance. The repository now defines the unit-square domain, inlet/outlet/wall boundary patches, Darcy loss weights, autograd residual helpers, and tests for the analytic residual behavior. It does not yet include a trained PINN workflow.

## Planned Phases

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Repository scaffold, documentation, PyTorch-oriented dependency direction, importable placeholders, import tests | Complete for scaffold pass |
| 2 | Shared unit-square flow domain and first Darcy-flow residual instance | In progress |
| 3 | Stokes flow at low Reynolds number | Pending |
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

The `src/pinn_fluid` package now contains the shared domain abstraction and the first Darcy-flow residual utilities. Training orchestration remains pending.

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

- interior residual `p_xx + p_yy`
- Darcy velocity `u = -grad(p)`
- inlet residual `p - 1`
- outlet residual `p`
- wall residual `grad(p) dot n`
- default loss weights: interior `1`, inlet `10`, outlet `10`, wall `1`

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

## Continuing Phase 2

Next, add sampling utilities for interior and boundary collocation points, then add a minimal pressure network and training loop around the existing Darcy residual API.
