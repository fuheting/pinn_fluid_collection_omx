# PINN Fluid Mechanics

Research scaffold for studying Physics-Informed Neural Networks applied to fluid mechanics. The project is organized as a phased repository: begin with a reproducible PyTorch-oriented foundation, then add governing-equation examples from simple porous-media flow through laminar incompressible flow.

## Research Objective

The long-term objective is to evaluate how PINNs converge on fluid mechanics problems when conservation laws and constitutive relations are enforced through loss terms. Each phase should add one problem family, document the governing equations, define verification targets, and preserve a clear connection between the learned field and the physical residuals.

## Current Status

Phase 1 is complete for this scaffold pass. The repository now has structure, documentation, dependency declarations, inert source placeholders, a retained data directory, and import tests. It does not implement a fluid model, neural network, residual loss, solver, training loop, or numerical validation.

## Planned Phases

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Repository scaffold, documentation, PyTorch-oriented dependency direction, importable placeholders, import tests | Complete for scaffold pass |
| 2 | First PINN model for Darcy flow | Pending |
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

The `src/pinn_fluid` package is intentionally inert in Phase 1. Its modules exist so later phases have importable extension points, but they do not define model, solver, residual, PDE, or training APIs yet.

## Environment Direction

PyTorch is the intended machine-learning framework for future phases. Phase 1 tests only check package structure and import behavior; they do not require importing PyTorch.

Install dependencies in a virtual environment when implementation work begins:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the current scaffold tests:

```bash
python -m pytest
```

## Continuing With Phase 2

Before adding Darcy flow, update `progress.md` with the new phase start, add a focused plan for the governing equation and boundary conditions, and introduce tests that validate the intended residual or analytic reference behavior. Keep Phase 2 narrowly scoped so the first physical model can be reviewed and verified independently.
