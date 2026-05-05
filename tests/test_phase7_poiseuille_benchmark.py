"""Tests for the analytic Poiseuille Navier-Stokes benchmark."""

import torch

from pinn_fluid.domains import interior_collocation_points
from pinn_fluid.models.navier_stokes import (
    poiseuille_channel_solution,
    poiseuille_pressure_drop,
    navier_stokes_residuals,
)


def test_poiseuille_channel_solution_has_expected_profile_and_pressure_drop():
    viscosity = 0.25
    peak_velocity = 1.5
    points = torch.tensor(
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    solution = poiseuille_channel_solution(
        points,
        viscosity=viscosity,
        peak_velocity=peak_velocity,
        pressure_reference=2.0,
    )

    assert set(solution) == {"u", "v", "pressure"}
    assert torch.allclose(
        solution["u"],
        torch.tensor([[0.0], [peak_velocity], [0.0]], dtype=torch.float64),
        atol=1e-10,
    )
    assert torch.allclose(solution["v"], torch.zeros_like(solution["v"]), atol=1e-10)
    assert torch.allclose(solution["pressure"][0], torch.tensor([2.0], dtype=torch.float64))
    assert torch.allclose(
        solution["pressure"][-1],
        torch.tensor(
            [2.0 - poiseuille_pressure_drop(viscosity, peak_velocity)],
            dtype=torch.float64,
        ),
    )


def test_poiseuille_channel_solution_has_zero_navier_stokes_residuals():
    viscosity = 0.25
    peak_velocity = 1.5
    points = interior_collocation_points(3, dtype=torch.float64).requires_grad_(True)
    solution = poiseuille_channel_solution(
        points,
        viscosity=viscosity,
        peak_velocity=peak_velocity,
    )

    residuals = navier_stokes_residuals(
        solution["u"],
        solution["v"],
        solution["pressure"],
        points,
        viscosity=viscosity,
    )

    zero = torch.zeros((points.shape[0], 1), dtype=torch.float64)
    assert torch.allclose(residuals["continuity"], zero, atol=1e-10)
    assert torch.allclose(residuals["x_momentum"], zero, atol=1e-10)
    assert torch.allclose(residuals["y_momentum"], zero, atol=1e-10)


def test_poiseuille_channel_solution_satisfies_horizontal_wall_no_slip():
    points = torch.tensor(
        [[0.2, 0.0], [0.7, 0.0], [0.2, 1.0], [0.7, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    solution = poiseuille_channel_solution(points, viscosity=1.0)

    assert torch.allclose(solution["u"], torch.zeros_like(solution["u"]), atol=1e-10)
    assert torch.allclose(solution["v"], torch.zeros_like(solution["v"]), atol=1e-10)
