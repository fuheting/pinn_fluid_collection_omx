"""Tests for minimal neural field modules."""

import torch

from pinn_fluid.models.darcy import DarcyPressureField
from pinn_fluid.models.fields import MLPField
from pinn_fluid.models.stokes import StokesVelocityPressureField


def test_mlp_field_maps_coordinates_to_configured_output_width():
    torch.manual_seed(0)
    field = MLPField(input_dim=2, output_dim=4, hidden_width=8, hidden_layers=2)
    coordinates = torch.zeros((5, 2), dtype=torch.float32)

    values = field(coordinates)

    assert values.shape == (5, 4)
    assert values.dtype == coordinates.dtype


def test_mlp_field_rejects_invalid_architecture_settings():
    invalid_settings = [
        {"input_dim": 0, "output_dim": 1},
        {"input_dim": 2, "output_dim": 0},
        {"input_dim": 2, "output_dim": 1, "hidden_width": 0},
        {"input_dim": 2, "output_dim": 1, "hidden_layers": -1},
    ]

    for settings in invalid_settings:
        try:
            MLPField(**settings)
        except ValueError as exc:
            assert "positive" in str(exc) or "non-negative" in str(exc)
        else:
            raise AssertionError(f"expected invalid settings to fail: {settings}")


def test_darcy_pressure_field_returns_scalar_pressure():
    torch.manual_seed(1)
    field = DarcyPressureField(hidden_width=6, hidden_layers=1)
    coordinates = torch.tensor(
        [[0.25, 0.50], [0.75, 0.25]],
        dtype=torch.float32,
    )

    pressure = field(coordinates)

    assert pressure.shape == (2, 1)


def test_stokes_velocity_pressure_field_splits_outputs_by_name():
    torch.manual_seed(2)
    field = StokesVelocityPressureField(hidden_width=6, hidden_layers=1)
    coordinates = torch.tensor(
        [[0.25, 0.50], [0.75, 0.25]],
        dtype=torch.float32,
    )

    outputs = field(coordinates)

    assert set(outputs) == {"u", "v", "pressure"}
    assert outputs["u"].shape == (2, 1)
    assert outputs["v"].shape == (2, 1)
    assert outputs["pressure"].shape == (2, 1)
