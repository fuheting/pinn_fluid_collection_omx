"""Tests for consolidated smoke benchmark summaries."""

import pytest

from pinn_fluid.benchmarks import (
    SmokeBenchmarkConfig,
    summarize_loss_history,
    run_smoke_benchmarks,
)


def test_summarize_loss_history_reports_reduction_metadata():
    result = summarize_loss_history("darcy", [4.0, 3.0, 1.0])

    assert result.model == "darcy"
    assert result.initial_loss == 4.0
    assert result.final_loss == 1.0
    assert result.steps == 2
    assert result.reduction == 3.0
    assert result.reduction_ratio == 0.75
    assert result.reduced is True


def test_summarize_loss_history_rejects_empty_histories():
    with pytest.raises(ValueError, match="history must contain at least one loss"):
        summarize_loss_history("darcy", [])


def test_run_smoke_benchmarks_returns_phase_ordered_loss_summaries():
    config = SmokeBenchmarkConfig(
        seed=0,
        hidden_width=8,
        hidden_layers=1,
        interior_points_per_axis=2,
        boundary_points_per_patch=2,
        steps=20,
        learning_rate=0.03,
        viscosity=1.0,
        oseen_convection_velocity=(1.0, 0.0),
    )

    results = run_smoke_benchmarks(config)

    assert [result.model for result in results] == [
        "darcy",
        "stokes",
        "oseen",
        "navier_stokes",
    ]
    assert all(result.steps == 20 for result in results)
    assert all(result.initial_loss > result.final_loss for result in results)
    assert all(result.reduced for result in results)
