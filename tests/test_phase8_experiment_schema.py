"""Tests for experiment result schema and component histories."""

import pytest

from pinn_fluid.experiments import ExperimentResult, TrainingHistory


def test_training_history_records_total_and_component_losses_by_iteration():
    history = TrainingHistory(
        total=[3.0, 2.0, 1.0],
        components={
            "residual": [2.0, 1.5, 0.5],
            "boundary": [1.0, 0.5, 0.5],
        },
    )

    assert history.steps == 2
    assert history.reduced is True
    assert history.component_names == ("boundary", "residual")
    assert history.to_json_dict() == {
        "total": [3.0, 2.0, 1.0],
        "components": {
            "boundary": [1.0, 0.5, 0.5],
            "residual": [2.0, 1.5, 0.5],
        },
    }


def test_training_history_rejects_component_length_mismatches():
    with pytest.raises(ValueError, match="same length"):
        TrainingHistory(total=[1.0, 0.5], components={"residual": [1.0]})


def test_experiment_result_schema_serializes_metrics_history_and_artifacts():
    history = TrainingHistory(total=[2.0, 1.0], components={"residual": [2.0, 1.0]})
    result = ExperimentResult(
        model="darcy",
        reference="finite_difference_laplace",
        grid_shape=(4, 4),
        history=history,
        metrics={"pressure_l2": 0.25},
        artifacts={"fields_npz": "artifacts/darcy/fields.npz"},
    )

    payload = result.to_json_dict()

    assert payload["model"] == "darcy"
    assert payload["reference"] == "finite_difference_laplace"
    assert payload["grid_shape"] == [4, 4]
    assert payload["history"]["total"] == [2.0, 1.0]
    assert payload["metrics"] == {"pressure_l2": 0.25}
    assert payload["artifacts"] == {"fields_npz": "artifacts/darcy/fields.npz"}
