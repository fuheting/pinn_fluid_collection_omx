from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pinn_fluid.dedicated_solvers import OPENFOAM_REFERENCE_METADATA
from pinn_fluid.experiments import ExperimentConfig


def write_fake_openfoam_sample(path: Path, grid_points: int) -> Path:
    values = np.linspace(0.0, 1.0, grid_points)
    lines = ["# x y z p Ux Uy Uz"]
    for x in values:
        for y in values:
            pressure = 1.0 - y + 0.1 * x
            u = 0.2 * x * (1.0 - x)
            v = -y * (1.0 - 0.25 * x)
            lines.append(f"{x} {y} 0.0 {pressure} {u} {v} 0.0")
    path.write_text("\n".join(lines))
    return path


def fake_openfoam_reference_fields(grid_points: int) -> dict[str, np.ndarray]:
    values = np.linspace(0.0, 1.0, grid_points)
    pressure = []
    u = []
    v = []
    for x in values:
        for y in values:
            pressure.append([1.0 - y + 0.1 * x])
            u.append([0.2 * x * (1.0 - x)])
            v.append([-y * (1.0 - 0.25 * x)])
    return {
        "reference_pressure": np.asarray(pressure, dtype=np.float64),
        "reference_u": np.asarray(u, dtype=np.float64),
        "reference_v": np.asarray(v, dtype=np.float64),
    }


def fake_openfoam_metadata() -> dict[str, object]:
    metadata = dict(OPENFOAM_REFERENCE_METADATA)
    metadata["sample_path"] = "test://fake-openfoam-sample"
    return metadata


@pytest.fixture
def openfoam_config_factory():
    def factory(**kwargs) -> ExperimentConfig:
        config = ExperimentConfig(**kwargs)
        return replace(
            config,
            vector_reference_source="openfoam",
            vector_reference_fields=fake_openfoam_reference_fields(config.grid_points),
            vector_reference_metadata=fake_openfoam_metadata(),
        )

    return factory


@pytest.fixture
def openfoam_sample_writer():
    return write_fake_openfoam_sample
