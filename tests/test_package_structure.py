"""Import checks for the Phase 1 repository scaffold."""

import importlib
import sys


def test_root_package_imports_without_framework_side_effects():
    torch_loaded_before = "torch" in sys.modules

    package = importlib.import_module("pinn_fluid")

    assert package.PROJECT_NAME == "pinn-fluid"
    assert package.PHASE == "neural-fields"
    if not torch_loaded_before:
        assert "torch" not in sys.modules


def test_placeholder_namespaces_are_importable_and_inert():
    modules = [
        importlib.import_module("pinn_fluid.models"),
        importlib.import_module("pinn_fluid.models.darcy"),
        importlib.import_module("pinn_fluid.models.fields"),
        importlib.import_module("pinn_fluid.models.stokes"),
        importlib.import_module("pinn_fluid.solvers"),
        importlib.import_module("pinn_fluid.utils"),
    ]

    for module in modules:
        assert isinstance(module.__all__, list)
