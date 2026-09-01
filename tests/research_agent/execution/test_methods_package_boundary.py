"""Contract tests for the research-agent statistical methods package."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


METHOD_MODULES = (
    "conformal",
    "decision_curve",
    "delong_auc",
    "fairness",
    "missing",
    "missing_data",
    "multiple_testing",
    "ordered_trends",
    "ph_schoenfeld",
    "rmst",
    "sensitivity",
    "temporal_features",
)


def test_statistical_methods_are_owned_by_methods_package() -> None:
    package_root = (
        Path(__file__).resolve().parents[3] / "src" / "easyicu" / "research_agent"
    )
    methods_root = package_root / "methods"

    assert (methods_root / "__init__.py").is_file()
    for module_name in METHOD_MODULES:
        assert (methods_root / f"{module_name}.py").is_file()
        assert not (package_root / f"{module_name}.py").exists()


@pytest.mark.parametrize("module_name", METHOD_MODULES)
def test_method_modules_import_from_canonical_package(module_name: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.methods.{module_name}")
    assert module.__name__.endswith(f"methods.{module_name}")


def test_existing_root_convenience_exports_remain_available() -> None:
    import easyicu.research_agent as research_agent

    for name in (
        "build_multiple_testing_report",
        "compute_e_value",
        "mice_impute",
        "run_subgroup_analysis",
    ):
        assert callable(getattr(research_agent, name))


def test_temporal_features_use_the_canonical_methods_module() -> None:
    canonical = importlib.import_module(
        "easyicu.research_agent.methods.temporal_features"
    )
    assert callable(canonical.onset_times)
    assert callable(canonical.incident_outcome_cohort)
