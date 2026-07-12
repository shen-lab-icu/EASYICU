"""Contract tests for the research-agent statistical methods package."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


METHOD_MODULES = (
    "conformal",
    "decision_curve",
    "delong_auc",
    "evalue",
    "fairness",
    "missing",
    "missing_data",
    "multiple_testing",
    "ordered_trends",
    "ph_schoenfeld",
    "rmst",
    "sensitivity",
    "survival",
    "temporal_features",
)

LEGACY_METHOD_SHIMS = {"temporal_features"}


def test_statistical_methods_are_owned_by_methods_package() -> None:
    package_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
    )
    methods_root = package_root / "methods"

    assert (methods_root / "__init__.py").is_file()
    for module_name in METHOD_MODULES:
        assert (methods_root / f"{module_name}.py").is_file()
        legacy_path = package_root / f"{module_name}.py"
        assert legacy_path.exists() is (module_name in LEGACY_METHOD_SHIMS)


@pytest.mark.parametrize("module_name", METHOD_MODULES)
def test_method_modules_import_from_canonical_package(module_name: str) -> None:
    module = importlib.import_module(
        f"easyicu.research_agent.methods.{module_name}"
    )
    assert module.__name__.endswith(f"methods.{module_name}")


def test_existing_root_convenience_exports_remain_available() -> None:
    import easyicu.research_agent as research_agent

    for name in (
        "build_multiple_testing_report",
        "compute_e_value",
        "fit_cox_model",
        "mice_impute",
        "run_subgroup_analysis",
    ):
        assert callable(getattr(research_agent, name))


def test_saved_temporal_feature_scripts_keep_their_legacy_import() -> None:
    from easyicu.research_agent import temporal_features as legacy
    from easyicu.research_agent.methods import temporal_features as canonical

    assert legacy.onset_times is canonical.onset_times
    assert legacy.incident_outcome_cohort is canonical.incident_outcome_cohort
