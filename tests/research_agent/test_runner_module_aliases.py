"""Canonical ownership contracts for deterministic runner modules."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest


RUNNER_MODULES = (
    "deterministic_descriptive",
    "deterministic_missingness",
    "deterministic_robustness",
    "trajectory_stability_executor",
)

RETIRED_PRIMARY_RUNNER_MODULES = (
    "deterministic_causal",
    "deterministic_clustering",
    "deterministic_cohort_flow",
    "deterministic_ordinal",
    "deterministic_sensitivity",
    "deterministic_survival",
)


@pytest.mark.parametrize("leaf", RUNNER_MODULES)
def test_runner_has_one_canonical_home(leaf: str) -> None:
    module = importlib.import_module(
        f"easyicu.research_agent.execution.runners.{leaf}"
    )
    assert module.__name__.endswith(f"execution.runners.{leaf}")
    assert "/execution/runners/" in Path(module.__file__).as_posix()


@pytest.mark.parametrize("leaf", RETIRED_PRIMARY_RUNNER_MODULES)
def test_retired_primary_runner_has_no_importable_implementation(leaf: str) -> None:
    assert (
        importlib.util.find_spec(f"easyicu.research_agent.execution.runners.{leaf}")
        is None
    )


def test_host_pipeline_imports_runner_implementations_from_canonical_package() -> None:
    for module_name in (
        "easyicu.research_agent.pipeline",
        "easyicu.research_agent.pipeline_execute",
        "easyicu.research_agent.gates.contract",
    ):
        tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(
            name.startswith("deterministic_")
            or name == "trajectory_stability_executor"
            for name in imported
        )


def test_generated_runner_adapters_use_canonical_import_paths() -> None:
    descriptive = importlib.import_module(
        "easyicu.research_agent.execution.runners.deterministic_descriptive"
    )
    robustness = importlib.import_module(
        "easyicu.research_agent.execution.runners.deterministic_robustness"
    )
    trajectory = importlib.import_module(
        "easyicu.research_agent.execution.runners.trajectory_stability_executor"
    )
    assert (
        "easyicu.research_agent.execution.runners.deterministic_descriptive"
        in descriptive.absolute_risk_context_code()
    )
    assert (
        "easyicu.research_agent.execution.runners.deterministic_robustness"
        in robustness.robustness_sensitivity_preflight_code()
    )
    assert (
        "easyicu.research_agent.execution.runners.trajectory_stability_executor"
        in inspect.getsource(trajectory.trajectory_stability_executor_code)
    )


def test_live_auxiliary_registry_uses_canonical_runner_coordinates() -> None:
    registry = importlib.import_module(
        "easyicu.research_agent.planning.capability_registry"
    )
    assert {runner.module for runner in registry.AUXILIARY_DETERMINISTIC_RUNNERS} == {
        "execution.runners.deterministic_descriptive",
        "execution.runners.deterministic_missingness",
        "execution.runners.deterministic_robustness",
        "execution.runners.trajectory_stability_executor",
    }
