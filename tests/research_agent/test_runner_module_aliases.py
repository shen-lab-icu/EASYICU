"""Compatibility contracts for deterministic-runner module migration."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

RUNNER_MODULE_ALIASES = tuple(
    (
        f"easyicu.research_agent.{leaf}",
        f"easyicu.research_agent.execution.runners.{leaf}",
    )
    for leaf in (
        "deterministic_causal",
        "deterministic_clustering",
        "deterministic_cohort_flow",
        "deterministic_descriptive",
        "deterministic_missingness",
        "deterministic_ordinal",
        "deterministic_robustness",
        "deterministic_sensitivity",
        "deterministic_survival",
        "trajectory_stability_executor",
    )
)


@pytest.mark.parametrize("legacy,canonical", RUNNER_MODULE_ALIASES)
def test_legacy_runner_path_is_the_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__
    assert "/execution/runners/" in str(Path(new_module.__file__).as_posix())


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "pipeline_first"))
def test_runner_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {RUNNER_MODULE_ALIASES!r}
if {order!r} == 'pipeline_first':
    importlib.import_module('easyicu.research_agent.pipeline_execute')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    first = importlib.import_module(names[0])
    second = importlib.import_module(names[1])
    assert first is second, (legacy, canonical, first, second)
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_host_pipeline_imports_runner_implementations_from_canonical_package() -> None:
    legacy_modules = {
        "deterministic_descriptive",
        "deterministic_missingness",
        "deterministic_robustness",
        "trajectory_stability_executor",
    }
    for module_name in (
        "easyicu.research_agent.pipeline",
        "easyicu.research_agent.pipeline_execute",
        "easyicu.research_agent.gates.contract",
    ):
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))
        legacy_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module in legacy_modules
        }
        assert not legacy_imports


def test_generated_runner_adapters_keep_archive_compatible_import_paths() -> None:
    descriptive = importlib.import_module(RUNNER_MODULE_ALIASES[3][1])
    robustness = importlib.import_module(RUNNER_MODULE_ALIASES[6][1])
    trajectory = importlib.import_module(RUNNER_MODULE_ALIASES[9][1])

    assert (
        "easyicu.research_agent.deterministic_descriptive"
        in descriptive.absolute_risk_context_code()
    )
    assert (
        "easyicu.research_agent.deterministic_robustness"
        in robustness.robustness_sensitivity_preflight_code()
    )
    source = inspect.getsource(trajectory.trajectory_stability_executor_code)
    assert "easyicu.research_agent.trajectory_stability_executor" in source


def test_live_auxiliary_registry_coordinates_use_canonical_runner_modules() -> None:
    registry = importlib.import_module("easyicu.research_agent.capability_registry")
    modules = {runner.module for runner in registry.AUXILIARY_DETERMINISTIC_RUNNERS}
    assert modules == {
        "execution.runners.deterministic_descriptive",
        "execution.runners.deterministic_missingness",
        "execution.runners.deterministic_robustness",
        "execution.runners.trajectory_stability_executor",
    }
