"""Ownership contracts for deterministic robustness runtime modules."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROBUSTNESS_MODULES = ("estimators", "panel", "primary_effect")
RETIRED_MODULES = ("estimators", "robustness_panel", "pipeline_primary_effect")


@pytest.mark.parametrize("leaf", ROBUSTNESS_MODULES)
def test_robustness_module_has_one_canonical_home(leaf: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.robustness.{leaf}")
    assert module.__name__ == f"easyicu.research_agent.robustness.{leaf}"
    assert "/robustness/" in Path(module.__file__).as_posix()


def test_robustness_package_is_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.robustness'
importlib.import_module(package)
loaded = sorted(name for name in sys.modules if name.startswith(package + '.'))
assert loaded == [], loaded
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


@pytest.mark.parametrize("leaf", RETIRED_MODULES)
def test_old_robustness_module_is_absent(leaf: str) -> None:
    assert importlib.util.find_spec(f"easyicu.research_agent.{leaf}") is None


@pytest.mark.parametrize("leaf", ROBUSTNESS_MODULES)
def test_robustness_runtime_does_not_reverse_import_pipeline(leaf: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.robustness.{leaf}")
    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(
                importlib.util.resolve_name(
                    "." * node.level + node.module,
                    module.__package__,
                )
                if node.level
                else node.module
            )
    forbidden = {
        "easyicu.research_agent.pipeline",
        "easyicu.research_agent.execution.phase",
    }
    assert imported.isdisjoint(forbidden)


def test_pipeline_consumers_use_canonical_robustness_modules() -> None:
    pipeline = importlib.import_module("easyicu.research_agent.pipeline")
    execute = importlib.import_module("easyicu.research_agent.execution.phase")
    panel = importlib.import_module("easyicu.research_agent.robustness.panel")
    estimators = importlib.import_module("easyicu.research_agent.robustness.estimators")
    primary_effect = importlib.import_module(
        "easyicu.research_agent.robustness.primary_effect"
    )
    assert pipeline.ensure_robustness_specs is panel.ensure_robustness_specs
    assert (
        pipeline._extract_primary_effect_row
        is primary_effect._extract_primary_effect_row
    )
    assert (
        execute.build_robustness_panel_from_records
        is panel.build_robustness_panel_from_records
    )
    assert (
        execute.fit_robustness_rows_from_records
        is estimators.fit_robustness_rows_from_records
    )
