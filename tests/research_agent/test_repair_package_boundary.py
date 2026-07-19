"""Canonical dependency contracts for repair-control modules."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPAIR_CONTROL_MODULES = (
    "authority.provider_budget",
    "gates.preflight",
    "repairs.source",
    "repairs.helpers",
    "repairs.reasons",
    "repairs.coordination",
    "repairs.patch",
    "repairs.summary",
)

REPAIR_CONSUMERS = (
    "easyicu.research_agent.agents.core",
    "easyicu.research_agent.pipeline",
    "easyicu.research_agent.pipeline_execute",
    "easyicu.research_agent.pipeline_resume",
    "easyicu.research_agent.gates.visual",
)


@pytest.mark.parametrize("target", REPAIR_CONTROL_MODULES)
def test_repair_module_has_one_canonical_home(target: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.{target}")
    assert module.__name__ == f"easyicu.research_agent.{target}"


def test_repairs_package_does_not_eagerly_import_implementation_modules() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.repairs'
importlib.import_module(package)
loaded = sorted(name for name in sys.modules if name.startswith(package + '.'))
assert loaded == [], loaded
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


@pytest.mark.parametrize("module_name", REPAIR_CONSUMERS)
def test_repair_consumers_import_canonical_packages(module_name: str) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(
                importlib.util.resolve_name(
                    "." * node.level + node.module, module.__package__
                )
                if node.level
                else node.module
            )
    retired = {
        "easyicu.research_agent.code_patch",
        "easyicu.research_agent.code_preflight",
        "easyicu.research_agent.code_repair",
        "easyicu.research_agent.code_repair_helpers",
        "easyicu.research_agent.provider_budget",
        "easyicu.research_agent.repair_coordination",
        "easyicu.research_agent.repair_reasons",
        "easyicu.research_agent.summary_repair",
    }
    assert imported.isdisjoint(retired)
