"""Compatibility and dependency contracts for repair-control modules."""

from __future__ import annotations

import importlib
import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPAIR_CONTROL_MODULE_ALIASES = (
    (
        "easyicu.research_agent.provider_budget",
        "easyicu.research_agent.authority.provider_budget",
    ),
    (
        "easyicu.research_agent.code_preflight",
        "easyicu.research_agent.gates.preflight",
    ),
    (
        "easyicu.research_agent.code_repair",
        "easyicu.research_agent.repairs.source",
    ),
    (
        "easyicu.research_agent.code_repair_helpers",
        "easyicu.research_agent.repairs.helpers",
    ),
    (
        "easyicu.research_agent.repair_reasons",
        "easyicu.research_agent.repairs.reasons",
    ),
    (
        "easyicu.research_agent.repair_coordination",
        "easyicu.research_agent.repairs.coordination",
    ),
    (
        "easyicu.research_agent.code_patch",
        "easyicu.research_agent.repairs.patch",
    ),
    (
        "easyicu.research_agent.summary_repair",
        "easyicu.research_agent.repairs.summary",
    ),
)

CANONICAL_REPAIR_CONSUMERS = (
    "easyicu.research_agent.agents",
    "easyicu.research_agent.pipeline",
    "easyicu.research_agent.pipeline_execute",
    "easyicu.research_agent.pipeline_resume",
    "easyicu.research_agent.gates.visual",
    "easyicu.research_agent.repairs.source",
    "easyicu.research_agent.repairs.coordination",
    "easyicu.research_agent.repairs.patch",
    "easyicu.research_agent.repairs.summary",
)


@pytest.mark.parametrize("legacy,canonical", REPAIR_CONTROL_MODULE_ALIASES)
def test_repair_control_legacy_path_is_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "pipeline_first"))
def test_repair_control_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {REPAIR_CONTROL_MODULE_ALIASES!r}
if {order!r} == 'pipeline_first':
    importlib.import_module('easyicu.research_agent.pipeline_execute')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    assert importlib.import_module(names[0]) is importlib.import_module(names[1])
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_provider_budget_legacy_monkeypatch_owner_is_canonical() -> None:
    legacy = importlib.import_module("easyicu.research_agent.provider_budget")
    canonical = importlib.import_module(
        "easyicu.research_agent.authority.provider_budget"
    )
    assert legacy.os is canonical.os


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


@pytest.mark.parametrize("module_name", CANONICAL_REPAIR_CONSUMERS)
def test_canonical_repair_consumers_never_route_through_legacy_facades(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                imported_modules.add(
                    importlib.util.resolve_name(
                        "." * node.level + node.module,
                        module.__package__,
                    )
                )
            else:
                imported_modules.add(node.module)

    legacy_modules = {legacy for legacy, _canonical in REPAIR_CONTROL_MODULE_ALIASES}
    assert imported_modules.isdisjoint(legacy_modules)
