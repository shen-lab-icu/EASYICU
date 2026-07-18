"""Compatibility contracts for responsibility-subpackage migrations."""

from __future__ import annotations

import importlib
import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_ALIASES = (
    (
        "easyicu.research_agent.evidence_registration",
        "easyicu.research_agent.authority.registration",
    ),
    (
        "easyicu.research_agent.gate_evaluator",
        "easyicu.research_agent.gates.visual",
    ),
    (
        "easyicu.research_agent.contract_gate",
        "easyicu.research_agent.gates.contract",
    ),
    (
        "easyicu.research_agent.concept_gate",
        "easyicu.research_agent.gates.concept",
    ),
    (
        "easyicu.research_agent.concept_audit_execution",
        "easyicu.research_agent.execution.concept_audit",
    ),
    (
        "easyicu.research_agent.figure_contract_preparation",
        "easyicu.research_agent.execution.figure_preparation",
    ),
    (
        "easyicu.research_agent.publication_figure_execution",
        "easyicu.research_agent.execution.publication_figure",
    ),
)

LEGACY_LEAF_MODULES = {
    legacy.rsplit(".", 1)[-1] for legacy, _canonical in MODULE_ALIASES
}


@pytest.mark.parametrize("legacy,canonical", MODULE_ALIASES)
def test_legacy_path_is_the_canonical_module_object(
    legacy: str, canonical: str
) -> None:
    assert importlib.import_module(legacy) is importlib.import_module(canonical)


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "pipeline_first"))
def test_module_aliases_survive_clean_import_order(order: str) -> None:
    pairs = repr(MODULE_ALIASES)
    script = f"""
import importlib
pairs = {pairs}
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


@pytest.mark.parametrize("legacy,canonical", MODULE_ALIASES)
def test_canonical_module_never_imports_legacy_facade_or_pipeline_execute(
    legacy: str,
    canonical: str,
) -> None:
    del legacy
    module = importlib.import_module(canonical)
    tree = ast.parse(inspect.getsource(module))
    imported_leaves: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_leaves.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            imported_leaves.update(
                alias.name.rsplit(".", 1)[-1] for alias in node.names
            )
    assert "pipeline_execute" not in imported_leaves
    assert imported_leaves.isdisjoint(LEGACY_LEAF_MODULES)


def test_pipeline_execute_uses_only_canonical_responsibility_paths() -> None:
    module = importlib.import_module("easyicu.research_agent.pipeline_execute")
    tree = ast.parse(inspect.getsource(module))
    imported_leaves = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported_leaves.isdisjoint(LEGACY_LEAF_MODULES)
