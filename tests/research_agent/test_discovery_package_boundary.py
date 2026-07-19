"""Compatibility and dependency contracts for the discovery package."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

DISCOVERY_MODULE_ALIASES = tuple(
    (
        f"easyicu.research_agent.{leaf}",
        f"easyicu.research_agent.discovery.{leaf}",
    )
    for leaf in (
        "idea_mining_schema",
        "idea_mining_pubmed",
        "idea_scope",
        "idea_registry",
        "hypothesis_generator",
        "idea_mining_data_first",
        "idea_mining_feasibility_tier",
        "concept_proposal",
        "idea_mining",
        "idea_mining_priorart",
        "idea_mining_funnel",
        "idea_mining_extended_feasibility",
        "idea_mining_eval",
        "discovery_handoff",
        "discovery_package",
        "discovery_story_figure",
    )
)


@pytest.mark.parametrize("legacy,canonical", DISCOVERY_MODULE_ALIASES)
def test_discovery_legacy_path_is_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__
    assert "/discovery/" in Path(new_module.__file__).as_posix()


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "root_first"))
def test_discovery_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {DISCOVERY_MODULE_ALIASES!r}
if {order!r} == 'root_first':
    root = importlib.import_module('easyicu.research_agent')
    getattr(root, 'IdeaCandidateRegistry')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    assert importlib.import_module(names[0]) is importlib.import_module(names[1])
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_discovery_package_is_lazy_and_canonical_modules_do_not_import_pipeline() -> (
    None
):
    package = importlib.import_module("easyicu.research_agent.discovery")
    package_tree = ast.parse(inspect.getsource(package))
    assert not [node for node in ast.walk(package_tree) if isinstance(node, ast.Import)]
    assert not [
        node for node in ast.walk(package_tree) if isinstance(node, ast.ImportFrom)
    ]

    for _legacy, canonical in DISCOVERY_MODULE_ALIASES:
        tree = ast.parse(inspect.getsource(importlib.import_module(canonical)))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(name.startswith("pipeline") for name in imported_modules)


def test_default_idea_quality_fixture_path_is_independent_of_module_depth() -> None:
    module = importlib.import_module(
        "easyicu.research_agent.discovery.idea_mining_eval"
    )
    package_root = (
        Path(importlib.import_module("easyicu").__file__).resolve().parents[2]
    )
    assert module.default_idea_quality_eval_path() == (
        package_root / "benchmark" / "idea_mining_quality_eval_set.json"
    )
