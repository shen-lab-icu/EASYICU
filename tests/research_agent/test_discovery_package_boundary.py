"""Canonical dependency contracts for the discovery package."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest


DISCOVERY_MODULES = (
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


@pytest.mark.parametrize("leaf", DISCOVERY_MODULES)
def test_discovery_module_has_one_canonical_home(leaf: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.discovery.{leaf}")
    assert module.__name__.endswith(f"discovery.{leaf}")
    assert "/discovery/" in Path(module.__file__).as_posix()


def test_discovery_package_is_lazy_and_modules_do_not_import_pipeline() -> None:
    package = importlib.import_module("easyicu.research_agent.discovery")
    package_tree = ast.parse(inspect.getsource(package))
    assert not [node for node in ast.walk(package_tree) if isinstance(node, ast.Import)]
    assert not [
        node for node in ast.walk(package_tree) if isinstance(node, ast.ImportFrom)
    ]
    for leaf in DISCOVERY_MODULES:
        tree = ast.parse(
            inspect.getsource(
                importlib.import_module(f"easyicu.research_agent.discovery.{leaf}")
            )
        )
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(name.startswith("pipeline") for name in imported)


def test_default_idea_quality_fixture_path_is_independent_of_module_depth() -> None:
    module = importlib.import_module(
        "easyicu.research_agent.discovery.idea_mining_eval"
    )
    package_root = Path(importlib.import_module("easyicu").__file__).resolve().parents[2]
    assert module.default_idea_quality_eval_path() == (
        package_root / "benchmark" / "idea_mining_quality_eval_set.json"
    )
