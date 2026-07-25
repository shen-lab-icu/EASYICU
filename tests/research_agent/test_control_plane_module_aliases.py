"""Canonical ownership contracts for the execution control plane."""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest


CANONICAL_CONTROL_MODULES = (
    "easyicu.research_agent.authority.registration",
    "easyicu.research_agent.gates.visual",
    "easyicu.research_agent.gates.contract",
    "easyicu.research_agent.gates.concept",
    "easyicu.research_agent.execution.concept_audit",
    "easyicu.research_agent.execution.figure_preparation",
    "easyicu.research_agent.execution.publication_figure",
)


@pytest.mark.parametrize("module_name", CANONICAL_CONTROL_MODULES)
def test_control_plane_module_has_one_canonical_home(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name
    assert "/research_agent/" in str(module.__file__)


@pytest.mark.parametrize("module_name", CANONICAL_CONTROL_MODULES)
def test_control_component_does_not_import_execute_orchestrator(
    module_name: str,
) -> None:
    tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    imported_leaves: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_leaves.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            imported_leaves.update(
                alias.name.rsplit(".", 1)[-1] for alias in node.names
            )
    assert "pipeline_execute" not in imported_leaves


def test_execute_orchestrator_imports_canonical_control_modules() -> None:
    module = importlib.import_module("easyicu.research_agent.execution.phase")
    tree = ast.parse(inspect.getsource(module))
    imported = {
        (
            importlib.util.resolve_name(
                "." * node.level + node.module,
                module.__package__,
            )
            if node.level
            else node.module
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert {
        "easyicu.research_agent.authority.registration",
        "easyicu.research_agent.gates.visual",
        "easyicu.research_agent.gates.contract",
        "easyicu.research_agent.gates.concept",
        "easyicu.research_agent.execution.concept_audit",
        "easyicu.research_agent.execution.figure_preparation",
        "easyicu.research_agent.execution.publication_figure",
    }.issubset(imported)
