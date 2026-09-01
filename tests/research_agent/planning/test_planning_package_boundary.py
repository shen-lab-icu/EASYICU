"""Dependency contracts for the canonical scientific-planning package."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys

import pytest


PLANNING_MODULES = (
    "study_design",
    "study_design_playbook",
    "capability_registry",
    "analysis_method_suite",
    "figure_strategy",
    "analysis_types",
)


@pytest.mark.parametrize("leaf", PLANNING_MODULES)
def test_planning_module_has_one_canonical_home(leaf: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.planning.{leaf}")
    assert module.__name__.endswith(f"planning.{leaf}")
    assert "/planning/" in Path(module.__file__).as_posix()


def test_planning_package_is_lazy() -> None:
    package_path = (
        Path(__file__).resolve().parents[2]
        / "src/easyicu/research_agent/planning/__init__.py"
    )
    tree = ast.parse(package_path.read_text(encoding="utf-8"))
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def test_registry_regeneration_commands_use_canonical_modules() -> None:
    import os

    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    for module_name in (
        "easyicu.research_agent.planning.capability_registry",
        "easyicu.research_agent.planning.analysis_method_suite",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        assert result.stdout.startswith("# ")


def test_root_planning_api_resolves_to_canonical_objects() -> None:
    root = importlib.import_module("easyicu.research_agent")
    analysis_types = importlib.import_module(
        "easyicu.research_agent.planning.analysis_types"
    )
    assert root.infer_analysis_type is analysis_types.infer_analysis_type
