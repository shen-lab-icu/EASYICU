"""Architecture contract for the Planner-owned scientific signature kernel."""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.research_agent import pipeline_execute
from easyicu.research_agent.authority import plan_scope


def test_pipeline_execute_reexports_plan_scope_objects_with_identity() -> None:
    assert plan_scope.__all__
    for name in plan_scope.__all__:
        assert getattr(pipeline_execute, name) is getattr(plan_scope, name)


def test_plan_scope_has_no_orchestration_or_mutation_dependency() -> None:
    tree = ast.parse(inspect.getsource(plan_scope))
    imported_leaves = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert imported_leaves.isdisjoint(
        {"pipeline", "pipeline_execute", "gates", "execution", "evidence"}
    )
    assert identifiers.isdisjoint(
        {
            "open",
            "write_text",
            "write_bytes",
            "register",
            "promote",
            "consume",
            "repair",
            "complete",
        }
    )


@pytest.mark.parametrize("canonical_first", [True, False])
def test_plan_scope_identity_survives_import_order(canonical_first: bool) -> None:
    canonical = "easyicu.research_agent.authority.plan_scope"
    legacy = "easyicu.research_agent.pipeline_execute"
    first, second = (canonical, legacy) if canonical_first else (legacy, canonical)
    script = f"""
import importlib
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
canonical = importlib.import_module({canonical!r})
legacy = importlib.import_module({legacy!r})
for name in canonical.__all__:
    assert getattr(legacy, name) is getattr(canonical, name), name
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
