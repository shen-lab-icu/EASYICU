"""Ownership contracts for the execute-phase module."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys


def test_execute_phase_has_one_canonical_home() -> None:
    module = importlib.import_module("easyicu.research_agent.execution.phase")
    assert module.__name__ == "easyicu.research_agent.execution.phase"
    assert Path(module.__file__).name == "phase.py"
    assert importlib.util.find_spec("easyicu.research_agent.pipeline_execute") is None


def test_execution_package_does_not_eagerly_import_phase() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.execution'
importlib.import_module(package)
assert package + '.phase' not in sys.modules
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_execute_phase_does_not_reverse_import_pipeline() -> None:
    module = importlib.import_module("easyicu.research_agent.execution.phase")
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
    assert "easyicu.research_agent.pipeline" not in imported


def test_execute_phase_relative_imports_resolve_from_canonical_package() -> None:
    module = importlib.import_module("easyicu.research_agent.execution.phase")
    tree = ast.parse(inspect.getsource(module))
    relative_modules = {
        importlib.util.resolve_name(
            "." * node.level + (node.module or ""),
            module.__package__,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
    }

    assert relative_modules
    assert all(importlib.util.find_spec(name) is not None for name in relative_modules)


def test_pipeline_delegates_to_canonical_execute_phase() -> None:
    pipeline = importlib.import_module("easyicu.research_agent.pipeline")
    phase = importlib.import_module("easyicu.research_agent.execution.phase")
    source = inspect.getsource(pipeline.ResearchAgentPipeline._run_execute_phase)
    assert "from .execution.phase import run_execute_phase" in source
    assert callable(phase.run_execute_phase)
