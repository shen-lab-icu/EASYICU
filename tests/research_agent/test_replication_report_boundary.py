"""Canonical ownership contract for cross-database report rendering."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys


def test_replication_report_has_one_canonical_home() -> None:
    module = importlib.import_module("easyicu.research_agent.replication.report")
    assert module.__name__ == "easyicu.research_agent.replication.report"
    assert "/replication/" in Path(module.__file__).as_posix()


def test_replication_package_is_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.replication'
importlib.import_module(package)
loaded = sorted(name for name in sys.modules if name.startswith(package + '.'))
assert loaded == [], loaded
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_replication_public_symbols_resolve_to_leaf_objects() -> None:
    package = importlib.import_module("easyicu.research_agent.replication")
    for name in package.__all__:
        value = getattr(package, name)
        leaf_name = package._SYMBOL_MODULE[name]
        leaf = importlib.import_module(
            f"easyicu.research_agent.replication{leaf_name}"
        )
        assert value is getattr(leaf, name)


def test_replication_report_does_not_reverse_import_pipeline() -> None:
    module = importlib.import_module("easyicu.research_agent.replication.report")
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imports.isdisjoint({"pipeline", "pipeline_execute"})


def test_pipeline_replication_helpers_use_canonical_objects() -> None:
    pipeline = importlib.import_module("easyicu.research_agent.pipeline")
    report = importlib.import_module("easyicu.research_agent.replication.report")
    for name in report.__all__:
        assert getattr(pipeline, name) is getattr(report, name)
