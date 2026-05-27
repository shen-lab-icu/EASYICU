"""Boundary checks for the extracted package/finalise module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_pipeline_package_entrypoints_are_importable() -> None:
    from easyicu.research_agent.pipeline_package import (
        finalise_aborted,
        finalise_success,
    )

    assert callable(finalise_success)
    assert callable(finalise_aborted)


def test_pipeline_package_does_not_import_pipeline_at_module_top() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "pipeline_package.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module in {"pipeline", "easyicu.research_agent.pipeline"}
        for node in top_imports
    )
