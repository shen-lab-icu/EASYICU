"""Boundary checks for the extracted write-phase module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_pipeline_write_phase_entrypoint_is_importable() -> None:
    from easyicu.research_agent.pipeline_write import run_write_phase

    assert callable(run_write_phase)


def test_pipeline_write_does_not_import_pipeline_at_module_top() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "pipeline_write.py"
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
