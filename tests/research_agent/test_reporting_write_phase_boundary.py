"""Boundary checks for the extracted write-phase module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_reporting_write_phase_entrypoint_is_importable() -> None:
    from easyicu.research_agent.reporting.write_phase import run_write_phase

    assert callable(run_write_phase)


def test_reporting_write_phase_does_not_import_pipeline_at_module_top() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "reporting"
        / "write_phase.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module in {"pipeline", "easyicu.research_agent.pipeline"}
        for node in top_imports
    )


def test_write_phase_keeps_stages_bounded() -> None:
    """The public phase remains orchestration, not another monolithic owner."""

    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "reporting"
        / "write_phase.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions["run_write_phase"].end_lineno - functions["run_write_phase"].lineno < 300
    for name in (
        "_activate_publication_inputs",
        "_draft_manuscript",
        "_bind_and_review_manuscript",
        "_publish_and_audit_manuscript",
        "_write_reproducibility_artifacts",
    ):
        function = functions[name]
        assert function.end_lineno - function.lineno < 500, name
