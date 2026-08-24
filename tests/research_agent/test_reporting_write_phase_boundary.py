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
        "_activate_publication_figure",
        "_activate_publication_inputs",
        "_draft_manuscript",
        "_bind_and_review_manuscript",
        "_publish_and_audit_manuscript",
        "_write_reproducibility_artifacts",
    ):
        function = functions[name]
        assert function.end_lineno - function.lineno < 500, name


def test_publication_figure_activation_precedes_analysis_pause() -> None:
    """A Writer pause must not skip the deterministic article-display suffix."""

    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "reporting"
        / "write_phase.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    run_write_phase = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_write_phase"
    )
    figure_calls = [
        node
        for node in ast.walk(run_write_phase)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_activate_publication_figure"
    ]
    completed_pause = next(
        node
        for node in run_write_phase.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "stop_after_analysis"
    )
    assert len(figure_calls) == 1
    assert figure_calls[0].lineno < completed_pause.lineno
