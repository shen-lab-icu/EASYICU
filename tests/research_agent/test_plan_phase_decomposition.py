"""Architecture guard for the characterized plan-phase split."""

from __future__ import annotations

import ast
from pathlib import Path


def test_plan_phase_stage_functions_stay_bounded() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "pipeline.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pipeline = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResearchAgentPipeline"
    )
    methods = {
        node.name: node
        for node in pipeline.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert methods["_run_plan_phase"].end_lineno - methods["_run_plan_phase"].lineno < 900
    assert (
        methods["_generate_or_resume_plan"].end_lineno
        - methods["_generate_or_resume_plan"].lineno
        < 450
    )
    assert (
        methods["_validate_and_persist_plan"].end_lineno
        - methods["_validate_and_persist_plan"].lineno
        < 450
    )


def test_plan_generation_handoff_is_immutable() -> None:
    from easyicu.research_agent.pipeline import _PlanGenerationResult

    assert _PlanGenerationResult.__dataclass_params__.frozen is True
