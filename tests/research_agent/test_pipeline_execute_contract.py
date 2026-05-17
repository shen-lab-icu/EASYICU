"""Contract-pinning tests for ``easyicu.research_agent.pipeline_execute``.

Background
----------
``pipeline_execute.py`` (~1,700 LOC) houses the probe → per-step
analysis loop with optional replanning. It is a free-function entry
point (``run_execute_phase``) deliberately split out of
``ResearchAgentPipeline._run_execute_phase`` so a future LangGraph-style
runner can wrap it directly.

Why this file holds *contract* tests and not behaviour tests
------------------------------------------------------------
``run_execute_phase`` is an integration-only entry: it immediately
constructs ``CoderAgent``, ``AnalyzerAgent``, ``RuntimeSupervisor`` and
calls ``pipeline._build_runner(...)``. Exercising it meaningfully
requires the same fixtures the end-to-end ``ResearchAgentPipeline.run``
tests already build. Duplicating those fixtures here would just give us
a slower copy of the same coverage.

What this file *does* protect against is the silent breakage class that
the e2e tests detect 9 minutes late: someone renames the function,
changes its keyword arguments, or breaks the ``(pipeline, plan_result)
→ _ExecutePhaseResult`` shape. We pin those at the import level so the
break shows up in the next ``pytest --collect-only``.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest


def test_module_is_importable():
    import easyicu.research_agent.pipeline_execute as pe  # noqa: F401


def test_run_execute_phase_is_exported():
    from easyicu.research_agent.pipeline_execute import run_execute_phase
    assert callable(run_execute_phase)


def test_run_execute_phase_signature_is_stable():
    """Lock the keyword-argument contract pipeline.py relies on.

    If a parameter is renamed or removed here, callers in pipeline.py
    will fail at import time elsewhere. Catching it as a one-line
    signature diff is far cheaper than the e2e failure.
    """
    from easyicu.research_agent.pipeline_execute import run_execute_phase

    sig = inspect.signature(run_execute_phase)
    params = sig.parameters

    # First positional is the pipeline collaborator; the rest are keyword-only.
    positional = [
        name for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["pipeline"], (
        "run_execute_phase must take exactly one positional collaborator "
        f"(the pipeline); got {positional}"
    )

    required_keywords = {
        "plan_result",
        "cohort_path",
        "run_dir",
        "run_id",
        "skill_obj",
        "notes",
        "emit_progress",
    }
    actual_keywords = {
        name for name, p in params.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    missing = required_keywords - actual_keywords
    assert not missing, (
        f"run_execute_phase is missing keyword-only params {missing}; "
        "downstream pipeline.py keyword call will break."
    )


def test_run_execute_phase_does_not_mutate_pipeline_state():
    """Lock the read-only-collaborator invariant.

    Module docstring states: 'pipeline instance is passed in only as a
    *read-only collaborator* … audit on 2026-05-15 confirmed zero
    ``self.* = ...`` writes inside the original method body.' If a
    refactor reintroduces a write, future graph-runner authors will
    have a confusing aliasing bug. We re-run the audit in CI.
    """
    import ast
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    tree = ast.parse(source)

    pipeline_writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "pipeline"
                ):
                    pipeline_writes.append(target.attr)
        elif isinstance(node, ast.AugAssign):
            if (
                isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "pipeline"
            ):
                pipeline_writes.append(node.target.attr)

    assert pipeline_writes == [], (
        "run_execute_phase must not mutate the pipeline collaborator; "
        f"found writes to: {pipeline_writes}. See module docstring."
    )


def test_plan_and_execute_result_dataclass_shapes_match_pipeline_module():
    """Pin the two dataclasses that flow through run_execute_phase.

    The function takes a ``_PlanPhaseResult`` and returns an
    ``_ExecutePhaseResult``; both are defined in ``pipeline.py``. If
    either shape drifts, the execute phase silently misreads its input
    or produces a malformed handoff to the write phase.
    """
    from easyicu.research_agent.pipeline import (
        _PlanPhaseResult,
        _ExecutePhaseResult,
    )

    plan_fields = {f.name for f in fields(_PlanPhaseResult)}
    # Names the execute phase actually reads off plan_result, verified
    # against pipeline_execute.run_execute_phase body 2026-05-17.
    required_plan_fields = {
        "context",
        "agent_context",
        "evidence",
        "findings",
        "plan",
        "plan_path",
        "role_resolver",
        "llm_signature",
        "prompt_version",
        "prompt_files",
        "resume_state",
    }
    missing = required_plan_fields - plan_fields
    assert not missing, (
        f"_PlanPhaseResult is missing fields {missing} consumed by "
        "run_execute_phase."
    )

    exec_fields = {f.name for f in fields(_ExecutePhaseResult)}
    required_exec_fields = {
        "plan",
        "per_step_records",
        "probe_summary",
        "runtime_state",
        "flush_partial_manifest",
    }
    missing_exec = required_exec_fields - exec_fields
    assert not missing_exec, (
        f"_ExecutePhaseResult is missing fields {missing_exec} produced "
        "by run_execute_phase / consumed by the write phase."
    )


def test_required_collaborators_are_importable():
    """Smoke-import each collaborator name pipeline_execute pulls in.

    A typo in one of the agent / validator / repair imports would only
    surface when the execute phase actually fires, which in the e2e
    suite is many minutes in. We import them upfront here.
    """
    from easyicu.research_agent.pipeline_execute import (  # noqa: F401
        AnalyzerAgent,
        ClinicalSemanticsAgent,
        CoderAgent,
        CriticAgent,
        DataExtractionAgent,
        ReplannerAgent,
        RuntimeSupervisor,
        StatisticalAnalysisAgent,
        VisualizationAgent,
        ClinicalConstraintValidator,
        ConceptUsageAuditor,
        LLMConceptAuditor,
        StatisticalGuard,
        StatisticalValidator,
        _deterministic_runner_repair,
        _deterministic_summary_repair,
        MockLLMClient,
    )
