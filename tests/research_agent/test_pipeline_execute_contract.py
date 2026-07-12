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


def test_critic_messages_exclude_info_but_keep_warnings_and_errors():
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _actionable_validator_messages,
    )

    messages = _actionable_validator_messages(
        [
            ValidationFinding(
                validator="audit",
                severity="info",
                message="Informational provenance note.",
            ),
            ValidationFinding(
                validator="audit",
                severity="warning",
                message="Review this warning.",
            ),
            ValidationFinding(
                validator="audit",
                severity="error",
                message="Repair this error.",
            ),
        ]
    )

    assert messages == ["Review this warning.", "Repair this error."]


def test_required_model_contract_error_fail_closes_outer_step_and_run():
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _step_status_from_contract_findings,
    )
    from easyicu.research_agent.pipeline_report import execution_gate_status
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    contract_findings = [
        ValidationFinding(
            validator="primary_model_contract",
            severity="error",
            message="A planner-required secondary model was not fitted.",
            detail={"issue": "required_model_not_fitted"},
        )
    ]
    status = _step_status_from_contract_findings(
        contract_findings=contract_findings,
        figure_source_findings=[],
        stat_findings=[],
    )
    plan = AnalysisPlan(
        research_question="Test a planner-owned model obligation.",
        steps=[
            AnalysisStep(
                step_id="01_models",
                intent="Fit the planned models.",
            )
        ],
    )

    assert status == "contract_failed"
    gate = execution_gate_status(
        plan=plan,
        per_step_records=[{"step_id": "01_models", "status": status}],
    )
    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [
        {"step_id": "01_models", "status": "contract_failed"}
    ]


@pytest.mark.parametrize(
    ("step_id", "intent"),
    [
        (
            "04_publication_figure_interpretation",
            "Interpret the downstream publication figure for the manuscript.",
        ),
        (
            "04_primary_model",
            "Estimate the association used in a publication-ready figure.",
        ),
    ],
)
def test_publication_figure_gate_ignores_name_only_mentions(step_id, intent):
    from easyicu.research_agent.pipeline_execute import (
        _step_requires_publication_figure_exports,
    )
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id=step_id,
        intent=intent,
        method="mixed_effects_regression",
        expected_outputs=["table:association_estimates"],
    )

    assert _step_requires_publication_figure_exports(step) is False


@pytest.mark.parametrize(
    ("method", "expected_outputs"),
    [
        ("publication_figure_generation", ["log:rendering_process"]),
        ("visualization", ["log:rendering_process"]),
        ("mixed_effects_regression", ["figure:association_forest_plot"]),
    ],
)
def test_publication_figure_gate_accepts_structural_figure_contracts(
    method, expected_outputs
):
    from easyicu.research_agent.pipeline_execute import (
        _step_requires_publication_figure_exports,
    )
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="04_results_publication_figure",
        intent="Render the requested publication figure.",
        method=method,
        expected_outputs=expected_outputs,
    )

    assert _step_requires_publication_figure_exports(step) is True


def test_execute_phase_mandatory_publication_gate_uses_structural_predicate():
    from easyicu.research_agent.pipeline_execute import run_execute_phase

    source = inspect.getsource(run_execute_phase)
    gate_start = source.index("publication_step =")
    gate_end = source.index("figure_role =", gate_start)
    gate_source = source[gate_start:gate_end]

    assert "_step_requires_publication_figure_exports" in gate_source
    assert "step.step_id" not in gate_source
    assert "step.intent" not in gate_source


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
        name
        for name, p in params.items()
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
        name for name, p in params.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
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


def test_execute_phase_preserves_repair_provenance_across_concept_and_runtime():
    """Every LLM mutation must outrank pure resume/runner provenance labels."""
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)

    # Concept, visual, contract, and runtime repairs each mark the same lineage
    # flag immediately after a successful coder.repair call.
    assert source.count("llm_repair_used = True") == 4
    assert "concept_repair_used=concept_repair_used" in source
    assert "llm_repair_used=llm_repair_used" in source
    # A repaired resumed script must receive a fresh analyzer interpretation;
    # only genuinely unchanged reuse and deterministic fallback skip it.
    assert 'final_generation_mode in {"resumed_code_reuse", "fallback"}' in source


def test_execute_phase_routes_figure_contracts_through_early_repair_loop():
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    early_gate = source.index("early_contract_errors = [")
    before_early_gate = source[:early_gate]

    assert "figure_contract_validator.audit(" in before_early_gate
    assert "figure_source_validator.audit(" in before_early_gate


def test_plan_and_execute_result_dataclass_shapes_match_contracts_module():
    """Pin the two dataclasses that flow through run_execute_phase.

    The pipeline phases exchange ``_PlanPhaseResult``,
    ``_ExecutePhaseResult`` and ``_WritePhaseResult``. They are defined in
    ``contracts.py`` and re-exported by ``pipeline.py`` / ``pipeline_state.py``
    for compatibility. If any shape drifts, a phase silently misreads its
    input or produces a malformed handoff to the next phase.
    """
    from easyicu.research_agent.contracts import (
        _PlanPhaseResult,
        _ExecutePhaseResult,
        _WritePhaseResult,
    )
    from easyicu.research_agent.pipeline import (
        _PlanPhaseResult as PipelinePlanPhaseResult,
        _ExecutePhaseResult as PipelineExecutePhaseResult,
        _WritePhaseResult as PipelineWritePhaseResult,
    )
    from easyicu.research_agent.pipeline_state import (
        PlanPhaseState,
        ExecutePhaseState,
        WritePhaseState,
    )

    assert PipelinePlanPhaseResult is _PlanPhaseResult
    assert PipelineExecutePhaseResult is _ExecutePhaseResult
    assert PipelineWritePhaseResult is _WritePhaseResult
    assert PlanPhaseState is _PlanPhaseResult
    assert ExecutePhaseState is _ExecutePhaseResult
    assert WritePhaseState is _WritePhaseResult

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
        f"_PlanPhaseResult is missing fields {missing} consumed by run_execute_phase."
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

    write_fields = {f.name for f in fields(_WritePhaseResult)}
    required_write_fields = {
        "literature",
        "bound_path",
        "manuscript_packet",
        "manuscript_critique",
    }
    missing_write = required_write_fields - write_fields
    assert not missing_write, (
        f"_WritePhaseResult is missing fields {missing_write} produced "
        "by the write phase / consumed by the package phase."
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


def test_visual_qa_demotes_only_cosmetic_layout_errors(ra):
    from easyicu.research_agent.pipeline_execute import (
        _demote_cosmetic_visual_findings,
    )
    from easyicu.research_agent.schema import ValidationFinding

    cosmetic = ValidationFinding(
        validator="visual_qa",
        severity="error",
        message=(
            "SVG figure 'x.svg' has overlapping text elements; "
            "multi-panel labels, annotations or axis text need more spacing."
        ),
    )
    hard = ValidationFinding(
        validator="visual_qa",
        severity="error",
        message="Could not open figure 'x.png': truncated image file",
    )
    vlm = ValidationFinding(
        validator="vlm_visual_qa",
        severity="error",
        message="Panel B axis values do not match source data.",
    )

    demoted, blocking = _demote_cosmetic_visual_findings([cosmetic, hard, vlm])

    assert demoted[0].severity == "warning"
    assert demoted[1].severity == "error"
    assert demoted[2].severity == "error"
    assert [f.message for f in blocking] == [hard.message, vlm.message]


def test_scope_findings_step_global_warning_does_not_taint_records():
    """A step-global warning (no evidence_ids) is an analysis-design advisory
    and must NOT taint the citability of the step's output records — otherwise
    one 'immortal-time-bias risk' note makes the primary result table
    uncitable and the manuscript unwinnable."""
    from easyicu.research_agent.pipeline_execute import scope_findings_to_records
    from easyicu.research_agent.schema import ValidationFinding

    global_warning = ValidationFinding(
        validator="clinical_constraint_validator",
        severity="warning",
        message="Treatment-effect analysis without an explicit time-zero.",
    )
    scoped = scope_findings_to_records(
        ["table_one", "adjusted_association"], [global_warning]
    )
    assert scoped["table_one"] == (None, [])
    assert scoped["adjusted_association"] == (None, [])


def test_scope_findings_targeted_finding_taints_only_named_record():
    """A finding that names specific records taints ONLY those records."""
    from easyicu.research_agent.pipeline_execute import scope_findings_to_records
    from easyicu.research_agent.schema import ValidationFinding

    global_warning = ValidationFinding(
        validator="clinical_constraint_validator",
        severity="warning",
        message="Design advisory.",
    )
    targeted = ValidationFinding(
        validator="critic_agent",
        severity="warning",
        message="Critique of the interpretation log.",
        evidence_ids=["log_critique_report_x"],
    )
    scoped = scope_findings_to_records(
        ["table_one", "log_critique_report_x"], [global_warning, targeted]
    )
    assert scoped["table_one"] == (None, [])
    severity, messages = scoped["log_critique_report_x"]
    assert severity == "warning"
    assert messages == ["Critique of the interpretation log."]


def test_scope_findings_step_global_error_stays_fail_closed():
    """A step-global ERROR keeps the blanket taint (fail-closed): a step-level
    error means the step's outputs are not to be trusted."""
    from easyicu.research_agent.pipeline_execute import scope_findings_to_records
    from easyicu.research_agent.schema import ValidationFinding

    global_error = ValidationFinding(
        validator="execution",
        severity="error",
        message="Step analysis crashed before producing a result.",
    )
    scoped = scope_findings_to_records(
        ["table_one", "adjusted_association"], [global_error]
    )
    for eid in ("table_one", "adjusted_association"):
        severity, messages = scoped[eid]
        assert severity == "error"
        assert messages == ["Step analysis crashed before producing a result."]
