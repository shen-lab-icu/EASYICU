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
import json
from dataclasses import fields

import pytest


def test_consistent_local_figure_source_descriptor_is_canonicalized_for_consumers(
    tmp_path,
):
    from easyicu.research_agent.discovery_package import _string_list
    from easyicu.research_agent.figure_skill import (
        _contract_payload_source_references,
    )
    from easyicu.research_agent.pipeline_execute import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "result_source_data.csv").write_text("x,y\na,1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [
                    {
                        "file": "result_source_data.csv",
                        "filename": "result_source_data.csv",
                        "path": "result_source_data.csv",
                        "relative_path": "result_source_data.csv",
                        "kind": "table",
                        "evidence_ids": [],
                    }
                ],
                "panels": [],
            }
        ),
        encoding="utf-8",
    )

    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    _before, canonical_text, names = candidate
    assert names == ["result_source_data.csv"]
    _install_figure_contract_source_data_canonicalization(
        contract_path=contract_path,
        expected_before=_before,
        canonical_text=canonical_text,
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert payload["source_data"] == ["result_source_data.csv"]
    assert _contract_payload_source_references(payload) == [
        "result_source_data.csv"
    ]
    assert _string_list(payload["source_data"]) == ["result_source_data.csv"]


def test_figure_contract_canonicalization_does_not_follow_predictable_temp_symlink(
    tmp_path,
):
    from easyicu.research_agent.pipeline_execute import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [{"file": "source.csv", "path": "source.csv"}],
            }
        ),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.json"
    outside.write_text("do-not-touch", encoding="utf-8")
    predictable = out_dir / ".result.figure_contract.json.schema.tmp"
    try:
        predictable.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    before, after, _names = candidate
    _install_figure_contract_source_data_canonicalization(
        contract_path=contract_path,
        expected_before=before,
        canonical_text=after,
    )

    assert outside.read_text(encoding="utf-8") == "do-not-touch"
    assert json.loads(contract_path.read_text(encoding="utf-8"))["source_data"] == [
        "source.csv"
    ]


def test_figure_contract_canonicalization_rejects_changed_reviewed_contract(
    tmp_path,
):
    from easyicu.research_agent.pipeline_execute import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [{"file": "source.csv", "path": "source.csv"}],
            }
        ),
        encoding="utf-8",
    )
    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    before, after, _names = candidate
    contract_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after canonicalization review"):
        _install_figure_contract_source_data_canonicalization(
            contract_path=contract_path,
            expected_before=before,
            canonical_text=after,
        )


@pytest.mark.parametrize(
    "source_data",
    [
        [{"file": "source.csv", "path": "other.csv"}],
        [{"file": 7}],
        [{"file": "/tmp/source.csv"}],
        [{"file": "nested/source.csv"}],
        [{"evidence_id": "table_source"}],
        [["source.csv"]],
    ],
)
def test_figure_source_descriptor_canonicalization_fails_closed(
    tmp_path,
    source_data,
):
    from easyicu.research_agent.pipeline_execute import (
        _figure_contract_source_data_canonicalization_candidate,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    (out_dir / "other.csv").write_text("x\n2\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps({"figure_id": "result", "source_data": source_data}),
        encoding="utf-8",
    )

    assert (
        _figure_contract_source_data_canonicalization_candidate(
            contract_path=contract_path,
            out_dir=out_dir,
        )
        is None
    )


def test_module_is_importable():
    import easyicu.research_agent.pipeline_execute as pe  # noqa: F401


def test_run_execute_phase_is_exported():
    from easyicu.research_agent.pipeline_execute import run_execute_phase

    assert callable(run_execute_phase)


def test_critic_messages_keep_only_blocking_errors():
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

    assert messages == ["Repair this error."]


def test_code_repair_findings_keep_only_blocking_errors():
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _blocking_validator_findings,
    )

    findings = _blocking_validator_findings(
        [
            ValidationFinding(
                validator="audit",
                severity="warning",
                message="Keep as advisory evidence only.",
            ),
            ValidationFinding(
                validator="audit",
                severity="error",
                message="Repair this blocking error.",
                detail={"reason": "blocking_contract"},
            ),
        ]
    )

    assert [finding.message for finding in findings] == [
        "Repair this blocking error."
    ]
    assert findings[0].detail == {"reason": "blocking_contract"}


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"status": "ok", "step_summary": {}}, False),
        ({"status": "ok", "replan_requested": True}, True),
        (
            {
                "status": "ok",
                "step_summary": {"plan_revision_requested": True},
            },
            True,
        ),
        ({"status": "ok", "step_summary": {"replan_requested": "true"}}, False),
        ({"status": "contract_failed", "replan_requested": True}, False),
    ],
)
def test_success_replanning_requires_an_exact_agent_request(record, expected):
    from easyicu.research_agent.pipeline_execute import (
        _successful_step_requests_replan,
    )

    assert _successful_step_requests_replan(record) is expected


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


def test_every_deterministic_statistical_error_fails_outer_step():
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _step_status_from_contract_findings,
    )

    status = _step_status_from_contract_findings(
        contract_findings=[],
        figure_source_findings=[],
        stat_findings=[
            ValidationFinding(
                validator="statistical_sanity",
                severity="error",
                message="A deterministic statistical contract failed.",
                detail={"issue": "impossible_denominator"},
            )
        ],
    )

    assert status == "contract_failed"


@pytest.mark.parametrize("critique_status", ["needs_revision", "blocked"])
def test_negative_critic_review_fail_closes_outer_step(critique_status):
    from easyicu.research_agent.pipeline_execute import (
        _step_status_from_contract_findings,
    )

    assert (
        _step_status_from_contract_findings(
            contract_findings=[],
            figure_source_findings=[],
            stat_findings=[],
            critique_status=critique_status,
        )
        == "critic_failed"
    )


def test_locked_measurement_data_quality_classifier_is_structural():
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _locked_measurement_data_quality_issues,
    )

    findings = [
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="Locked data contain invalid pairs.",
            detail={"issue": "measurement_provenance_invalid_pairs"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="Locked data contain discordance.",
            detail={"issue": "measurement_provenance_count_flag_discordance"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="Generated code reported the wrong count.",
            detail={"issue": "measurement_provenance_host_count_mismatch"},
        ),
        ValidationFinding(
            validator="another_validator",
            severity="error",
            message="Same words, wrong authority.",
            detail={"issue": "measurement_provenance_invalid_pairs"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="The planned flag is absent from the locked cohort.",
            detail={"issue": "measurement_provenance_measured_column_missing"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="The companion column is ambiguous.",
            detail={"issue": "measurement_provenance_count_column_ambiguous"},
        ),
    ]

    assert _locked_measurement_data_quality_issues(findings) == [
        "measurement_provenance_count_column_ambiguous",
        "measurement_provenance_count_flag_discordance",
        "measurement_provenance_invalid_pairs",
        "measurement_provenance_measured_column_missing",
    ]


def test_locked_measurement_data_quality_terminates_before_code_repair():
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    route_start = source.index(
        "locked_data_quality_issues = (", source.index("early_contract_errors = [")
    )
    route_end = source.index("if sealed_renderer_authorized_code_sha256", route_start)
    terminal_route = source[route_start:route_end]

    assert "measurement_provenance_repair_suppressed" in terminal_route
    assert '"diagnostic_only": True' in terminal_route
    assert '"locked_cohort_data_quality_failed"' in terminal_route
    assert "return step_record" in terminal_route
    assert "_deterministic_summary_repair" not in terminal_route
    assert "deterministic_contract_repair" not in terminal_route
    assert "coder.repair" not in terminal_route


def test_locked_measurement_preflight_runs_before_every_coder_repair():
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    preflight = source.index("audit_locked_measurement_data_quality(")
    first_coder_repair = source.index("coder.repair(")

    assert preflight < first_coder_repair


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

    # Initial concept, post-mutation concept, visual, contract, and runtime
    # repairs each mark the same lineage flag after a successful coder call.
    assert source.count("llm_repair_used = True") == 5
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


def test_execute_phase_host_verifies_measurement_provenance_at_every_contract_gate():
    import ast

    from easyicu.research_agent import pipeline_execute

    tree = ast.parse(inspect.getsource(pipeline_execute.run_execute_phase))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "audit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "step_summary_integrity_validator"
    ]

    assert len(calls) == 3
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords.get("cohort_path"), ast.Name)
        assert keywords["cohort_path"].id == "cohort_path"


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
