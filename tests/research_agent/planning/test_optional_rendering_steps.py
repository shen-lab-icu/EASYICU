"""Optional presentation cannot turn completed science into diagnostics only."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceStore,
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
)
from easyicu.research_agent.authority.runtime_artifacts import current_evidence_records
from easyicu.research_agent.planning.figure_plan_mutation import (
    optional_render_step_ids,
)
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan as _compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.reporting.readiness import (
    execution_gate_status,
    render_report,
    write_readiness_artifacts,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    UserPreferences,
    ValidationFinding,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _payload,
)


def compile_progressive_plan(**kwargs):
    # The agent adapter supplies execution ownership at final compilation.
    from easyicu.research_agent.execution import owner_declaration

    return _compile_progressive_plan(
        **kwargs,
        resolve_executor=owner_declaration.resolve_standard_executor,
    )


def test_unconsumed_untyped_outline_figure_is_omitted_with_a_recorded_note():
    skeleton = ProgressivePlanSkeleton.model_validate(_payload())
    plan, receipt = compile_progressive_plan(skeleton=skeleton, context=_context())

    assert "07_figure" not in {step.step_id for step in plan.steps}
    (note,) = receipt.omitted_render_steps
    assert note.step_id == "07_figure"
    assert note.expected_outputs == ["figure:primary_results"]
    assert note.reason == "unconsumed_optional_render_without_typed_owner"
    assert note.message in plan.rationale


@pytest.mark.parametrize("obligation", ["optional", "consumed", "requested"])
def test_host_checks_the_final_compiled_plan_without_exempting_rendering(obligation):
    from easyicu.research_agent.execution.owner_declaration import require_host_rendered_plan

    payload = _payload()
    # Keep one typed analytical step and its final optional figure.
    payload["steps"] = [s for s in payload["steps"] if s["step_id"] in {"05_primary", "07_figure"}]
    payload["steps"][0]["depends_on"] = []
    payload["steps"][1]["depends_on"] = ["05_primary"]
    payload["steps"][1]["product_inputs"] = [payload["steps"][1]["product_inputs"][1]]
    payload["robustness_intents"] = []
    context = _context()
    if obligation == "requested":
        context.user_preferences.must_have_outputs = "figure:primary_results"
    if obligation == "consumed":
        consumer = deepcopy(payload["steps"][-1])
        consumer.update(
            step_id="08_consumer", depends_on=["07_figure"],
            module_id="custom_analysis", custom_method="synthetic_figure_audit",
            product_inputs=[{"producer_step_id": "07_figure", "product_id": "figure:primary_results"}],
            outputs=[{"product_id": "table:render_audit", "semantic_role": "custom"}],
        )
        payload["steps"].append(consumer)
    plan, receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context,
    )
    if obligation == "optional":
        assert [note.step_id for note in receipt.omitted_render_steps] == ["07_figure"]
        require_host_rendered_plan(plan=plan)
    else:
        assert "07_figure" in {step.step_id for step in plan.steps}
        with pytest.raises(ValueError, match="research_pipeline_plan_step_needs_sandbox") as error:
            require_host_rendered_plan(plan=plan)
        assert error.value.easyicu_safe_diagnostic["step_id"] == "07_figure"


def _run_plan():
    return AnalysisPlan(
        research_question="Describe the synthetic measurement distribution.",
        analysis_type="descriptive_epidemiology",
        steps=[
            AnalysisStep(
                step_id="analysis",
                planned_analysis_role="primary",
                method="descriptive_cohort_summary",
                intent="Summarize the measurement.",
                scientific_action_id="descriptive.descriptive_summary",
                inputs=["artifact:analysis_cohort", "age_years"],
                expected_outputs=["table:cohort_summary", "table:distribution_summary"],
            ),
            AnalysisStep(
                step_id="presentation",
                method="visualization",
                planned_analysis_role="auxiliary",
                intent="Render the summary.",
                inputs=["table:cohort_summary"],
                expected_outputs=["figure:summary"],
            ),
        ],
    )


def _records():
    return [
        {"step_id": "analysis", "status": "ok", "step_summary": {"status": "ok"}},
        {"step_id": "presentation", "status": "failed"},
    ]


@pytest.mark.parametrize("finding_shape", ["typed", "runner", "timeout", "coder"])
def test_a_failed_optional_figure_preserves_validated_analysis_results(
    tmp_path, finding_shape
):
    plan, records = _run_plan(), _records()
    findings = [
        ValidationFinding(
            validator="executor",
            severity="error",
            message="Rendering failed.",
            detail={"step_id": "presentation"},
        )
    ]
    if finding_shape != "typed":
        # These are the legacy terminal messages emitted by candidate_loop.
        findings = [
            ValidationFinding(
                validator="runner" if finding_shape != "coder" else "coder",
                severity="error",
                message={
                    "runner": "Step presentation failed with returncode 1.",
                    "timeout": "Step presentation timed out with returncode 1.",
                    "coder": "Coder repair failed after post-mutation concept audit for step presentation: unavailable.",
                }[finding_shape],
            )
        ]
    evidence = EvidenceStore(tmp_path)
    gates, _ = write_readiness_artifacts(
        context=_context(),
        plan=plan,
        per_step_records=records,
        findings=findings,
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=tmp_path / "absent.md",
        stop_after_analysis=True,
    )
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "analysis_only"
    assert gates["execution_complete"] is True
    assert gates["analysis_validated"] is True, gates["analysis_errors"]
    assert gates["failed_steps"] == []
    assert gates["optional_render_failures"][0]["step_id"] == "presentation"
    assert gates["step_completion_states"][1]["execution_ok"] is False
    assert gates["evidence_complete"] is False
    assert gates["paper_authorized"] is False
    report = render_report(
        context=_context(),
        plan=plan,
        findings=findings,
        per_step_records=records,
        evidence=evidence,
        readiness=gates,
    )
    assert "Optional figure limitations" in report
    assert "presentation" in report


@pytest.mark.parametrize(
    "obligation", ["product_consumer", "dependency", "requested", "required_rule"]
)
def test_compiler_retains_consumed_or_required_figures(obligation):
    payload, context = _payload(), _context()
    kwargs = {}
    if obligation in {"product_consumer", "dependency"}:
        consumer = deepcopy(payload["steps"][-1])
        consumer.update(
            step_id="08_consumer",
            module_id="report",
            depends_on=["07_figure", "05_primary"],
            product_inputs=(
                [
                    {
                        "producer_step_id": "07_figure",
                        "product_id": "figure:primary_results",
                    }
                ]
                if obligation == "product_consumer"
                else [
                    {
                        "producer_step_id": "05_primary",
                        "product_id": "table:adjusted_association_estimates",
                    }
                ]
            ),
            outputs=[{"product_id": "report:results", "semantic_role": "report"}],
        )
        payload["steps"].append(consumer)
    elif obligation == "requested":
        context.user_preferences.must_have_outputs = "figure:primary_results"
    else:
        kwargs["required_visualization_step"] = True
    plan, receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload),
        context=context,
        **kwargs,
    )
    assert "07_figure" in {step.step_id for step in plan.steps}
    assert not receipt.omitted_render_steps


def test_final_compilation_keeps_untyped_analysis_and_prefix_keeps_future_sources():
    skeleton = ProgressivePlanSkeleton.model_validate(_payload())
    prefix, prefix_receipt = compile_progressive_plan(
        skeleton=skeleton,
        context=_context(),
        omit_optional_render_steps=False,
    )
    final, receipt = compile_progressive_plan(skeleton=skeleton, context=_context())
    assert prefix.steps[-1].step_id == "07_figure"
    assert not prefix_receipt.omitted_render_steps
    # The prefix receipt remains a complete source-coordinate ledger.
    assert receipt.compiled_steps == prefix_receipt.compiled_steps
    assert final.steps[-1].model_dump() == prefix.steps[-2].model_dump()
    assert final.steps[-1].planned_analysis_role == "sensitivity"


def test_compiler_without_owner_authority_cannot_omit_a_step():
    plan, receipt = _compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_payload()),
        context=_context(),
    )
    assert plan.steps[-1].step_id == "07_figure"
    assert not receipt.omitted_render_steps


def test_only_the_requested_figure_product_is_required_in_host_descriptive_plans():
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                inferred_analysis_family="descriptive_epidemiology",
                must_have_outputs="plot:summary",
            )
        }
    )
    assert not optional_render_step_ids(_run_plan(), context=context)
    context.user_preferences.must_have_outputs = "table:cohort_summary"
    assert optional_render_step_ids(_run_plan(), context=context) == {"presentation"}


@pytest.mark.parametrize("owner_state", ["claimed", "unevaluated", "panels"])
def test_compiler_does_not_remove_typed_or_unevaluated_rendering(
    monkeypatch, owner_state
):
    from easyicu.research_agent.execution import owner_declaration
    from easyicu.research_agent.planning import progressive_compiler

    if owner_state == "panels":
        original = progressive_compiler._compile_one_step

        def compile_with_panel(**kwargs):
            result = original(**kwargs)
            if result.step.step_id == "07_figure":
                from easyicu.research_agent.contracts.figure_plan import (
                    PlannedFigurePanelSpec,
                )

                result.step.figure_panels = [
                    PlannedFigurePanelSpec(
                        panel_id="effect",
                        figure_output="figure:primary_results",
                        article_role="primary_result",
                        chart_type="forest_plot",
                        source_products=["table:adjusted_association_estimates"],
                    )
                ]
            return result

        monkeypatch.setattr(
            progressive_compiler, "_compile_one_step", compile_with_panel
        )
    else:

        def resolve(*args, **kwargs):
            if owner_state == "unevaluated":
                raise ValueError("Synthetic unresolved declaration")
            return SimpleNamespace(claimed_by="synthetic_typed_renderer")

        monkeypatch.setattr(owner_declaration, "resolve_standard_executor", resolve)
    plan, receipt = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_payload()),
        context=_context(),
    )
    assert "07_figure" in {step.step_id for step in plan.steps}
    assert not receipt.omitted_render_steps


@pytest.mark.parametrize(
    "updates",
    [
        {"planned_analysis_role": role}
        for role in ("primary", "secondary", "sensitivity")
    ]
    + [
        {"method": "robustness_replay"},
        {"method": "custom_analysis"},
        {"expected_outputs": []},
        {"expected_outputs": ["figure:summary", "table:estimate"]},
        {"expected_outputs": ["summary.svg"]},
        {"inputs": []},
        {"inputs": ["age_years"]},
        {"inputs": ["artifact:analysis_cohort"]},
        {"sensitivity_spec_ids": ["predeclared"]},
        {"scientific_action_id": "descriptive.descriptive_summary"},
        {"scientific_capability": "descriptive_measurement_v1"},
        {"model_requirements": [object()]},
        {"family_primary_result_requirement": object()},
        {"robustness_replay_spec": object()},
    ],
)
def test_analysis_or_unproven_rendering_failures_remain_blocking(updates):
    plan = _run_plan()
    plan.steps[1] = plan.steps[1].model_copy(update=updates)
    gate = execution_gate_status(plan=plan, per_step_records=_records())
    assert not gate["execution_complete"]
    assert gate["failed_steps"][0]["step_id"] == "presentation"
    assert not gate["optional_render_failures"]


@pytest.mark.parametrize(
    "consumer_input", ["figure:summary", "plot:summary", "presentation"]
)
def test_consumed_rendering_failure_remains_blocking(consumer_input):
    plan = _run_plan()
    plan.steps.append(
        AnalysisStep(
            step_id="consumer",
            intent="Use the rendered product.",
            inputs=[consumer_input],
        )
    )
    assert not execution_gate_status(
        plan=plan,
        per_step_records=[*_records(), {"step_id": "consumer", "status": "ok"}],
    )["execution_complete"]


def test_required_or_missing_rendering_is_not_excused():
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(must_have_outputs="Required figure")
        }
    )
    assert not execution_gate_status(
        plan=_run_plan(), per_step_records=_records(), context=context
    )["execution_complete"]
    assert not execution_gate_status(plan=_run_plan(), per_step_records=_records()[:1])[
        "execution_complete"
    ]
    assert not optional_render_step_ids(_run_plan(), required_visualization_step=True)


def test_primary_failure_still_fails_closed():
    records = _records()
    records[0]["status"] = "failed"
    gate = execution_gate_status(plan=_run_plan(), per_step_records=records)
    assert not gate["execution_complete"]
    assert gate["failed_steps"] == [{"step_id": "analysis", "status": "failed"}]


@pytest.mark.parametrize(
    "unsafe", ["input_authority_findings", "runner_output_safety_reason"]
)
def test_untrusted_authority_or_mutable_outputs_remain_blocking(unsafe):
    records = _records()
    records[1][unsafe] = (
        ["invalid input authority"]
        if unsafe == "input_authority_findings"
        else "runner_output_teardown_unconfirmed"
    )
    gate = execution_gate_status(plan=_run_plan(), per_step_records=records)
    assert gate["execution_complete"] is False
    assert gate["optional_render_failures"] == []


@pytest.mark.parametrize(
    "role,method",
    [("primary", "descriptive_cohort_summary"), ("sensitivity", "robustness_replay")],
)
def test_scientific_failure_keeps_the_run_diagnostic_only(tmp_path, role, method):
    plan = _run_plan()
    plan.steps[1] = plan.steps[1].model_copy(
        update={"planned_analysis_role": role, "method": method}
    )
    gates, _ = write_readiness_artifacts(
        context=_context(),
        plan=plan,
        per_step_records=_records(),
        findings=[],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=tmp_path / "absent.md",
        stop_after_analysis=True,
    )
    assert not gates["analysis_validated"]
    assert (
        json.loads((tmp_path / "run_status.json").read_text())["status"]
        == "diagnostic_only"
    )


def test_intent_and_identifier_do_not_decide_optional_rendering():
    plan = _run_plan()
    plan.steps[1] = plan.steps[1].model_copy(
        update={
            "step_id": "unrelated_name",
            "intent": "Any opaque description, including model words.",
        }
    )
    assert optional_render_step_ids(plan) == {"unrelated_name"}


def test_failed_figure_never_gains_evidence_authority(tmp_path):
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    with pytest.raises(EvidenceEnforcementError):
        store.bind_manuscript(
            "See {evidence:missing_figure}.", per_step_records=_records()
        )
    assert (
        current_evidence_records(
            [{"evidence_id": "stale_figure", "produced_by_step": "presentation"}],
            [
                *_records(),
                {
                    "step_id": "presentation",
                    "status": "failed",
                    "evidence_ids": ["stale_figure"],
                },
            ],
        )
        == []
    )


@pytest.mark.parametrize(
    "validator,step_id",
    [
        ("manuscript_numeric_auditor", "presentation"),
        ("evidence_bound_writer", "presentation"),
        ("executor", "analysis"),
        ("executor", None),
    ],
)
def test_unrelated_or_evidence_errors_are_still_reported(tmp_path, validator, step_id):
    records = _records()
    if step_id == "analysis":
        records[0]["status"] = "failed"
    gates, _ = write_readiness_artifacts(
        context=_context(),
        plan=_run_plan(),
        per_step_records=records,
        findings=[
            ValidationFinding(
                validator=validator,
                severity="error",
                message="Unresolved check.",
                detail={"step_id": step_id} if step_id else {},
            )
        ],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=tmp_path / "absent.md",
        stop_after_analysis=True,
    )
    assert (
        "Unresolved check."
        in gates["numeric_errors"] + gates["evidence_errors"] + gates["analysis_errors"]
    )
