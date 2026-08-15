"""Generic typed-DAG guards for plan capping and execution order."""

import pytest

from easyicu.research_agent.plan_utils import (
    _augment_report_typed_product_inputs,
    _cap_plan_preserving_figure_steps,
    _typed_plan_dag_findings,
)
from easyicu.research_agent.pipeline import (
    _defer_typed_plan_dag_findings_until_probe,
)
from easyicu.research_agent.execution.owner_declaration import (
    owner_declaration_plan_findings,
)
from easyicu.research_agent.planning.replan_gate import (
    partition_replan_candidate_findings,
    replan_candidate_contract_findings,
    replan_candidate_rejection_finding,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
    ValidationFinding,
)


def _step(
    step_id: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    method: str = "descriptive",
) -> AnalysisStep:
    return AnalysisStep(
        step_id=step_id,
        intent=step_id.replace("_", " "),
        inputs=inputs or [],
        expected_outputs=outputs or [],
        method=method,
    )


def test_cap_keeps_recursive_typed_producers_before_consumer_and_drops_figures():
    plan = AnalysisPlan(
        research_question="Generic adjusted association",
        steps=[
            _step("00_context", outputs=["table:context"]),
            _step("01_display_a", outputs=["figure:context"]),
            _step(
                "02_primary_model",
                inputs=["artifact:analysis_frame", "manifest:model_spec"],
                outputs=["statistic:adjusted_odds_ratio"],
                method="logistic_regression",
            ),
            _step("03_display_b", outputs=["figure:diagnostic_b"]),
            _step("04_audit", outputs=["table:audit"]),
            _step("05_display_c", outputs=["figure:diagnostic_c"]),
            _step(
                "06_analysis_frame",
                inputs=["dataset:source_rows"],
                outputs=["artifact:analysis_frame"],
            ),
            _step("07_source_rows", outputs=["dataset:source_rows"]),
            _step("08_model_spec", outputs=["manifest:model_spec"]),
        ],
    )

    capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=6)
    step_ids = [step.step_id for step in capped.steps]

    assert len(step_ids) == 6
    assert {
        "02_primary_model",
        "06_analysis_frame",
        "07_source_rows",
        "08_model_spec",
    } <= set(step_ids)
    assert step_ids.index("07_source_rows") < step_ids.index("06_analysis_frame")
    assert step_ids.index("06_analysis_frame") < step_ids.index("02_primary_model")
    assert step_ids.index("08_model_spec") < step_ids.index("02_primary_model")
    assert len({"01_display_a", "03_display_b", "05_display_c"} & set(step_ids)) <= 1
    assert not [finding for finding in findings if finding.severity == "error"]
    assert _typed_plan_dag_findings(capped) == []


def test_under_cap_plan_is_stably_topologically_ordered():
    plan = AnalysisPlan(
        research_question="Generic product chain",
        steps=[
            _step(
                "01_consumer", inputs=["artifact:prepared"], outputs=["table:result"]
            ),
            _step("02_unrelated", outputs=["table:notes"]),
            _step("03_producer", outputs=["artifact:prepared"]),
        ],
    )

    normalized, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=5)

    assert [step.step_id for step in normalized.steps] == [
        "02_unrelated",
        "03_producer",
        "01_consumer",
    ]
    assert any(
        (finding.detail or {}).get("reason") == "typed_dependency_topological_reorder"
        for finding in findings
    )
    assert _typed_plan_dag_findings(normalized) == []


def test_missing_typed_producer_remains_fail_closed():
    plan = AnalysisPlan(
        research_question="Generic orphaned input",
        steps=[
            _step(
                "01_model",
                inputs=["artifact:missing_analysis_frame"],
                outputs=["statistic:estimate"],
            )
        ],
    )

    normalized, cap_findings = _cap_plan_preserving_figure_steps(plan=plan, cap=5)
    dag_findings = _typed_plan_dag_findings(normalized)

    assert any(finding.severity == "error" for finding in cap_findings)
    assert any(
        (finding.detail or {}).get("reason") == "typed_input_producer_missing"
        for finding in dag_findings
    )


def test_replan_candidate_contract_rejects_ambiguous_producer():
    plan = AnalysisPlan(
        research_question="Audit a measured exposure.",
        steps=[
            _step("01_audit", outputs=["table:quality_audit"]),
            _step("02_validate", outputs=["table:quality_audit"]),
            _step(
                "03_figure",
                inputs=["table:quality_audit"],
                outputs=["figure:quality_audit"],
                method="visualization",
            ),
        ],
        revision=2,
    )
    context = ResearchContext(
        research_question="Audit a measured exposure.",
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[
            ConceptDescriptor(name="exposure", role=VariableRole.LAB, dtype="float64")
        ],
    )

    findings = replan_candidate_contract_findings(plan=plan, context=context)

    assert any(
        finding.severity == "error"
        and (finding.detail or {}).get("reason") == "typed_input_producer_ambiguous"
        for finding in findings
    )


def test_replan_candidate_rejects_distribution_figure_single_row_drift() -> None:
    input_key = "table:exposure_outcome_distribution"
    plan = AnalysisPlan(
        research_question="Describe an exposure and outcome.",
        steps=[
            _step("01_distribution", outputs=[input_key]),
            AnalysisStep(
                step_id="02_figure",
                intent="Render the exposure prevalence and outcome risk.",
                planned_analysis_role="auxiliary",
                method="visualization",
                inputs=[input_key],
                expected_outputs=["figure:absolute_risk"],
                input_consumption_contracts=[
                    ArtifactConsumptionContract(
                        input_key=input_key,
                        mode="single_row",
                    )
                ],
            ),
        ],
        revision=2,
    )

    owner_findings = owner_declaration_plan_findings(plan=plan)
    findings = replan_candidate_contract_findings(
        plan=plan,
        context=ResearchContext(
            research_question=plan.research_question,
            cohort=CohortDescriptor(
                cohort_name="cohort",
                database="synthetic",
                n_patients=10,
                n_stays=10,
            ),
            variables=[],
        ),
        owner_declaration_findings=owner_findings,
    )

    assert any(
        finding.validator == "plan_owner_declaration"
        and (finding.detail or {}).get("analysis_kind")
        == "exposure_outcome_distribution_figure"
        for finding in findings
    )


def test_rejected_replan_candidate_errors_remain_diagnostic_only():
    normalization_error = ValidationFinding(
        validator="replanner",
        severity="error",
        message="Candidate output kind is not materializable.",
        detail={
            "reason": "typed_output_kind_not_materializable",
            "typed_product": "protocol:robustness",
        },
    )
    normalization_warning = ValidationFinding(
        validator="replanner",
        severity="warning",
        message="Immutable scope was restored.",
        detail={"reason": "completed_step_snapshot_immutable"},
    )
    duplicate_contract_error = normalization_error.model_copy(
        update={"validator": "replanner"}
    )

    active, errors = partition_replan_candidate_findings(
        normalization_findings=[normalization_warning, normalization_error],
        contract_findings=[duplicate_contract_error],
    )
    rejection = replan_candidate_rejection_finding(
        contract_errors=errors,
        trigger="probe_summary",
        candidate_revision=3,
    )

    assert active == [normalization_warning]
    assert len(errors) == 1
    assert rejection.severity == "warning"
    assert rejection.detail["contract_findings"][0]["detail"] == {
        "reason": "typed_output_kind_not_materializable",
        "typed_product": "protocol:robustness",
    }


def test_preprobe_typed_error_becomes_pending_but_unrelated_error_stays_current():
    initial = [
        ValidationFinding(
            validator="plan_typed_dag",
            severity="error",
            message="Multiple declared producers require planner repair.",
            detail={
                "reason": "typed_input_producer_ambiguous",
                "typed_product": "table:result",
            },
        ),
        ValidationFinding(
            validator="planner",
            severity="error",
            message="The plan cap cannot preserve all protected steps.",
            detail={"reason": "typed_dependency_closure_exceeds_cap"},
        ),
    ]

    deferred = _defer_typed_plan_dag_findings_until_probe(initial)

    assert deferred[0].validator == "plan_contract_pending"
    assert deferred[0].severity == "warning"
    assert deferred[0].detail == {
        "reason": "typed_input_producer_ambiguous",
        "typed_product": "table:result",
        "pending_probe_replan": True,
        "original_validator": "plan_typed_dag",
    }
    assert deferred[1] == initial[1]


@pytest.mark.parametrize("kind", ["feature", "qc"])
def test_plan_rejects_typed_kinds_without_materialization_and_binding(kind: str):
    product = f"{kind}:derived_product"
    plan = AnalysisPlan(
        research_question="Generic derived product contract",
        steps=[
            _step("01_derive", inputs=["raw_value"], outputs=[product]),
            _step("02_consume", inputs=[product], outputs=["table:result"]),
        ],
    )

    findings = _typed_plan_dag_findings(plan)
    reasons = {(finding.detail or {}).get("reason") for finding in findings}

    assert "typed_output_kind_not_materializable" in reasons
    assert "typed_input_kind_not_runtime_bindable" in reasons
    assert "typed_input_producer_missing" not in reasons
    assert all(finding.severity == "error" for finding in findings)


@pytest.mark.parametrize("kind", ["dataset", "table"])
def test_plan_accepts_materializable_runtime_bindable_product_kinds(kind: str):
    product = f"{kind}:derived_product"
    plan = AnalysisPlan(
        research_question="Generic derived product contract",
        steps=[
            _step("01_derive", inputs=["raw_value"], outputs=[product]),
            _step("02_consume", inputs=[product], outputs=["table:result"]),
        ],
    )

    assert _typed_plan_dag_findings(plan) == []


def test_report_consumes_unique_prior_typed_results_without_raw_recomputation():
    plan = AnalysisPlan(
        research_question="Generic observational analysis",
        steps=[
            _step("01_cohort", outputs=["table:cohort_flow", "artifact:frame"]),
            _step("02_model", outputs=["statistic:primary_estimate"]),
            _step("03_figure", outputs=["figure:primary_estimate"]),
            _step(
                "04_report",
                inputs=["raw_marker"],
                outputs=["report:analysis_results"],
                method="scientific_reporting",
            ),
        ],
    )

    revised, findings = _augment_report_typed_product_inputs(plan=plan)
    report = revised.steps[-1]

    assert report.inputs == [
        "raw_marker",
        "table:cohort_flow",
        "statistic:primary_estimate",
    ]
    assert "artifact:frame" not in report.inputs
    assert "figure:primary_estimate" not in report.inputs
    assert (findings[0].detail or {}).get("reason") == (
        "report_typed_product_input_closure"
    )
    assert _typed_plan_dag_findings(revised) == []


def test_report_closure_skips_ambiguous_and_later_products():
    plan = AnalysisPlan(
        research_question="Generic reporting dependencies",
        steps=[
            _step("01_a", outputs=["table:duplicate"]),
            _step("02_b", outputs=["table:duplicate"]),
            _step("03_report", outputs=["report:interim"]),
            _step("04_later", outputs=["table:later_result"]),
        ],
    )

    revised, findings = _augment_report_typed_product_inputs(plan=plan)

    assert revised.steps[2].inputs == []
    assert findings == []
