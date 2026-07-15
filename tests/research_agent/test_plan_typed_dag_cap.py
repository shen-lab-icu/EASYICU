"""Generic typed-DAG guards for plan capping and execution order."""

from easyicu.research_agent.plan_utils import (
    _augment_report_typed_product_inputs,
    _cap_plan_preserving_figure_steps,
    _typed_plan_dag_findings,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


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
            _step("01_consumer", inputs=["artifact:prepared"], outputs=["table:result"]),
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
        (finding.detail or {}).get("reason")
        == "typed_dependency_topological_reorder"
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
