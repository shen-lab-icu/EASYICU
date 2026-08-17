"""Plan-phase display-contract guards: a result-bearing plan must declare a
publication figure and an audit/robustness panel.

E1 run13 scored plan=0.6: deepseek's plan declared a table-one but no figure
step (its question never says "figure", so the question-gated guard skipped it)
and no audit panel — even though the publication-figure skill produces a figure
and the run produces robustness/data-quality evidence. The scorer reads
``analysis_plan.json``, so the fix declares both at plan phase:
``_ensure_publication_figure_step_in_plan(force=...)`` and
``ensure_data_quality_figure_step``.
"""

from __future__ import annotations

from easyicu.research_agent.evaluation_scorecard import score_plan
from easyicu.research_agent.icu_agent_bench import ICUAgentBenchTask
from easyicu.research_agent.plan_utils import (
    _ensure_publication_figure_step_in_plan,
    _step_produces_figure,
)
from easyicu.research_agent.planning.figure_plan_shaping import (
    bind_deterministic_figure_panels,
    dedicated_renderer_consumes_typed_source,
    ensure_cohort_accounting_figure_step,
    ensure_data_quality_figure_step,
    ensure_primary_result_figure_step,
    step_declares_audit_panel,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _table_only_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=(
            "Among adult ICU patients, what is the prevalence of Sepsis-3 and "
            "is it associated with in-hospital mortality after adjustment?"
        ),  # deliberately never says "figure" / "plot"
        steps=[
            AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="02_primary_model",
                intent="Fit the adjusted logistic regression.",
                expected_outputs=["table:regression_results"],
            ),
        ],
    )


class _Ctx:
    def __init__(self, question: str):
        self.research_question = question


def test_figure_guard_needs_force_when_question_is_silent():
    plan = _table_only_plan()
    ctx = _Ctx(plan.research_question)

    # Default (question-gated): the question never says "figure", so no step.
    unforced, findings = _ensure_publication_figure_step_in_plan(
        plan=plan, context=ctx, force=False
    )
    assert not any(_step_produces_figure(s) for s in unforced.steps)
    assert findings == []

    # Forced (a figure WILL be produced): declare it.
    forced, findings = _ensure_publication_figure_step_in_plan(
        plan=plan, context=ctx, force=True
    )
    assert any(_step_produces_figure(s) for s in forced.steps)
    assert findings and findings[0].validator == "plan_contract"


def _plan_with_typed_data_quality_sources() -> AnalysisPlan:
    plan = _table_only_plan()
    return plan.model_copy(
        update={
            "steps": [
                *plan.steps,
                AnalysisStep(
                    step_id="03_missingness_measurement_audit",
                    intent="Audit missingness with owner denominators.",
                    method="measurement_audit",
                    expected_outputs=["table:missingness_measurement_audit"],
                ),
                AnalysisStep(
                    step_id="04_measurement_process_audit",
                    intent="Audit measurement opportunity and repeats.",
                    method="measurement_audit",
                    expected_outputs=["table:measurement_process_audit"],
                ),
            ]
        }
    )


def test_data_quality_figure_guard_binds_typed_sources_then_is_idempotent():
    plan = _plan_with_typed_data_quality_sources()
    ctx = _Ctx(plan.research_question)

    with_audit, findings = ensure_data_quality_figure_step(plan=plan, context=ctx)
    assert any(step_declares_audit_panel(s) for s in with_audit.steps)
    figure = with_audit.steps[-1]
    assert figure.expected_outputs == ["figure:data_quality"]
    assert figure.inputs == [
        "table:missingness_measurement_audit",
        "table:measurement_process_audit",
    ]
    assert [panel.panel_id for panel in figure.figure_panels] == [
        "source_availability",
        "measurement_process_coverage",
    ]
    assert findings and findings[0].validator == "data_quality_figure_contract"

    # Idempotent: a plan that already declares an audit panel is left alone.
    again, findings2 = ensure_data_quality_figure_step(plan=with_audit, context=ctx)
    assert len(again.steps) == len(with_audit.steps)
    assert findings2 == []


def test_data_quality_figure_guard_never_invents_an_unbound_renderer():
    plan = _table_only_plan()
    shaped, findings = ensure_data_quality_figure_step(
        plan=plan,
        context=_Ctx(plan.research_question),
    )

    assert shaped == plan
    assert findings[0].detail["reason_code"] == (
        "data_quality_figure_source_not_closed"
    )
    assert not any(step_declares_audit_panel(step) for step in shaped.steps)


def test_deterministic_cohort_renderer_gets_digest_bound_panel_contract() -> None:
    step = AnalysisStep(
        step_id="06_cohort_figure",
        planned_analysis_role="auxiliary",
        intent="Render the exact cohort flow.",
        method="visualization",
        inputs=["table:cohort_flow"],
        expected_outputs=["figure:cohort_flow"],
        input_consumption_contracts=[
            {"input_key": "table:cohort_flow", "mode": "all_rows"}
        ],
    )
    plan = AnalysisPlan(research_question="Describe the cohort.", steps=[step])

    shaped, findings = bind_deterministic_figure_panels(plan=plan)

    assert [panel.model_dump() for panel in shaped.steps[0].figure_panels] == [
        {
            "schema_version": "easyicu.planned_figure_panel/1",
            "panel_id": "cohort_accounting",
            "figure_output": "figure:cohort_flow",
            "article_role": "cohort_accounting",
            "chart_type": "cohort_flow",
            "source_products": ["table:cohort_flow"],
        }
    ]
    assert findings[0].detail["reason"] == "deterministic_figure_panels_bound"


def test_deterministic_renderer_normalizes_draft_visual_semantics_before_review() -> (
    None
):
    step = AnalysisStep(
        step_id="06_cohort_figure",
        planned_analysis_role="auxiliary",
        intent="Render the exact cohort flow.",
        method="visualization",
        inputs=["table:cohort_flow"],
        expected_outputs=["figure:cohort_flow"],
        input_consumption_contracts=[
            {"input_key": "table:cohort_flow", "mode": "all_rows"}
        ],
        figure_panels=[
            {
                "panel_id": "wrong_panel",
                "figure_output": "figure:cohort_flow",
                "article_role": "distribution",
                "chart_type": "histogram",
                "source_products": ["table:cohort_flow"],
            }
        ],
    )
    plan = AnalysisPlan(research_question="Describe the cohort.", steps=[step])

    shaped, findings = bind_deterministic_figure_panels(plan=plan)

    assert shaped != plan
    assert shaped.steps[0].figure_panels[0].panel_id == "cohort_accounting"
    assert shaped.steps[0].figure_panels[0].article_role == "cohort_accounting"
    assert shaped.steps[0].figure_panels[0].chart_type == "cohort_flow"
    assert findings[0].severity == "warning"
    assert findings[0].detail["reason"] == ("deterministic_figure_panels_normalized")


def test_deterministic_renderer_is_not_bound_without_all_rows_authority() -> None:
    step = AnalysisStep(
        step_id="06_cohort_figure",
        planned_analysis_role="auxiliary",
        intent="Render a cohort summary without a cardinality contract.",
        method="visualization",
        inputs=["table:cohort_flow"],
        expected_outputs=["figure:cohort_flow"],
    )
    plan = AnalysisPlan(research_question="Describe the cohort.", steps=[step])

    shaped, findings = bind_deterministic_figure_panels(plan=plan)

    assert shaped == plan
    assert findings == []


def test_grouped_distribution_draft_is_normalized_to_point_range() -> None:
    step = AnalysisStep(
        step_id="08_age_distribution_figure",
        planned_analysis_role="auxiliary",
        intent="Plot the prespecified grouped age summary.",
        method="visualization",
        inputs=["table:distribution_prevalence"],
        expected_outputs=["figure:age_distribution"],
        input_consumption_contracts=[
            {"input_key": "table:distribution_prevalence", "mode": "all_rows"}
        ],
        figure_panels=[
            {
                "panel_id": "draft_distribution",
                "figure_output": "figure:age_distribution",
                "article_role": "distribution",
                "chart_type": "distribution_plot",
                "source_products": ["table:distribution_prevalence"],
            }
        ],
    )
    plan = AnalysisPlan(research_question="Describe age by group.", steps=[step])

    shaped, findings = bind_deterministic_figure_panels(plan=plan)

    assert [panel.model_dump() for panel in shaped.steps[0].figure_panels] == [
        {
            "schema_version": "easyicu.planned_figure_panel/1",
            "panel_id": "grouped_distribution",
            "figure_output": "figure:age_distribution",
            "article_role": "distribution",
            "chart_type": "point_range",
            "source_products": ["table:distribution_prevalence"],
        }
    ]
    assert findings[0].detail["reason"] == ("deterministic_figure_panels_normalized")


def test_primary_result_figure_is_added_even_when_secondary_figure_exists() -> None:
    primary = AnalysisStep(
        step_id="03_primary_distribution",
        planned_analysis_role="primary",
        intent="Estimate the prespecified primary descriptive result.",
        method="descriptive",
        expected_outputs=["table:exposure_outcome_distribution"],
    )
    secondary = AnalysisStep(
        step_id="04_secondary_distribution",
        planned_analysis_role="secondary",
        intent="Describe a secondary continuous variable.",
        method="descriptive_distribution",
        expected_outputs=["table:distribution_prevalence"],
    )
    secondary_figure = AnalysisStep(
        step_id="05_secondary_figure",
        planned_analysis_role="auxiliary",
        intent="Render the secondary distribution.",
        method="visualization",
        inputs=["table:distribution_prevalence"],
        expected_outputs=["figure:secondary_distribution"],
        input_consumption_contracts=[
            {"input_key": "table:distribution_prevalence", "mode": "all_rows"}
        ],
    )
    plan = AnalysisPlan(
        research_question="Describe the primary exposure and outcome.",
        steps=[primary, secondary, secondary_figure],
    )

    shaped, findings = ensure_primary_result_figure_step(plan=plan)
    shaped, panel_findings = bind_deterministic_figure_panels(plan=shaped)

    assert len(shaped.steps) == 4
    hero = shaped.steps[-1]
    assert hero.inputs == ["table:exposure_outcome_distribution"]
    assert hero.figure_panels[0].article_role == "distribution"
    assert hero.figure_panels[0].chart_type == "prevalence_panel"
    assert hero.figure_panels[1].chart_type == "dot_interval_absolute_risk"
    assert findings[0].detail["reason"] == (
        "primary_result_figure_bound_to_typed_primary_source"
    )
    assert panel_findings[0].detail["reason"] == ("deterministic_figure_panels_bound")


def test_counts_only_primary_figure_has_no_interval_panel() -> None:
    primary = AnalysisStep.model_validate(
        {
            "step_id": "03_primary_distribution",
            "planned_analysis_role": "primary",
            "intent": "Report counts and observed proportions only.",
            "method": "descriptive",
            "inputs": ["artifact:analysis_cohort", "exposure", "outcome"],
            "expected_outputs": ["table:exposure_outcome_distribution"],
            "exposure_outcome_distribution_spec": {
                "schema_version": "easyicu.exposure_outcome_distribution/3",
                "exposure": "exposure",
                "exposure_levels": [0, 1],
                "outcome": "outcome",
                "outcome_levels": [0, 1],
                "outcome_positive_value": 1,
                "level_match_policy": "exact_typed",
                "denominator_policy": "all_declared_rows",
                "missing_outcome_policy": "structural_absence_is_non_event",
                "interval_method": "none_counts_only",
                "repeated_unit_interval_method": None,
                "confidence_level": None,
            },
        }
    )
    plan = AnalysisPlan(research_question="Describe counts.", steps=[primary])

    shaped, _ = ensure_primary_result_figure_step(plan=plan)
    shaped, _ = bind_deterministic_figure_panels(plan=shaped)

    assert shaped.steps[-1].figure_panels[1].chart_type == "point_absolute_risk"


def test_typed_measurement_alias_is_normalized_to_availability_panel() -> None:
    producer = AnalysisStep(
        step_id="05_measurement_missingness_audit",
        planned_analysis_role="auxiliary",
        intent="Audit source availability.",
        method="measurement_audit",
        expected_outputs=["table:missingness_data_quality"],
        measurement_audit_spec={
            "products": [
                {
                    "product_id": "missingness_data_quality",
                    "audit": "measurement_missingness",
                }
            ]
        },
    )
    figure = AnalysisStep(
        step_id="09_missingness_figure",
        planned_analysis_role="auxiliary",
        intent="Render the audited source availability.",
        method="visualization",
        inputs=["table:missingness_data_quality"],
        expected_outputs=["figure:missingness_data_quality"],
        input_consumption_contracts=[
            {"input_key": "table:missingness_data_quality", "mode": "all_rows"}
        ],
        figure_panels=[
            {
                "panel_id": "draft_heatmap",
                "figure_output": "figure:missingness_data_quality",
                "article_role": "data_quality",
                "chart_type": "coverage_heatmap",
                "source_products": ["table:missingness_data_quality"],
            }
        ],
    )
    plan = AnalysisPlan(
        research_question="Audit measurement availability.",
        steps=[producer, figure],
    )

    shaped, findings = bind_deterministic_figure_panels(plan=plan)

    assert [panel.model_dump() for panel in shaped.steps[1].figure_panels] == [
        {
            "schema_version": "easyicu.planned_figure_panel/1",
            "panel_id": "source_availability",
            "figure_output": "figure:missingness_data_quality",
            "article_role": "data_quality",
            "chart_type": "availability_panel",
            "source_products": ["table:missingness_data_quality"],
        }
    ]
    assert findings[0].detail["reason"] == ("deterministic_figure_panels_normalized")


def test_untyped_or_wrong_measurement_alias_is_never_normalized() -> None:
    figure = AnalysisStep(
        step_id="09_missingness_figure",
        planned_analysis_role="auxiliary",
        intent="Render a table whose audit meaning is not closed.",
        method="visualization",
        inputs=["table:missingness_data_quality"],
        expected_outputs=["figure:missingness_data_quality"],
        input_consumption_contracts=[
            {"input_key": "table:missingness_data_quality", "mode": "all_rows"}
        ],
    )
    no_owner = AnalysisPlan(
        research_question="Audit measurement availability.", steps=[figure]
    )
    wrong_owner = AnalysisPlan(
        research_question="Audit measurement availability.",
        steps=[
            AnalysisStep(
                step_id="05_event_timing_audit",
                planned_analysis_role="auxiliary",
                intent="Audit event timing.",
                method="measurement_audit",
                expected_outputs=["table:missingness_data_quality"],
                measurement_audit_spec={
                    "products": [
                        {
                            "product_id": "missingness_data_quality",
                            "audit": "event_timing",
                        }
                    ]
                },
            ),
            figure,
        ],
    )

    for plan in (no_owner, wrong_owner):
        shaped, findings = bind_deterministic_figure_panels(plan=plan)
        assert shaped == plan
        assert findings == []


def test_plan_dimension_lifts_after_both_guards():
    task = ICUAgentBenchTask(
        task_id="t1",
        kind="descriptive_association",
        title="Sepsis-3 mortality",
        objective="Prevalence and adjusted association with mortality.",
        difficulty="basic",  # non-advanced -> needs >= 1 result figure
    )
    gates = {"required_step_count": 2}

    before = score_plan(
        task,
        plan_steps=[s.model_dump() for s in _table_only_plan().steps],
        gates=gates,
    )
    assert before.signals["result_figure_count"] == 0
    assert before.signals["has_audit_panel"] is False

    plan = _plan_with_typed_data_quality_sources()
    ctx = _Ctx(plan.research_question)
    plan, _ = _ensure_publication_figure_step_in_plan(
        plan=plan, context=ctx, force=True
    )
    plan, _ = ensure_data_quality_figure_step(plan=plan, context=ctx)

    after = score_plan(
        task,
        plan_steps=[s.model_dump() for s in plan.steps],
        gates=gates,
    )
    assert after.signals["result_figure_count"] >= 1
    assert after.signals["has_audit_panel"] is True
    assert after.subscore > before.subscore


def test_typed_source_detects_one_explicit_renderer_only():
    source = "table:robustness_matrix"
    renderer = AnalysisStep(
        step_id="04_robustness_figure",
        intent="Render the verified robustness matrix.",
        method="visualization",
        inputs=[source],
        expected_outputs=["figure:robustness"],
        input_consumption_contracts=[
            {"input_key": source, "mode": "all_rows"}
        ],
    )
    assert dedicated_renderer_consumes_typed_source([renderer], source=source)

    ambiguous_renderer = renderer.model_copy(
        update={"expected_outputs": ["figure:robustness", "figure:second_view"]}
    )
    assert not dedicated_renderer_consumes_typed_source(
        [ambiguous_renderer], source=source
    )


def test_mixed_renderer_does_not_suppress_exact_article_role_renderers() -> None:
    cohort = AnalysisStep(
        step_id="cohort_denominator",
        planned_analysis_role="auxiliary",
        intent="Define the cohort and account for attrition.",
        method="cohort_definition_and_attrition",
        expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
    )
    audit = AnalysisStep(
        step_id="measurement_process_audit",
        planned_analysis_role="auxiliary",
        intent="Audit source availability and measurement opportunity.",
        method="missing_data",
        expected_outputs=[
            "table:measurement_process_audit",
            "table:measurement_missingness",
        ],
        measurement_audit_spec={
            "products": [
                {
                    "product_id": "measurement_process_audit",
                    "audit": "measurement_process",
                },
                {
                    "product_id": "measurement_missingness",
                    "audit": "measurement_missingness",
                },
            ]
        },
    )
    primary = AnalysisStep(
        step_id="exposure_outcome_distribution",
        planned_analysis_role="primary",
        intent="Describe the prespecified exposure and outcome.",
        method="descriptive",
        expected_outputs=["table:exposure_outcome_distribution"],
    )
    mixed = AnalysisStep(
        step_id="descriptive_figure_suite",
        planned_analysis_role="auxiliary",
        intent="Render a broad descriptive display.",
        method="visualization",
        inputs=[
            "exposure",
            "table:measurement_process_audit",
            "table:exposure_outcome_distribution",
        ],
        expected_outputs=["figure:descriptive_figure_suite"],
        input_consumption_contracts=[
            {"input_key": "table:measurement_process_audit", "mode": "all_rows"},
            {
                "input_key": "table:exposure_outcome_distribution",
                "mode": "all_rows",
            },
        ],
    )
    plan = AnalysisPlan(
        research_question="Describe the cohort, exposure, outcome, and data quality.",
        steps=[cohort, audit, primary, mixed],
    )

    assert not dedicated_renderer_consumes_typed_source(
        plan.steps,
        source="table:exposure_outcome_distribution",
    )
    shaped, _ = ensure_primary_result_figure_step(plan=plan)
    shaped, _ = ensure_cohort_accounting_figure_step(plan=shaped)
    shaped, _ = ensure_data_quality_figure_step(
        plan=shaped,
        context=_Ctx(plan.research_question),
    )
    shaped, _ = bind_deterministic_figure_panels(plan=shaped)

    exact_renderers = {
        tuple(step.inputs): step
        for step in shaped.steps
        if step.method == "visualization" and len(step.inputs) <= 2
    }
    assert set(exact_renderers) == {
        ("table:exposure_outcome_distribution",),
        ("table:cohort_flow",),
        (
            "table:measurement_missingness",
            "table:measurement_process_audit",
        ),
    }
    data_quality = exact_renderers[
        (
            "table:measurement_missingness",
            "table:measurement_process_audit",
        )
    ]
    assert [panel.article_role for panel in data_quality.figure_panels] == [
        "data_quality",
        "data_quality",
    ]
    assert [panel.source_products for panel in data_quality.figure_panels] == [
        ["table:measurement_missingness"],
        ["table:measurement_process_audit"],
    ]
    assert exact_renderers[("table:cohort_flow",)].figure_panels[0].article_role == (
        "cohort_accounting"
    )
    assert exact_renderers[
        ("table:exposure_outcome_distribution",)
    ].figure_panels[0].article_role == "distribution"
