"""Typed primary-estimand ownership and structural-contract regressions."""

from __future__ import annotations

import pytest

from easyicu.research_agent.plan_utils import (
    _article_display_roles,
    _cap_plan_preserving_figure_steps,
    _output_declares_figure,
    _split_table_and_figure_outputs_in_plan,
    _step_is_primary_estimand_model,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(
    step_id: str,
    method: str | None,
    *,
    planned_analysis_role: str = "auxiliary",
) -> AnalysisStep:
    expected_outputs = []
    method_text = str(method or "").lower()
    if step_id.endswith("_figure"):
        expected_outputs = ["figure:primary_result"]
    elif any(
        token in method_text for token in ("logistic", "cox", "iptw", "causal_primary")
    ):
        expected_outputs = ["statistic:primary_estimate"]
    return AnalysisStep(
        step_id=step_id,
        intent=step_id.replace("_", " "),
        method=method,
        planned_analysis_role=planned_analysis_role,
        expected_outputs=expected_outputs,
    )


def test_combined_model_plus_figure_step_stays_primary_and_is_not_duplicated():
    # Regression (2719ce4): the replanner can emit the primary model step BEFORE
    # the figure/table splitter runs, bundling a figure output into it. The guard
    # in _step_is_primary_estimand_model must exclude only a PURE figure/render
    # child, not a combined model+figure step, so structural validation does
    # not misclassify the Planner-owned primary result.
    combined = AnalysisStep(
        step_id="05_primary_adjusted_association",
        intent="primary adjusted association",
        method="adjusted_logistic_regression",
        planned_analysis_role="primary",
        expected_outputs=["statistic:adjusted_odds_ratio", "figure:forest_plot"],
    )
    figure_only = AnalysisStep(
        step_id="05_primary_adjusted_association_figure",
        intent="forest plot",
        method="figure",
        expected_outputs=["figure:forest_plot"],
    )
    assert _step_is_primary_estimand_model(combined) is True
    assert _step_is_primary_estimand_model(figure_only) is False

    assert _step_is_primary_estimand_model(combined)


def test_predicate_keys_on_method_not_id_tokens():
    # Only the real estimator method counts; descriptive/audit/sensitivity steps
    # that carry "primary"/"association" in their id do NOT.
    assert _step_is_primary_estimand_model(
        _step(
            "05_primary_adjusted_association",
            "logistic_regression",
            planned_analysis_role="primary",
        )
    )
    assert not _step_is_primary_estimand_model(
        _step(
            "04_exposure_derivation_and_absolute_risk",
            "descriptive_association_context",
        )
    )
    assert not _step_is_primary_estimand_model(
        _step("04b_primary_cohort_reconciliation_audit", "data_quality_audit")
    )
    assert not _step_is_primary_estimand_model(
        _step("06_cohort_and_missingness_sensitivity_grid", "sensitivity_analysis")
    )
    # Other estimator families are recognised too.
    assert _step_is_primary_estimand_model(
        _step(
            "05_primary_cox",
            "cox_proportional_hazards",
            planned_analysis_role="primary",
        )
    )
    assert _step_is_primary_estimand_model(
        _step(
            "05_primary_iptw",
            "causal_primary_iptw",
            planned_analysis_role="primary",
        )
    )
    # A model figure step is never the estimand.
    assert not _step_is_primary_estimand_model(
        _step("05_primary_adjusted_association_figure", "logistic_regression")
    )


def test_typed_role_is_required_before_primary_model_contract_checks():
    for role in ("auxiliary", "secondary", "sensitivity"):
        decoy = _step(
            "05_primary_adjusted_association",
            "logistic_regression",
            planned_analysis_role=role,
        )
        assert not _step_is_primary_estimand_model(decoy), role

    wrong_structure = _step(
        "05_primary_adjusted_association",
        "data_quality_audit",
        planned_analysis_role="primary",
    )
    assert not _step_is_primary_estimand_model(wrong_structure)


def test_article_display_primary_estimand_requires_typed_primary_role():
    primary = _step(
        "05_primary_adjusted_association",
        "logistic_regression",
        planned_analysis_role="primary",
    )
    sensitivity = primary.model_copy(
        update={
            "step_id": "06_sensitivity_adjusted_association",
            "planned_analysis_role": "sensitivity",
        }
    )

    assert "primary_estimand" in _article_display_roles([primary])
    assert "primary_estimand" not in _article_display_roles([sensitivity])


def test_split_primary_model_keeps_primary_parent_and_auxiliary_figure_child():
    combined = AnalysisStep(
        step_id="05_primary_adjusted_association",
        intent="Fit the Planner-selected primary model and render its forest plot.",
        method="logistic_regression",
        planned_analysis_role="primary",
        expected_outputs=[
            "table:association_estimates",
            "figure:forest_plot",
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(
        AnalysisPlan(research_question="q", steps=[combined])
    )

    assert findings
    assert [step.planned_analysis_role for step in revised.steps] == [
        "primary",
        "auxiliary",
    ]
    assert _step_is_primary_estimand_model(revised.steps[0])
    assert not _step_is_primary_estimand_model(revised.steps[1])


def test_method_rider_cannot_hide_a_primary_effect_owner():
    step = AnalysisStep(
        step_id="05_primary_association",
        intent="Estimate the association with hospital-clustered standard errors.",
        method="mixed_effects_regression_with_cohort_robust_se",
        planned_analysis_role="primary",
        expected_outputs=["table:association_estimates"],
    )
    assert _step_is_primary_estimand_model(step)


def test_typed_table_product_named_figure_does_not_launder_primary_owner():
    step = AnalysisStep(
        step_id="05_primary_association",
        intent="Fit the adjusted primary association model.",
        method="logistic_regression",
        planned_analysis_role="primary",
        expected_outputs=[
            "statistic:primary_or",
            "table:figure_summary",
        ],
    )

    assert not _output_declares_figure("table:figure_summary")
    assert _output_declares_figure("figure:primary_result")
    assert _step_is_primary_estimand_model(step)

    plan = AnalysisPlan(research_question="q", steps=[step])
    revised, findings = _split_table_and_figure_outputs_in_plan(plan)

    assert revised is plan
    assert findings == []


def test_propensity_preparation_is_not_a_primary_estimand_owner():
    prep = AnalysisStep(
        step_id="02_propensity_scores",
        intent="Estimate propensity scores for the planned weighting model.",
        method="propensity_score_estimation",
        expected_outputs=["table:propensity_scores"],
    )
    assert not _step_is_primary_estimand_model(prep)


def test_primary_estimand_survives_plan_cap():
    """The host cap may remove auxiliary work but not a Planner primary."""

    current = AnalysisPlan(
        research_question="Generic ICU association study",
        steps=[
            _step("01_cohort", "cohort_definition"),
            AnalysisStep(
                step_id="02_table_one",
                intent="Baseline characteristics",
                method="descriptive",
                expected_outputs=["table:table_one"],
            ),
            _step("03_missingness", "data_quality_audit"),
            _step("04_absolute_risk", "descriptive_context"),
            _step(
                "05_primary_adjusted_model",
                "logistic_regression",
                planned_analysis_role="primary",
            ),
            _step("06_sensitivity", "sensitivity_analysis"),
        ],
    )
    revised = AnalysisPlan(
        research_question=current.research_question,
        steps=[
            _step("00_probe", "data_probe"),
            _step("01_cohort", "cohort_definition"),
            _step("02_reconciliation", "data_quality_reconciliation"),
            _step("03_repair", "evidence_repair"),
            *current.steps[1:],
            _step("04_repair_figure", "figure"),
            _step("05_primary_adjusted_model_figure", "figure"),
        ],
    )

    capped, _ = _cap_plan_preserving_figure_steps(plan=revised, cap=8)
    step_ids = [step.step_id for step in capped.steps]

    assert len(step_ids) == 8
    assert "02_table_one" in step_ids
    assert "05_primary_adjusted_model" in step_ids


@pytest.mark.parametrize(
    ("method", "output"),
    [
        ("cox_proportional_hazards", "table:survival_estimates"),
        ("kmeans", "table:cluster_assignments"),
        ("g_computation", "table:causal_contrasts"),
    ],
)
def test_late_primary_owner_survives_cap_independent_of_method_family(
    method: str,
    output: str,
) -> None:
    steps = [
        AnalysisStep(
            step_id=f"0{index}_aux",
            intent="Prepare supporting material.",
            expected_outputs=[f"table:support_{index}"],
        )
        for index in range(1, 5)
    ]
    steps.append(
        AnalysisStep(
            step_id="05_primary",
            intent="Estimate the Planner-selected headline result.",
            method=method,
            planned_analysis_role="primary",
            expected_outputs=[output],
        )
    )
    capped, _ = _cap_plan_preserving_figure_steps(
        plan=AnalysisPlan(research_question="q", steps=steps),
        cap=4,
    )
    assert "05_primary" in {step.step_id for step in capped.steps}
