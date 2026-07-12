"""Regression: the replanner must not silently drop the primary estimand step.

Root cause (2026-07-08, M1): replan revision 7 removed
``05_primary_adjusted_association`` (method=logistic_regression) while inserting
``04b_primary_cohort_reconciliation_audit`` (method=data_quality_audit), even
though its own rationale claimed it was keeping the primary model. The run then
produced NO adjusted OR and replan-exhausted to diagnostic_only.

The token-based checks could not catch it: the descriptive step
``04_exposure_derivation_and_absolute_risk`` and the audit step
``04b_primary_cohort_reconciliation_audit`` both carry "primary"/"association"
in their ids, so ``_primary_estimand_step_index`` and the role detector both
still "found" an estimand. The METHOD family is the discriminator.
"""

from __future__ import annotations

from easyicu.research_agent.plan_utils import (
    _cap_plan_preserving_figure_steps,
    _output_declares_figure,
    _preserve_primary_estimand_step_after_replan,
    _split_table_and_figure_outputs_in_plan,
    _step_is_primary_estimand_model,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(step_id: str, method: str | None) -> AnalysisStep:
    expected_outputs = []
    method_text = str(method or "").lower()
    if step_id.endswith("_figure"):
        expected_outputs = ["figure:primary_result"]
    elif any(
        token in method_text
        for token in ("logistic", "cox", "iptw", "causal_primary")
    ):
        expected_outputs = ["statistic:primary_estimate"]
    return AnalysisStep(
        step_id=step_id,
        intent=step_id.replace("_", " "),
        method=method,
        expected_outputs=expected_outputs,
    )


def _m1_rev6() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="hepatobiliary missingness -> mortality",
        steps=[
            _step("00_probe", None),
            _step("01_primary_cohort_definition_attrition", "cohort_definition"),
            _step("02_table_one_baseline_context", "descriptive"),
            _step("03_missingness_measurement_process_audit", "data_quality_audit"),
            _step("04_exposure_derivation_and_absolute_risk", "descriptive_association_context"),
            _step("05_primary_adjusted_association", "logistic_regression"),
            _step("06_cohort_and_missingness_sensitivity_grid", "sensitivity_analysis"),
        ],
    )


def _m1_rev7() -> AnalysisPlan:
    # rev7 dropped the logistic_regression estimand; added a data_quality_audit.
    return AnalysisPlan(
        research_question="hepatobiliary missingness -> mortality",
        steps=[
            _step("00_probe", None),
            _step("01_primary_cohort_definition_attrition", "cohort_definition"),
            _step("02_table_one_baseline_context", "descriptive"),
            _step("03_missingness_measurement_process_audit", "data_quality_audit"),
            _step("04_exposure_derivation_and_absolute_risk", "descriptive_association_context"),
            _step("04b_primary_cohort_reconciliation_audit", "data_quality_audit"),
            _step("06_cohort_and_missingness_sensitivity_grid", "sensitivity_analysis"),
        ],
    )


def test_predicate_keys_on_method_not_id_tokens():
    # Only the real estimator method counts; descriptive/audit/sensitivity steps
    # that carry "primary"/"association" in their id do NOT.
    assert _step_is_primary_estimand_model(
        _step("05_primary_adjusted_association", "logistic_regression")
    )
    assert not _step_is_primary_estimand_model(
        _step("04_exposure_derivation_and_absolute_risk", "descriptive_association_context")
    )
    assert not _step_is_primary_estimand_model(
        _step("04b_primary_cohort_reconciliation_audit", "data_quality_audit")
    )
    assert not _step_is_primary_estimand_model(
        _step("06_cohort_and_missingness_sensitivity_grid", "sensitivity_analysis")
    )
    # Other estimator families are recognised too.
    assert _step_is_primary_estimand_model(_step("05_primary_cox", "cox_proportional_hazards"))
    assert _step_is_primary_estimand_model(_step("05_primary_iptw", "causal_primary_iptw"))
    # A model figure step is never the estimand.
    assert not _step_is_primary_estimand_model(
        _step("05_primary_adjusted_association_figure", "logistic_regression")
    )


def test_method_rider_cannot_hide_a_primary_effect_owner():
    step = AnalysisStep(
        step_id="05_primary_association",
        intent="Estimate the association with hospital-clustered standard errors.",
        method="mixed_effects_regression_with_cohort_robust_se",
        expected_outputs=["table:association_estimates"],
    )
    assert _step_is_primary_estimand_model(step)

    current = AnalysisPlan(research_question="q", steps=[step])
    revised = AnalysisPlan(research_question="q", steps=[])
    preserved, findings = _preserve_primary_estimand_step_after_replan(
        current=current,
        revised=revised,
    )
    assert [item.step_id for item in preserved.steps] == [step.step_id]
    assert findings and findings[0].detail["preserved_step_ids"] == [step.step_id]


def test_typed_table_product_named_figure_does_not_launder_primary_owner():
    step = AnalysisStep(
        step_id="05_primary_association",
        intent="Fit the adjusted primary association model.",
        method="logistic_regression",
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


def test_propensity_prep_cannot_hide_dropped_iptw_estimand():
    current = AnalysisPlan(
        research_question="causal contrast",
        steps=[
            AnalysisStep(
                step_id="02_propensity_scores",
                intent="Prepare propensity scores.",
                method="propensity_score_estimation",
                expected_outputs=["table:propensity_scores"],
            ),
            AnalysisStep(
                step_id="03_iptw_effect",
                intent="Estimate the prespecified weighted effect.",
                method="iptw",
                expected_outputs=["statistic:adjusted_effect"],
            ),
        ],
    )
    revised = current.model_copy(update={"steps": [current.steps[0]]})

    preserved, findings = _preserve_primary_estimand_step_after_replan(
        current=current,
        revised=revised,
    )

    assert [step.step_id for step in preserved.steps] == [
        "02_propensity_scores",
        "03_iptw_effect",
    ]
    assert findings and findings[0].detail["preserved_step_ids"] == [
        "03_iptw_effect"
    ]


def test_dropped_estimand_is_reattached():
    preserved, findings = _preserve_primary_estimand_step_after_replan(
        current=_m1_rev6(), revised=_m1_rev7()
    )
    ids = [s.step_id for s in preserved.steps]
    assert "05_primary_adjusted_association" in ids
    assert findings and findings[0].severity == "warning"
    assert "05_primary_adjusted_association" in findings[0].detail["preserved_step_ids"]


def test_noop_when_revised_keeps_an_estimand():
    # Healthy replan (estimand still present) -> unchanged, no findings.
    preserved, findings = _preserve_primary_estimand_step_after_replan(
        current=_m1_rev6(), revised=_m1_rev6()
    )
    assert [s.step_id for s in preserved.steps] == [s.step_id for s in _m1_rev6().steps]
    assert findings == []


def test_noop_when_estimand_renamed_but_present():
    # The replanner may legitimately RENAME the estimand step (same method). It
    # must NOT be duplicated, since a model producing the estimand still exists.
    renamed = AnalysisPlan(
        research_question="q",
        steps=[
            _step("01_cohort", "cohort_definition"),
            _step("05_adjusted_primary_model_v2", "logistic_regression"),
        ],
    )
    preserved, findings = _preserve_primary_estimand_step_after_replan(
        current=_m1_rev6(), revised=renamed
    )
    ids = [s.step_id for s in preserved.steps]
    assert "05_primary_adjusted_association" not in ids  # not duplicated
    assert findings == []


def test_noop_when_current_had_no_estimand():
    # If the current plan itself had no model estimand (e.g. a pure descriptive
    # audit plan), there is nothing to preserve.
    descriptive_only = AnalysisPlan(
        research_question="q",
        steps=[
            _step("01_cohort", "cohort_definition"),
            _step("02_audit", "data_quality_audit"),
        ],
    )
    preserved, findings = _preserve_primary_estimand_step_after_replan(
        current=descriptive_only, revised=descriptive_only
    )
    assert [s.step_id for s in preserved.steps] == [s.step_id for s in descriptive_only.steps]
    assert findings == []


def test_primary_estimand_survives_preserve_then_plan_cap():
    """The cap must not undo the primary-preservation guard."""

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
            _step("05_primary_adjusted_model", "logistic_regression"),
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

    preserved, findings = _preserve_primary_estimand_step_after_replan(
        current=current,
        revised=revised,
    )
    assert findings == []  # raw revision still contains the genuine model

    capped, _ = _cap_plan_preserving_figure_steps(plan=preserved, cap=8)
    step_ids = [step.step_id for step in capped.steps]

    assert len(step_ids) == 8
    assert "02_table_one" in step_ids
    assert "05_primary_adjusted_model" in step_ids
