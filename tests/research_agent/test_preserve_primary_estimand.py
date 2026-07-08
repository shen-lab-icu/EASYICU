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
    _preserve_primary_estimand_step_after_replan,
    _step_is_primary_estimand_model,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(step_id: str, method: str | None) -> AnalysisStep:
    return AnalysisStep(step_id=step_id, intent=step_id.replace("_", " "), method=method)


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
