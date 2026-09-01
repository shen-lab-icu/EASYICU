"""What a proportional-hazards verdict may decide, and who decides it.

Three defects these lock, all in the survival primary contract on ``ba11f52``:

1. The PH verdict read only the ``global`` row. That row is Bonferroni --
   ``min(1, k * min_j p_j)`` -- so it cannot see a time-varying effect in the
   one coefficient the manuscript reports once the model carries a handful of
   covariates. The arithmetic is exact, not a guess: an exposure at p=0.01
   leaves the global at 0.05 for k=5 and 0.08 for k=8, both "not rejected" at
   alpha=0.05. Five covariates is an ordinary adjusted ICU Cox model.

2. ``report_only`` kept ``paper_authorization_allowed`` true through a
   significant violation. The Planner chooses that policy *before* execution,
   so declaring the loosest one pre-authorized a result the diagnostic had not
   yet examined -- the plan granting itself permission.

3. ``human_review`` named a workflow that did not exist. It produced a
   permanent block indistinguishable from ``block_paper_authorization``: no
   review request, no reviewer decision bound to the PH/result digests, no
   resumption of the publication gate.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.contracts.survival import (
    SurvivalAnalysisReceipt,
    canonical_survival_applied_filter,
    canonical_survival_formula,
)


_DIGEST = "a" * 64
_OTHER = "b" * 64
_THIRD = "c" * 64
_FOURTH = "d" * 64

_TERMS = [
    ModelTermSpec(
        name="treatment",
        role="exposure",
        coding="binary",
        levels=["0", "1"],
        reference_level="0",
        transform="treatment_contrast",
    ),
    ModelTermSpec(
        name="age", role="covariate", coding="continuous", transform="identity"
    ),
]


def _receipt(**overrides) -> SurvivalAnalysisReceipt:
    design_columns = ["treatment__is_1", "age"]
    payload = {
        "result_product": "table:survival_effect_estimates",
        "result_evidence_id": f"sha256:{_DIGEST}",
        "result_sha256": _DIGEST,
        "input_product": "table:analysis_cohort",
        "input_evidence_id": f"sha256:{_OTHER}",
        "input_sha256": _OTHER,
        "analysis_frame_sha256": _THIRD,
        "ph_diagnostic_evidence_id": f"sha256:{_FOURTH}",
        "ph_diagnostic_sha256": _FOURTH,
        "exposure_source": "treatment",
        "outcome": "death",
        "effect_scale": "hazard_ratio",
        "analysis_population": "adults",
        "n_source_rows": 100,
        "n_analysis_rows": 90,
        "n_complete_case_dropped": 10,
        "n_censored_at_horizon": 5,
        "n_events": 20,
        "time_origin": "icu_admission",
        "time_column": "followup_time",
        "time_unit": "days",
        "event_column": "death",
        "event_value": 1,
        "event_definition": "in-hospital death",
        "censoring_strategy": "administrative at 28 days",
        "competing_risk_strategy": "none",
        "time_horizon": "28 days",
        "time_horizon_value": 28.0,
        "estimator": "cox_proportional_hazards",
        "effect_measure": "hazard_ratio",
        "formula": canonical_survival_formula(
            time_column="followup_time",
            event_column="death",
            event_value=1,
            exposure_source="treatment",
            covariates=["age"],
            design_columns=design_columns,
        ),
        "covariates": ["age"],
        "model_terms": _TERMS,
        "design_columns": design_columns,
        "exposure_design_column": "treatment__is_1",
        "applied_filter": canonical_survival_applied_filter(
            time_column="followup_time",
            event_column="death",
            event_value=1,
            exposure_source="treatment",
            covariates=["age"],
            model_terms=_TERMS,
            time_horizon_value=28.0,
            time_unit="days",
        ),
        "package_versions": {
            "easyicu": "1.0.0",
            "lifelines": "0.30.3",
            "pandas": "2.0.0",
        },
        "proportional_hazards_diagnostic": "schoenfeld_global_test",
        "proportional_hazards_p_value": 0.90,
        "proportional_hazards_exposure_p_value": 0.80,
        "proportional_hazards_alpha": 0.05,
        "proportional_hazards_policy": "block_paper_authorization",
        "proportional_hazards_status": "not_rejected",
        "paper_authorization_allowed": True,
    }
    payload.update(overrides)
    return SurvivalAnalysisReceipt(**payload)


# --- 1. the exposure's own PH test is part of the verdict --------------------


def test_the_bonferroni_global_hides_an_exposure_violation() -> None:
    """The arithmetic that makes the global row insufficient."""

    alpha, exposure_p = 0.05, 0.01
    for k, hidden in ((2, False), (5, True), (8, True)):
        global_p = min(1.0, k * exposure_p)
        assert (exposure_p < alpha) is True
        assert (global_p >= alpha) is hidden, k


def test_an_exposure_violation_rejects_even_when_the_global_does_not() -> None:
    receipt = _receipt(
        proportional_hazards_p_value=0.08,
        proportional_hazards_exposure_p_value=0.01,
        proportional_hazards_status="violation_block_paper_authorization",
        paper_authorization_allowed=False,
    )
    assert receipt.proportional_hazards_decision_rule == "exposure_or_global"
    assert receipt.paper_authorization_allowed is False


def test_the_global_alone_still_rejects() -> None:
    receipt = _receipt(
        proportional_hazards_p_value=0.01,
        proportional_hazards_exposure_p_value=0.90,
        proportional_hazards_status="violation_block_paper_authorization",
        paper_authorization_allowed=False,
    )
    assert receipt.paper_authorization_allowed is False


def test_an_exposure_violation_cannot_be_recorded_as_not_rejected() -> None:
    with pytest.raises(ValidationError, match="does not follow the declared policy"):
        _receipt(
            proportional_hazards_p_value=0.08,
            proportional_hazards_exposure_p_value=0.01,
            proportional_hazards_status="not_rejected",
            paper_authorization_allowed=True,
        )


# --- 2. no declared policy authorizes the paper past a violation -------------


def test_report_only_does_not_authorize_the_paper_after_a_violation() -> None:
    with pytest.raises(ValidationError, match="must be false whenever"):
        _receipt(
            proportional_hazards_p_value=0.01,
            proportional_hazards_exposure_p_value=0.01,
            proportional_hazards_policy="report_only",
            proportional_hazards_status="violation_report_only",
            paper_authorization_allowed=True,
        )


def test_report_only_is_still_recorded_as_the_declared_disclosure() -> None:
    """It is not silently rewritten to block: the reader sees what was asked."""

    receipt = _receipt(
        proportional_hazards_p_value=0.01,
        proportional_hazards_exposure_p_value=0.01,
        proportional_hazards_policy="report_only",
        proportional_hazards_status="violation_report_only",
        paper_authorization_allowed=False,
    )
    assert receipt.proportional_hazards_policy == "report_only"
    assert receipt.proportional_hazards_status == "violation_report_only"
    assert receipt.paper_authorization_allowed is False


def test_a_clean_diagnostic_still_authorizes() -> None:
    assert _receipt().paper_authorization_allowed is True


# --- 3. the vocabulary states only workflows that exist ----------------------


def test_human_review_is_not_a_declarable_ph_policy() -> None:
    with pytest.raises(ValidationError):
        _receipt(proportional_hazards_policy="human_review")


def test_human_review_is_not_a_recordable_ph_status() -> None:
    with pytest.raises(ValidationError):
        _receipt(
            proportional_hazards_p_value=0.01,
            proportional_hazards_exposure_p_value=0.01,
            proportional_hazards_status="violation_human_review",
            paper_authorization_allowed=False,
        )


def test_the_planned_contract_rejects_human_review_too() -> None:
    from easyicu.research_agent.contracts.family_primary import (
        FamilyPrimaryResultRequirement,
    )

    with pytest.raises(ValidationError):
        FamilyPrimaryResultRequirement(
            analysis_family="survival",
            exposure_source="treatment",
            outcome="death",
            expected_result_product="table:survival_effect_estimates",
            estimator="cox_proportional_hazards",
            effect_scale="hazard_ratio",
            population="adults",
            uncertainty_method="wald_95ci",
            proportional_hazards_policy="human_review",
        )
