"""Critic consumes a bounded host semantic summary, not raw patient rows."""

from __future__ import annotations

from easyicu.research_agent.agents.core import CriticAgent
from easyicu.research_agent.review.step_semantics import (
    summarize_step_scientific_semantics,
)


def _step(ra):
    return ra.AnalysisStep(
        step_id="04_missingness",
        intent="Audit missingness and measurement semantics.",
        inputs=["death_time", "susp_inf_first"],
        expected_outputs=["table:missingness_measurement_audit"],
        method="missingness_measurement_audit",
    )


def _evidence(ra):
    return [ra.EvidenceRef(evidence_id="table_missingness")]


def _valid_summary():
    return {
        "status": "ok",
        "interpretation_class": "missingness_measurement_audit",
        "observation_semantics_audit": {
            "susp_inf_first": {
                "indicator_semantics": "binary_event_presence",
                "n_total": 5,
                "event_present_n": 3,
                "event_absent_n": 2,
                "invalid_pair_n": 0,
                "discordant_n": 0,
                "representative_invalid_n": 0,
                "positive_representative_missing_n": 0,
                "negative_representative_positive_n": 0,
            },
            "death_time": {
                "observation_semantics": "conditional_event_time",
                "n_total": 5,
                "eligible_event_n": 3,
                "not_applicable_event_absent_n": 2,
                "observed_event_time_n": 2,
                "missing_event_time_n": 1,
                "before_origin_n": 0,
                "contradictory_event_absent_with_time_n": 0,
            },
        },
        "temporal_validity_audit": {"status": "ok", "reason_codes": []},
    }


def test_valid_typed_missingness_summary_passes_critic(ra):
    summary = _valid_summary()

    compact = summarize_step_scientific_semantics(summary)
    critique = CriticAgent().review_step(
        step=_step(ra),
        step_summary=summary,
        evidence_refs=_evidence(ra),
        findings=[],
    )

    assert len(compact.metrics) == 2
    assert compact.issues == ()
    assert critique.status == "pass"


def test_missing_typed_semantics_receipt_blocks_missingness_step(ra):
    critique = CriticAgent().review_step(
        step=_step(ra),
        step_summary={
            "status": "ok",
            "interpretation_class": "missingness_measurement_audit",
            "worst_measured_concepts": [
                {"concept": "death_time", "value_missing_pct": 89.9}
            ],
        },
        evidence_refs=_evidence(ra),
        findings=[],
    )

    assert critique.status == "blocked"
    assert any(
        "observation_semantics_receipt_missing" in concern
        for concern in critique.concerns
    )


def test_negative_event_time_audit_passes_with_attributable_protocol_finding(ra):
    summary = _valid_summary()
    summary["observation_semantics_audit"]["death_time"]["before_origin_n"] = 28
    summary["temporal_validity_audit"] = {
        "status": "flagged_requires_downstream_protocol",
        "reason_codes": ["event_time_before_declared_origin:death_time:28"],
    }

    critique = CriticAgent().review_step(
        step=_step(ra),
        step_summary=summary,
        evidence_refs=_evidence(ra),
        findings=[],
    )

    assert critique.status == "pass"
    assert any(
        "event_time_before_origin_reported" in item for item in critique.concerns
    )
    assert critique.suggested_repairs == []


def test_unresolved_temporal_analysis_still_blocks_critic(ra):
    summary = _valid_summary()
    summary["temporal_validity_audit"] = {
        "status": "blocked",
        "reason_codes": ["event_time_before_declared_origin:death_time:28"],
    }

    critique = CriticAgent().review_step(
        step=_step(ra),
        step_summary=summary,
        evidence_refs=_evidence(ra),
        findings=[],
    )

    assert critique.status == "blocked"
    assert any("temporal_validity_blocked" in item for item in critique.concerns)


def test_unclosed_event_partition_blocks_critic(ra):
    summary = _valid_summary()
    summary["observation_semantics_audit"]["susp_inf_first"]["event_absent_n"] = 1

    critique = CriticAgent().review_step(
        step=_step(ra),
        step_summary=summary,
        evidence_refs=_evidence(ra),
        findings=[],
    )

    assert critique.status == "blocked"
    assert any("event_partition_not_closed" in item for item in critique.concerns)


def test_legacy_profile_label_is_needs_revision_even_with_files(ra):
    summary = {
        "status": "ok",
        "model_contracts": [
            {
                "fit_method": "statsmodels_Logit_MLE",
                "interval_method": "statsmodels_Logit_profile_normal",
            }
        ],
    }

    critique = CriticAgent().review_step(
        step=ra.AnalysisStep(
            step_id="05_model",
            intent="Fit the primary model.",
            expected_outputs=["table:model"],
            method="multivariable_logistic_regression",
        ),
        step_summary=summary,
        evidence_refs=[ra.EvidenceRef(evidence_id="table_model")],
        findings=[],
    )

    assert critique.status == "needs_revision"
    assert any(
        "confidence_interval_method_mislabeled" in concern
        for concern in critique.concerns
    )
