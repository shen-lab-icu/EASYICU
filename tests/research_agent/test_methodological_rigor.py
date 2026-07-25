"""Methodological-rigor audits.

These lock the layer that separates a research agent from a template
dispatcher: the analysis METHOD must match the study design. The headline case
is a survival question answered with a static odds ratio -- a figure renderer
would happily draw a "hazard ratio" forest from any table, but the method audit
must flag that no time-to-event estimator was used.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.review.methodological_rigor import (
    MethodSignals,
    MethodologicalRigorAuditor,
    audit_method_appropriateness,
    extract_method_signals,
)


def _severities(findings):
    return {f.severity for f in findings}


def _messages(findings):
    return " ".join(f.message.lower() for f in findings)


def test_survival_question_answered_with_odds_ratio_is_an_error():
    findings = audit_method_appropriateness(
        MethodSignals(
            family="time_to_event",
            has_odds_ratio=True,
            has_hazard_ratio=False,
            has_survival_curve=False,
            landmark_or_timezero_defined=True,
        )
    )
    assert "error" in _severities(findings)
    assert "hazard" in _messages(findings) or "time-to-event" in _messages(findings)


def test_survival_with_cox_and_landmark_has_no_method_mismatch():
    findings = audit_method_appropriateness(
        MethodSignals(
            family="time_to_event",
            has_hazard_ratio=True,
            has_survival_curve=True,
            landmark_or_timezero_defined=True,
        )
    )
    assert "error" not in _severities(findings)


def test_survival_without_landmark_warns_immortal_time():
    findings = audit_method_appropriateness(
        MethodSignals(
            family="time_to_event",
            has_hazard_ratio=True,
            has_survival_curve=True,
            landmark_or_timezero_defined=False,
        )
    )
    assert "immortal-time" in _messages(findings)


def test_prediction_without_calibration_warns():
    findings = audit_method_appropriateness(
        MethodSignals(
            family="prediction",
            has_auroc=True,
            has_calibration=False,
            held_out_reported=True,
        )
    )
    assert "calibration" in _messages(findings)
    assert "error" not in _severities(findings)


def test_causal_family_is_delegated_to_causal_audit_not_duplicated():
    # Causal rigor (balance/positivity/negative-control/E-value) is owned by
    # causal_audit.run_causal_audit; this module must NOT re-flag it and
    # double-fire against the wired causal auditor.
    findings = audit_method_appropriateness(
        MethodSignals(family="causal_emulation", has_covariate_balance=False)
    )
    assert findings == []


def test_phenotyping_clusters_without_stability_warns():
    findings = audit_method_appropriateness(
        MethodSignals(
            family="phenotyping",
            has_cluster_assignment=True,
            has_cluster_stability=False,
        )
    )
    assert "stability" in _messages(findings)


def test_complete_case_under_high_missingness_warns_any_family():
    findings = audit_method_appropriateness(
        MethodSignals(
            family="association",
            has_odds_ratio=True,
            complete_case_used=True,
            missing_fraction=0.46,
        )
    )
    assert "complete-case" in _messages(findings)


def test_complete_case_under_low_missingness_is_quiet():
    findings = audit_method_appropriateness(
        MethodSignals(
            family="association",
            has_odds_ratio=True,
            complete_case_used=True,
            missing_fraction=0.03,
        )
    )
    assert findings == []


def test_association_with_odds_ratio_is_appropriate():
    findings = audit_method_appropriateness(
        MethodSignals(family="association", has_odds_ratio=True)
    )
    assert findings == []


def test_extract_method_signals_reads_survival_evidence(ra, tmp_path: Path):
    evidence = ra.EvidenceStore(tmp_path)
    cox = tmp_path / "cox_summary.csv"
    cox.write_text("term,hr,lower,upper\nvent,1.8,1.5,2.2\n", encoding="utf-8")
    evidence.register_file(
        kind="table",
        description="Cox proportional-hazards summary",
        source_path=cox,
        evidence_id="cox_summary",
        producer="coder",
    )
    km = tmp_path / "km_curve.csv"
    km.write_text("group,time,survival\nvent,0,1.0\n", encoding="utf-8")
    evidence.register_file(
        kind="table",
        description="Kaplan-Meier curve points",
        source_path=km,
        evidence_id="km_curve",
        producer="coder",
    )
    context = ra.ResearchContext(
        research_question="Estimate the survival hazard of ventilation on mortality with a landmark at ICU admission.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[],
        primary_exposure="ventilation",
        target_outcome="death",
    )
    signals = extract_method_signals(context, evidence)
    assert signals.family == "time_to_event"
    assert signals.has_hazard_ratio is True
    assert signals.has_survival_curve is True
    assert signals.landmark_or_timezero_defined is True
    # A correct survival run raises no methodological-rigor finding.
    assert MethodologicalRigorAuditor().audit(context=context, evidence=evidence) == []


def test_method_signals_ignore_retired_evidence_in_current_authority_snapshot(
    ra,
    tmp_path: Path,
):
    evidence = ra.EvidenceStore(tmp_path)
    retired_path = tmp_path / "retired_cox.csv"
    retired_path.write_text("term,hr\nexposure,9.9\n", encoding="utf-8")
    retired = evidence.register_file(
        kind="table",
        description="Retired Cox proportional-hazards summary",
        source_path=retired_path,
        evidence_id="retired_cox",
        aliases=["cox_summary"],
        produced_by_step="03_model",
        producer="coder",
    )
    current_path = tmp_path / "current_descriptive.csv"
    current_path.write_text("group,n\nall,100\n", encoding="utf-8")
    current = evidence.register_file(
        kind="table",
        description="Current descriptive output",
        source_path=current_path,
        evidence_id="current_descriptive",
        produced_by_step="04_current",
        producer="coder",
    )
    step_records = [
        {
            "step_id": "03_model",
            "status": "ok",
            "evidence_ids": [retired.evidence_id],
        },
        {
            "step_id": "03_model",
            "status": "contract_failed",
            "evidence_ids": [],
        },
        {
            "step_id": "04_current",
            "status": "ok",
            "evidence_ids": [current.evidence_id],
        },
    ]
    current_records = evidence.current_verified_records(step_records)
    context = ra.ResearchContext(
        research_question=(
            "Estimate time-to-event mortality after exposure with a landmark "
            "at ICU admission."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )

    # The append-only legacy view can still see the retired semantic alias.
    assert extract_method_signals(context, evidence).has_hazard_ratio is True

    signals = extract_method_signals(
        context,
        evidence,
        evidence_records=current_records,
    )
    findings = MethodologicalRigorAuditor().audit(
        context=context,
        evidence=evidence,
        evidence_records=current_records,
    )

    assert signals.has_hazard_ratio is False
    assert "error" in _severities(findings)
