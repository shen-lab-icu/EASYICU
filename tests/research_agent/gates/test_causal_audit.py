"""Tests for the causal-claim audit (O18)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EvRec:
    def __init__(
        self,
        *,
        evidence_id,
        kind,
        relative_path,
        description="",
        metadata=None,
    ):
        self.evidence_id = evidence_id
        self.kind = kind
        self.relative_path = relative_path
        self.description = description
        self.metadata = dict(metadata or {})


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def test_default_effect_is_associational(ra, tmp_path):
    recs = [
        _EvRec(
            evidence_id="primary_association",
            kind="table",
            relative_path="primary.csv",
            description="Logistic regression of outcome on sofa2.",
        )
    ]
    labels = ra.label_effects(evidence_records=recs, run_dir=tmp_path)
    assert len(labels) == 1
    assert labels[0].label == "associational"
    assert labels[0].estimand == "odds_ratio"


def test_explicit_iptw_with_all_supports_is_causal(ra, tmp_path):
    recs = [
        _EvRec(
            evidence_id="primary_association_iptw",
            kind="statistic",
            relative_path="iptw.json",
            description="IPTW risk difference for SOFA-2.",
            metadata={
                "identification_strategy": {
                    "method": "iptw",
                    "supporting_evidence_ids": [
                        "dag",
                        "positivity_diagnostic",
                        "negative_control",
                        "e_value",
                    ],
                }
            },
        ),
        _EvRec(evidence_id="dag", kind="log", relative_path="dag.svg"),
        _EvRec(
            evidence_id="positivity_diagnostic",
            kind="log",
            relative_path="positivity.log",
        ),
        _EvRec(
            evidence_id="negative_control",
            kind="statistic",
            relative_path="neg.csv",
        ),
        _EvRec(evidence_id="e_value", kind="statistic", relative_path="e.csv"),
    ]
    labels = ra.label_effects(evidence_records=recs, run_dir=tmp_path)
    effect = next(l for l in labels if l.evidence_id == "primary_association_iptw")
    assert effect.label == "causal_explicit"
    assert effect.identification_strategy == "iptw"
    assert effect.missing_supports == []


def test_explicit_with_missing_supports_is_overclaimed(ra, tmp_path):
    recs = [
        _EvRec(
            evidence_id="primary_association_tmle",
            kind="statistic",
            relative_path="tmle.json",
            description="TMLE ATE for SOFA-2",
            metadata={"identification_strategy": {"method": "tmle"}},
        )
    ]
    labels = ra.label_effects(evidence_records=recs, run_dir=tmp_path)
    assert len(labels) == 1
    assert labels[0].label == "causal_overclaimed"
    assert set(labels[0].missing_supports) >= {"dag", "positivity_diagnostic"}


def test_table_one_is_ignored(ra, tmp_path):
    recs = [
        _EvRec(
            evidence_id="table_one",
            kind="table",
            relative_path="table_one.csv",
            description="Cohort demographics.",
        )
    ]
    labels = ra.label_effects(evidence_records=recs, run_dir=tmp_path)
    assert labels == []


# ---------------------------------------------------------------------------
# Manuscript scan
# ---------------------------------------------------------------------------


def test_causal_language_over_associational_triggers_warning(ra):
    labels = [
        ra.EffectLabel(
            evidence_id="primary_association",
            artefact_path="primary.csv",
            estimand="odds_ratio",
            label="associational",
            rationale="default",
        )
    ]
    scaffold = (
        "Results. A higher SOFA-2 score was caused by sepsis "
        "{evidence:primary_association}."
    )
    hits = ra.scan_manuscript_for_causal_language(
        bound_manuscript=scaffold, effect_labels=labels,
    )
    assert len(hits) == 1
    assert hits[0].severity == "warning"
    assert "primary_association" in hits[0].linked_evidence_ids


def test_causal_language_over_overclaimed_triggers_error(ra):
    labels = [
        ra.EffectLabel(
            evidence_id="primary_association_tmle",
            artefact_path="tmle.json",
            estimand="risk_difference",
            label="causal_overclaimed",
            rationale="missing supports",
            identification_strategy="tmle",
            missing_supports=["dag"],
        )
    ]
    scaffold = (
        "SOFA-2 directly causes ICU mortality "
        "{evidence:primary_association_tmle}."
    )
    hits = ra.scan_manuscript_for_causal_language(
        bound_manuscript=scaffold, effect_labels=labels,
    )
    assert len(hits) == 1
    assert hits[0].severity == "error"


def test_causal_language_over_explicit_is_silent(ra):
    labels = [
        ra.EffectLabel(
            evidence_id="primary_association_iptw",
            artefact_path="iptw.json",
            estimand="risk_difference",
            label="causal_explicit",
            rationale="all supports present",
            identification_strategy="iptw",
        )
    ]
    scaffold = (
        "Under the IPTW specification, SOFA-2 increases ICU mortality by "
        "3.1 percentage points {evidence:primary_association_iptw}."
    )
    hits = ra.scan_manuscript_for_causal_language(
        bound_manuscript=scaffold, effect_labels=labels,
    )
    assert hits == []


def test_weak_patterns_without_evidence_id_are_ignored(ra):
    # Bare "improves" / "reduces" without any {evidence:} is noisy; should
    # not trigger.
    scaffold = "The protocol improves workflow."
    hits = ra.scan_manuscript_for_causal_language(
        bound_manuscript=scaffold, effect_labels=[]
    )
    assert hits == []


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_run_causal_audit_combines_labels_and_hits(ra, tmp_path):
    recs = [
        _EvRec(
            evidence_id="primary_association",
            kind="table",
            relative_path="primary.csv",
            description="Logistic regression",
        )
    ]
    scaffold = (
        "Higher SOFA-2 is attributable to admission illness severity "
        "{evidence:primary_association}."
    )
    report = ra.run_causal_audit(
        evidence_records=recs, run_dir=tmp_path, bound_manuscript=scaffold,
    )
    s = report.summary()
    assert s["n_effects_labelled"] == 1
    assert s["n_associational"] == 1
    assert s["n_language_warnings"] == 1


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_writes_causal_audit_by_default(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert (run_dir / "causal_audit_report.json").exists()
    assert (run_dir / "causal_audit_report.md").exists()

    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = [r["evidence_id"] for r in manifest["evidence"]]
    assert "causal_audit_report" in ev_ids
    assert "causal_audit_summary" in ev_ids
    # At least one causal_audit finding should be emitted
    findings = [f for f in manifest["findings"] if f["validator"] == "causal_audit"]
    assert len(findings) >= 1


def test_pipeline_causal_audit_can_be_disabled(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_causal_audit=False,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert not (run_dir / "causal_audit_report.json").exists()
