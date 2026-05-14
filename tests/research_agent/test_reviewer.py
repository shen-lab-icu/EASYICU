"""Tests for the three-role simulated reviewer round (O15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class _EvRec:
    def __init__(self, evidence_id):
        self.evidence_id = evidence_id


class _Finding:
    def __init__(self, validator, severity, message, detail=None):
        self.validator = validator
        self.severity = severity
        self.message = message
        self.detail = detail or {}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_clean_run_recommends_accept(ra):
    recs = [
        _EvRec(i)
        for i in (
            "primary_association",
            "missingness",
            "multiple_testing_report",
            "reporting_checklist_strobe",
            "reproducibility_envelope",
            "literature_bundle",
        )
    ]
    findings = [
        _Finding(
            "multiple_testing",
            "info",
            "Ran BH-FDR across 3 tests at alpha=0.050",
        ),
        _Finding(
            "reporting_checklist",
            "info",
            "STROBE coverage 80%",
            detail={"coverage": 0.8, "n_addressed": 18},
        ),
        _Finding("causal_audit", "info", "Labelled 1 effect(s)"),
    ]
    report = ra.run_reviewer_round(evidence_records=recs, findings=findings)
    # Every role accepts; no reject / major.
    assert report.aggregated_recommendation() == "accept"
    for c in report.critiques:
        assert c.recommendation() == "accept"


def test_missing_primary_estimate_triggers_statistician_major(ra):
    recs = [_EvRec("table_one")]
    report = ra.run_reviewer_round(evidence_records=recs, findings=[])
    stats = next(c for c in report.critiques if c.reviewer == "statistician")
    assert any(
        c.topic == "effect_estimate" and c.severity == "major" for c in stats.comments
    )


def test_causal_error_triggers_clinician_reject(ra):
    recs = [_EvRec("primary_association"), _EvRec("causal_audit_report")]
    findings = [
        _Finding(
            "causal_audit",
            "error",
            "Causal language over strong pattern caus cited [causal_overclaimed]",
        )
    ]
    report = ra.run_reviewer_round(evidence_records=recs, findings=findings)
    clin = next(c for c in report.critiques if c.reviewer == "clinician")
    assert clin.recommendation() == "reject"
    assert report.aggregated_recommendation() == "reject"


def test_low_checklist_coverage_triggers_methodologist_major(ra):
    recs = [_EvRec("primary_association"), _EvRec("reporting_checklist_strobe")]
    findings = [
        _Finding(
            "reporting_checklist",
            "info",
            "STROBE coverage 30%",
            detail={"coverage": 0.3, "n_addressed": 6},
        )
    ]
    report = ra.run_reviewer_round(evidence_records=recs, findings=findings)
    meth = next(c for c in report.critiques if c.reviewer == "methodologist")
    assert any(
        c.topic == "reporting_guideline" and c.severity == "major" for c in meth.comments
    )


def test_markdown_contains_per_role_header(ra):
    report = ra.run_reviewer_round(evidence_records=[], findings=[])
    md = report.to_markdown()
    assert "## Statistician" in md
    assert "## Clinician" in md
    assert "## Methodologist" in md


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_writes_reviewer_report_by_default(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
    )
    result = pipeline.run(
        skill="sofa_mortality",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert (run_dir / "reviewer_report.md").exists()
    assert (run_dir / "reviewer_report.json").exists()
    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = {r["evidence_id"] for r in manifest["evidence"]}
    assert "reviewer_report" in ev_ids
    assert "reviewer_report_json" in ev_ids
    findings = [f for f in manifest["findings"] if f["validator"] == "reviewer_round"]
    assert len(findings) == 1
    # Read the structured report and make sure three reviewers appear.
    payload = json.loads((run_dir / "reviewer_report.json").read_text())
    assert len(payload["critiques"]) == 3
    assert {c["reviewer"] for c in payload["critiques"]} == {
        "statistician",
        "clinician",
        "methodologist",
    }


def test_pipeline_reviewer_can_be_disabled(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_reviewer_round=False,
    )
    result = pipeline.run(
        skill="sofa_mortality",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert not (run_dir / "reviewer_report.md").exists()
