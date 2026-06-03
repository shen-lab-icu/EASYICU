"""Tests for reporting-guideline checklists (O16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class _EvRec:
    def __init__(self, evidence_id):
        self.evidence_id = evidence_id


def _recs(*ids):
    return [_EvRec(i) for i in ids]


# ---------------------------------------------------------------------------
# STROBE
# ---------------------------------------------------------------------------


def test_strobe_empty_inputs_are_mostly_open(ra):
    report = ra.build_strobe_checklist(
        evidence_records=[], bound_manuscript="",
    )
    assert report.name == "STROBE"
    # All items instantiated.
    assert len(report.items) == 22
    # With no evidence and no manuscript, nothing should be addressed.
    assert all(i.status in {"open", "partial"} for i in report.items)
    summary = report.summary()
    assert summary["n_addressed"] == 0


def test_strobe_common_ids_fill_key_methods_items(ra):
    recs = _recs(
        "analysis_plan",
        "research_context",
        "table_one",
        "missingness",
        "outcome_rate",
        "primary_association",
        "multiple_testing_report",
        "causal_audit_report",
        "manuscript_scaffold_bound",
        "literature_bundle",
        "hypothesis_blueprint",
        "cohort_audit",
    )
    scaffold = (
        "This retrospective cohort study analysed ICU patients. "
        "The study was supported by grant 123."
    )
    report = ra.build_strobe_checklist(
        evidence_records=recs, bound_manuscript=scaffold,
    )
    by_id = {i.item_id: i for i in report.items}
    # A cross-section of items should now be addressed.
    assert by_id["3"].status == "addressed"   # objectives/hypotheses
    assert by_id["4"].status == "addressed"   # design elements
    assert by_id["6"].status == "addressed"   # eligibility
    assert by_id["9"].status == "addressed"   # sources of bias
    assert by_id["12e"].status == "addressed" # sensitivity
    assert by_id["15"].status == "addressed"  # outcome events
    assert by_id["16"].status == "addressed"  # primary association
    # Funding keyword picked up.
    assert by_id["22"].status == "addressed"
    # "retrospective cohort" keyword satisfies item 1a.
    assert by_id["1a"].status == "addressed"


def test_strobe_partial_when_only_some_requirements_present(ra):
    # Item 1a has both required evidence (manuscript_scaffold_bound) and
    # required keywords (cohort/observational/retrospective/case-control).
    # With only the evidence present, status should be "partial".
    recs = _recs("manuscript_scaffold_bound")
    report = ra.build_strobe_checklist(
        evidence_records=recs, bound_manuscript=""
    )
    by_id = {i.item_id: i for i in report.items}
    assert by_id["1a"].status == "partial"


# ---------------------------------------------------------------------------
# TRIPOD+AI
# ---------------------------------------------------------------------------


def test_tripod_ai_has_27ish_items_and_coverage_grows_with_evidence(ra):
    blank = ra.build_tripod_ai_checklist(evidence_records=[], bound_manuscript="")
    assert blank.name == "TRIPOD+AI"
    assert len(blank.items) >= 25
    assert blank.summary()["coverage"] < 0.2

    full = ra.build_tripod_ai_checklist(
        evidence_records=_recs(
            "manuscript_scaffold_bound",
            "literature_bundle",
            "hypothesis_blueprint",
            "analysis_plan",
            "research_context",
            "table_one",
            "outcome_rate",
            "missingness",
            "model_performance",
            "cross_database_summary",
            "multiple_testing_report",
            "causal_audit_report",
            "reproducibility_envelope",
        ),
        bound_manuscript=(
            "Prediction model with AUROC and calibration. Subgroup by age "
            "and sex. Funding: institutional grant. Registered on ClinicalTrials.gov."
        ),
    )
    assert full.summary()["coverage"] > blank.summary()["coverage"]


def test_choose_checklist_selects_tripod_for_prediction_family(ra):
    assert "tripod_ai" in ra.choose_checklist("prediction_study")
    assert "tripod_ai" in ra.choose_checklist("validation_study")
    # Generic / association should get STROBE only.
    assert ra.choose_checklist("association_study") == ("strobe",)
    assert ra.choose_checklist(None) == ("strobe",)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_markdown_includes_coverage_line(ra):
    report = ra.build_strobe_checklist(
        evidence_records=_recs("table_one"), bound_manuscript=""
    )
    md = report.to_markdown()
    assert "Coverage:" in md
    assert "| Item | Section | Statement | Status | Evidence |" in md


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_writes_strobe_by_default(ra, synthetic_cohort, tmp_path):
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
    assert (run_dir / "reporting_checklist_strobe.md").exists()
    assert (run_dir / "reporting_checklist_strobe.json").exists()
    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = {r["evidence_id"] for r in manifest["evidence"]}
    assert "reporting_checklist_strobe" in ev_ids
    assert "reporting_checklist_strobe_json" in ev_ids
    # Finding emitted
    findings = [
        f for f in manifest["findings"] if f["validator"] == "reporting_checklist"
    ]
    assert len(findings) >= 1


def test_pipeline_checklist_can_be_disabled(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_reporting_checklist=False,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert not (run_dir / "reporting_checklist_strobe.md").exists()


def test_pipeline_explicit_override_selects_tripod_too(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        reporting_checklist_names=["strobe", "tripod_ai"],
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert (run_dir / "reporting_checklist_strobe.md").exists()
    assert (run_dir / "reporting_checklist_tripod_ai.md").exists()
