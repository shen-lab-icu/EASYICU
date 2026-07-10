"""Tests for multiple-testing correction (O22)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# BH / Bonferroni unit tests
# ---------------------------------------------------------------------------


def test_bh_matches_textbook_example(ra):
    # Benjamini-Hochberg hand-worked example: sorted p-values
    # [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    # BH-adjusted (monotone): expected result also monotone nondecreasing.
    from easyicu.research_agent.methods.multiple_testing import _benjamini_hochberg

    pvals = [0.216, 0.042, 0.001, 0.041, 0.039, 0.060, 0.074, 0.008, 0.205, 0.212]
    adj = _benjamini_hochberg(pvals)
    # Adjusted values paired with originals should be non-decreasing when
    # sorted by the raw p value.
    pairs = sorted(zip(pvals, adj), key=lambda x: x[0])
    sorted_adj = [q for _, q in pairs]
    for prev, cur in zip(sorted_adj, sorted_adj[1:]):
        assert cur + 1e-12 >= prev
    # Smallest raw p=0.001 with n=10 ⇒ BH ≥ 0.001 * 10 / 1 = 0.01 (unless
    # a running min from above reduces it further, which it doesn't here).
    adj_sorted = [q for _, q in pairs]
    assert abs(adj_sorted[0] - 0.01) < 1e-12


def test_bh_cap_at_one(ra):
    from easyicu.research_agent.methods.multiple_testing import _benjamini_hochberg

    pvals = [0.9, 0.95, 0.99]
    adj = _benjamini_hochberg(pvals)
    assert all(q <= 1.0 for q in adj)


def test_bonferroni_is_exact(ra):
    from easyicu.research_agent.methods.multiple_testing import _bonferroni

    assert _bonferroni([0.01, 0.02, 0.1]) == pytest.approx([0.03, 0.06, 0.3])


def test_bh_empty(ra):
    from easyicu.research_agent.methods.multiple_testing import _benjamini_hochberg

    assert _benjamini_hochberg([]) == []


# ---------------------------------------------------------------------------
# P-value extraction
# ---------------------------------------------------------------------------


class _EvRec:
    """Minimal evidence-record stand-in for the extractor."""

    def __init__(self, *, evidence_id, kind, relative_path):
        self.evidence_id = evidence_id
        self.kind = kind
        self.relative_path = relative_path


def _write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_extract_pvalues_from_column(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "primary.csv"
    _write_csv(
        csv_path,
        rows=[
            ["sofa2", 1.23, 0.03],
            ["age", 1.05, 0.001],
            ["sex_M", 0.95, 0.9],
        ],
        header=["term", "or", "p_value"],
    )
    recs = [_EvRec(evidence_id="primary_association", kind="table", relative_path="primary.csv")]
    report = ra.build_multiple_testing_report(
        evidence_records=recs, run_dir=run_dir, alpha=0.05,
    )
    assert report.n_tests == 3
    assert all(0.0 <= r.p_value <= 1.0 for r in report.records)
    # raw: 2 <= 0.05, bh should still have 2 (n=3 is small and raw p=0.03
    # maps to BH = 0.03*3/2 = 0.045).
    summary = report.summary()
    assert summary["n_tests"] == 3
    assert summary["n_significant_raw"] == 2
    assert summary["n_significant_bh"] == 2
    assert summary["n_significant_bonferroni"] == 1  # only 0.001*3 <= 0.05


def test_extract_skips_out_of_range_pvalues(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "broken.csv"
    _write_csv(
        csv_path,
        rows=[
            ["ok", 0.01],
            ["nan_row", "NaN"],
            ["neg", -0.1],
            ["over", 1.5],
        ],
        header=["term", "pvalue"],
    )
    recs = [_EvRec(evidence_id="broken", kind="table", relative_path="broken.csv")]
    report = ra.build_multiple_testing_report(
        evidence_records=recs, run_dir=run_dir,
    )
    assert report.n_tests == 1
    assert report.records[0].p_value == 0.01


def test_extract_from_json_recurses(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    json_path = run_dir / "evidence" / "stat.json"
    json_path.write_text(
        json.dumps(
            {
                "primary": {"estimate": 1.2, "p_value": 0.004},
                "secondary": [
                    {"term": "age", "p_value": 0.33},
                    {"term": "sex", "p_value": 0.78},
                ],
            }
        )
    )
    recs = [_EvRec(evidence_id="stat", kind="statistic", relative_path="stat.json")]
    report = ra.build_multiple_testing_report(
        evidence_records=recs, run_dir=run_dir,
    )
    assert report.n_tests == 3
    pvals = sorted(r.p_value for r in report.records)
    assert pvals == pytest.approx([0.004, 0.33, 0.78])


def test_no_pvalues_produces_note(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "desc.csv"
    _write_csv(csv_path, rows=[["age", 65], ["n", 800]], header=["field", "value"])
    recs = [_EvRec(evidence_id="desc", kind="table", relative_path="desc.csv")]
    report = ra.build_multiple_testing_report(
        evidence_records=recs, run_dir=run_dir,
    )
    assert report.n_tests == 0
    assert any("No p-values" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_writes_multiple_testing_report_by_default(
    ra, synthetic_cohort, tmp_path
):
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
    assert (run_dir / "multiple_testing_report.csv").exists()
    assert (run_dir / "multiple_testing_report.md").exists()

    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = [r["evidence_id"] for r in manifest["evidence"]]
    assert "multiple_testing_report" in ev_ids
    assert "multiple_testing_summary" in ev_ids

    # At least one finding from the multiple_testing validator should be
    # emitted (info or warning) because the association-analysis skill runs a
    # logistic regression with a reported p-value.
    mt_findings = [f for f in manifest["findings"] if f["validator"] == "multiple_testing"]
    assert len(mt_findings) >= 1


def test_pipeline_multiple_testing_can_be_disabled(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_multiple_testing_correction=False,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert not (run_dir / "multiple_testing_report.csv").exists()
    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = [r["evidence_id"] for r in manifest["evidence"]]
    assert "multiple_testing_report" not in ev_ids
