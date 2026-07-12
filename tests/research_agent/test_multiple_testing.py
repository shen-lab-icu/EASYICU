"""Tests for multiple-testing correction (O22)."""

from __future__ import annotations

import csv
import hashlib
import inspect
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

    def __init__(
        self,
        *,
        evidence_id,
        kind,
        relative_path,
        source_path: Path,
        produced_by_step=None,
    ):
        self.evidence_id = evidence_id
        self.kind = kind
        self.relative_path = relative_path
        self.sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        self.produced_by_step = produced_by_step


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
            ["sofa2 vs reference", 1.23, 0.03],
            ["age-group comparison", 1.05, 0.001],
            ["sex comparison", 0.95, 0.9],
        ],
        header=["comparison", "statistic", "p_value"],
    )
    recs = [
        _EvRec(
            evidence_id="primary_association",
            kind="table",
            relative_path="primary.csv",
            source_path=csv_path,
        )
    ]
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
    assert summary["n_families"] == 1
    assert report.records[0].family_source == "source-local"


def test_pvalue_column_matching_is_exact_not_substring(ra):
    from easyicu.research_agent.methods.multiple_testing import _is_pvalue_column

    assert _is_pvalue_column("p_value") is True
    assert _is_pvalue_column("P-Value") is True
    assert _is_pvalue_column("raw_p") is True
    assert _is_pvalue_column("primary_p_value") is True
    assert _is_pvalue_column("group_value") is False
    assert _is_pvalue_column("p_value_bounded") is False
    assert _is_pvalue_column("adjusted_p") is False


def test_primary_p_value_exact_field_remains_backward_compatible(ra, tmp_path):
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    summary_path = evidence_dir / "step_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "primary_predictor": "exposure",
                "outcome": "death",
                "primary_p_value": 0.01,
                "p_value_bounded": False,
            }
        ),
        encoding="utf-8",
    )

    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="step_summary",
                kind="statistic",
                relative_path="step_summary.json",
                source_path=summary_path,
            )
        ],
        run_dir=run_dir,
    )

    assert report.n_tests == 1
    assert report.records[0].column == "primary_p_value"
    assert report.records[0].p_value == pytest.approx(0.01)


def test_group_value_and_pvalue_bounded_are_never_tests(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "trend.csv"
    _write_csv(
        csv_path,
        rows=[
            ["mortality", 0, True, 0.01],
            ["length of stay", 1, False, 0.20],
        ],
        header=["comparison", "group_value", "p_value_bounded", "p_value"],
    )
    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="trend",
                kind="table",
                relative_path="trend.csv",
                source_path=csv_path,
            )
        ],
        run_dir=run_dir,
    )

    assert report.n_tests == 2
    assert {record.column for record in report.records} == {"p_value"}


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
    recs = [
        _EvRec(
            evidence_id="broken",
            kind="table",
            relative_path="broken.csv",
            source_path=csv_path,
        )
    ]
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
                "primary": {
                    "test_id": "primary_test",
                    "outcome": "death",
                    "family_id": "planned_tests",
                    "estimate": 1.2,
                    "p_value": 0.004,
                },
                "secondary": [
                    {
                        "test_id": "age_test",
                        "outcome": "death",
                        "family_id": "planned_tests",
                        "p_value": 0.33,
                    },
                    {
                        "test_id": "sex_test",
                        "outcome": "death",
                        "family_id": "planned_tests",
                        "p_value": 0.78,
                    },
                ],
            }
        )
    )
    recs = [
        _EvRec(
            evidence_id="stat",
            kind="statistic",
            relative_path="stat.json",
            source_path=json_path,
        )
    ]
    report = ra.build_multiple_testing_report(
        evidence_records=recs, run_dir=run_dir,
    )
    assert report.n_tests == 3
    pvals = sorted(r.p_value for r in report.records)
    assert pvals == pytest.approx([0.004, 0.33, 0.78])


def test_structured_coefficients_exclude_nuisance_and_sensitivity_rows(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "coefficients.csv"
    _write_csv(
        csv_path,
        rows=[
            ["model_a", "const", "intercept", "primary", "planned_effects", 0.8],
            ["model_a", "age", "adjustment", "primary", "planned_effects", 0.04],
            ["model_a", "available", "availability", "primary", "planned_effects", 0.03],
            ["model_a", "offset", "nuisance", "primary", "planned_effects", 0.02],
            ["model_a", "exposure", "exposure", "primary", "planned_effects", 0.01],
            ["model_a", "interaction", "contrast", "primary", "planned_effects", 0.04],
            ["model_a", "exposure", "exposure", "sensitivity", "planned_effects", 0.005],
        ],
        header=[
            "model_id",
            "term",
            "term_role",
            "analysis_role",
            "hypothesis_family_id",
            "p_value",
        ],
    )
    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="coefficients",
                kind="table",
                relative_path="coefficients.csv",
                source_path=csv_path,
            )
        ],
        run_dir=run_dir,
    )

    assert report.n_tests == 2
    assert {record.label for record in report.records} == {"exposure", "interaction"}
    assert {record.family_id for record in report.records} == {"declared:planned_effects"}
    assert report.bh_adjusted == pytest.approx([0.02, 0.04])


def test_untyped_structured_coefficient_dump_is_omitted(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "coefficients.csv"
    _write_csv(
        csv_path,
        rows=[
            ["model_a", "exposure", "exposure", 1.4, 0.01],
            ["model_a", "age", "adjustment", 1.1, 0.02],
        ],
        header=["model_id", "term", "term_role", "estimate", "p_value"],
    )
    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="coefficients",
                kind="table",
                relative_path="coefficients.csv",
                source_path=csv_path,
            )
        ],
        run_dir=run_dir,
    )

    assert report.n_tests == 0
    assert any("structured coefficient table" in note for note in report.notes)


@pytest.mark.parametrize("term_column", ["variable", "predictor"])
def test_untyped_coefficient_alias_dump_is_omitted(ra, tmp_path, term_column):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "coefficients.csv"
    _write_csv(
        csv_path,
        rows=[
            ["model_a", "Intercept", 0.1, 0.9],
            ["model_a", "age", 0.2, 0.01],
            ["model_a", "exposure", 0.3, 0.02],
        ],
        header=["model_id", term_column, "coefficient", "p_value"],
    )

    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="coefficients",
                kind="table",
                relative_path="coefficients.csv",
                source_path=csv_path,
            )
        ],
        run_dir=run_dir,
    )

    assert report.n_tests == 0
    assert any("structured coefficient table" in note for note in report.notes)


def test_explicit_hypothesis_families_are_corrected_separately(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "planned_tests.csv"
    _write_csv(
        csv_path,
        rows=[
            ["a1", "family_a", 0.01],
            ["a2", "family_a", 0.04],
            ["b1", "family_b", 0.03],
        ],
        header=["comparison", "hypothesis_family_id", "p_value"],
    )
    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="planned_tests",
                kind="table",
                relative_path="planned_tests.csv",
                source_path=csv_path,
            )
        ],
        run_dir=run_dir,
    )

    adjusted = {
        record.label: value
        for record, value in zip(report.records, report.bh_adjusted)
    }
    assert report.summary()["n_families"] == 2
    assert adjusted == pytest.approx({"a1": 0.02, "a2": 0.04, "b1": 0.03})

    markdown_path = report.write_markdown(tmp_path / "multiple_testing.md")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "single run-wide family" not in markdown
    assert "independently within declared or source-local" in markdown


def test_explicit_family_id_is_authoritative_across_models(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "planned_tests.csv"
    _write_csv(
        csv_path,
        rows=[
            ["h1", "model_a", "planned_joint_family", 0.01],
            ["h2", "model_b", "planned_joint_family", 0.04],
        ],
        header=["hypothesis_id", "model_id", "hypothesis_family_id", "p_value"],
    )

    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="planned_tests",
                kind="table",
                relative_path="planned_tests.csv",
                source_path=csv_path,
            )
        ],
        run_dir=run_dir,
    )

    assert report.family_ids == ["declared:planned_joint_family"]
    assert report.bh_adjusted == pytest.approx([0.02, 0.04])


def test_csv_and_json_duplicate_statistics_are_counted_once(ra, tmp_path):
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    trend_path = evidence_dir / "trend.csv"
    _write_csv(
        trend_path,
        rows=[
            ["death", "cochran_armitage", "ordered_trends", 0.01],
            ["los_icu", "jonckheere_terpstra", "ordered_trends", 0.02],
        ],
        header=["outcome", "test_id", "family_id", "p_value"],
    )
    summary_path = evidence_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "trend_results": [
                    {
                        "outcome": "death",
                        "test_id": "cochran_armitage",
                        "family_id": "ordered_trends",
                        "p_value": 0.01,
                        "p_value_bounded": False,
                    },
                    {
                        "outcome": "los_icu",
                        "test_id": "jonckheere_terpstra",
                        "family_id": "ordered_trends",
                        "p_value": 0.02,
                        "p_value_bounded": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = ra.build_multiple_testing_report(
        evidence_records=[
            _EvRec(
                evidence_id="trend",
                kind="table",
                relative_path="trend.csv",
                source_path=trend_path,
            ),
            _EvRec(
                evidence_id="summary",
                kind="statistic",
                relative_path="summary.json",
                source_path=summary_path,
            ),
        ],
        run_dir=run_dir,
    )

    assert report.n_tests == 2
    assert report.bh_adjusted == pytest.approx([0.02, 0.02])
    assert any("Collapsed 2 duplicate" in note for note in report.notes)


def test_resume_report_is_not_reingested_and_duplicate_evidence_is_deduped(
    ra, tmp_path
):
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    tests_path = evidence_dir / "tests.csv"
    _write_csv(
        tests_path,
        rows=[["primary comparison", "primary_family", 0.01]],
        header=["comparison", "family_id", "p_value"],
    )
    prior_report_path = evidence_dir / "multiple_testing_report.csv"
    _write_csv(
        prior_report_path,
        rows=[["primary comparison", 0.01]],
        header=["label", "p_raw"],
    )
    source = _EvRec(
        evidence_id="tests",
        kind="table",
        relative_path="tests.csv",
        source_path=tests_path,
    )
    report = ra.build_multiple_testing_report(
        evidence_records=[
            source,
            source,
            _EvRec(
                evidence_id="multiple_testing_report",
                kind="statistic",
                relative_path="multiple_testing_report.csv",
                source_path=prior_report_path,
            ),
        ],
        run_dir=run_dir,
    )

    assert report.n_tests == 1
    assert report.records[0].p_value == 0.01
    assert any("Collapsed 1 duplicate" in note for note in report.notes)


def test_resume_uses_only_latest_step_evidence_and_versions_o22_report(ra, tmp_path):
    from easyicu.research_agent.pipeline_package import (
        _active_step_evidence_ids,
        _register_multiple_testing_outputs,
    )

    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    for name, p_value in (("old.csv", 0.01), ("new.csv", 0.03)):
        _write_csv(
            evidence_dir / name,
            rows=[["primary", "planned_family", p_value]],
            header=["hypothesis_id", "family_id", "p_value"],
        )
    records = [
        _EvRec(
            evidence_id="old_result",
            kind="table",
            relative_path="old.csv",
            source_path=evidence_dir / "old.csv",
            produced_by_step="06_model",
        ),
        _EvRec(
            evidence_id="new_result",
            kind="table",
            relative_path="new.csv",
            source_path=evidence_dir / "new.csv",
            produced_by_step="06_model",
        ),
    ]
    active_ids = _active_step_evidence_ids(
        [
            {
                "step_id": "06_model",
                "status": "ok",
                "evidence_ids": ["old_result"],
            },
            {
                "step_id": "06_model",
                "status": "ok",
                "evidence_ids": ["new_result"],
            },
        ]
    )
    report = ra.build_multiple_testing_report(
        evidence_records=records,
        run_dir=run_dir,
        active_evidence_ids=active_ids,
    )

    assert report.n_tests == 1
    assert report.records[0].evidence_id == "new_result"
    assert report.records[0].p_value == pytest.approx(0.03)

    store = ra.EvidenceStore(run_dir)
    csv_path = run_dir / "multiple_testing_report.csv"
    markdown_path = run_dir / "multiple_testing_report.md"
    report.write_csv(csv_path)
    report.write_markdown(markdown_path)
    first_csv_id, first_md_id = _register_multiple_testing_outputs(
        evidence=store,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    resumed_csv_id, resumed_md_id = _register_multiple_testing_outputs(
        evidence=store,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )

    assert first_csv_id == "multiple_testing_report"
    assert first_md_id == "multiple_testing_summary"
    assert resumed_csv_id == "multiple_testing_report_v2"
    assert resumed_md_id == "multiple_testing_summary_v2"
    resumed_csv_record = store.get(resumed_csv_id)
    resumed_md_record = store.get(resumed_md_id)
    assert resumed_csv_record is not None
    assert resumed_md_record is not None
    assert (run_dir / resumed_csv_record.relative_path).read_bytes() == csv_path.read_bytes()
    assert (run_dir / resumed_md_record.relative_path).read_bytes() == markdown_path.read_bytes()


def test_package_scopes_o22_evidence_to_its_own_current_producer(ra, tmp_path):
    from easyicu.research_agent import pipeline_package
    from easyicu.research_agent.runtime_artifacts import current_evidence_records

    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    for name, p_value in (("current.csv", 0.03), ("borrowed.csv", 0.001)):
        _write_csv(
            evidence_dir / name,
            rows=[[name, "planned_family", p_value]],
            header=["hypothesis_id", "family_id", "p_value"],
        )
    evidence_records = [
        _EvRec(
            evidence_id="current_result",
            kind="table",
            relative_path="current.csv",
            source_path=evidence_dir / "current.csv",
            produced_by_step="01_current",
        ),
        _EvRec(
            evidence_id="borrowed_result",
            kind="table",
            relative_path="borrowed.csv",
            source_path=evidence_dir / "borrowed.csv",
            produced_by_step="02_failed",
        ),
    ]
    per_step_records = [
        {
            "step_id": "01_current",
            "status": "ok",
            # Global id membership alone must not borrow 02_failed's output.
            "evidence_ids": ["current_result", "borrowed_result"],
        },
        {"step_id": "02_failed", "status": "contract_failed", "evidence_ids": []},
    ]

    report = ra.build_multiple_testing_report(
        evidence_records=current_evidence_records(
            evidence_records, per_step_records
        ),
        run_dir=run_dir,
        active_evidence_ids=pipeline_package._active_step_evidence_ids(
            per_step_records
        ),
    )

    assert [record.evidence_id for record in report.records] == ["current_result"]
    package_source = inspect.getsource(pipeline_package.finalise_success)
    assert "evidence_records=current_evidence_records(" in package_source


def test_no_pvalues_produces_note(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    csv_path = run_dir / "evidence" / "desc.csv"
    _write_csv(csv_path, rows=[["age", 65], ["n", 800]], header=["field", "value"])
    recs = [
        _EvRec(
            evidence_id="desc",
            kind="table",
            relative_path="desc.csv",
            source_path=csv_path,
        )
    ]
    report = ra.build_multiple_testing_report(
        evidence_records=recs, run_dir=run_dir,
    )
    assert report.n_tests == 0
    assert any("No p-values" in n for n in report.notes)


def test_pvalue_source_with_stale_digest_is_not_scanned(ra, tmp_path):
    run_dir = tmp_path / "run"
    source = run_dir / "evidence" / "tests.csv"
    _write_csv(source, rows=[["primary", 0.01]], header=["term", "p_value"])
    record = _EvRec(
        evidence_id="tests",
        kind="table",
        relative_path="tests.csv",
        source_path=source,
    )
    source.write_text("term,p_value\nprimary,0.99\n", encoding="utf-8")

    report = ra.build_multiple_testing_report(
        evidence_records=[record],
        run_dir=run_dir,
    )

    assert report.n_tests == 0


def test_pvalue_source_cannot_escape_evidence_directory(ra, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    _write_csv(outside, rows=[["primary", 0.01]], header=["term", "p_value"])
    record = _EvRec(
        evidence_id="outside",
        kind="table",
        relative_path="../../outside.csv",
        source_path=outside,
    )

    report = ra.build_multiple_testing_report(
        evidence_records=[record],
        run_dir=run_dir,
    )

    assert report.n_tests == 0


def test_pvalue_source_symlink_is_not_scanned(ra, tmp_path):
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    _write_csv(outside, rows=[["primary", 0.01]], header=["term", "p_value"])
    link = evidence_dir / "linked.csv"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    record = _EvRec(
        evidence_id="linked",
        kind="table",
        relative_path="linked.csv",
        source_path=outside,
    )

    report = ra.build_multiple_testing_report(
        evidence_records=[record],
        run_dir=run_dir,
    )

    assert report.n_tests == 0


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

    # O22 always records that the audit ran. If the mock emits only an untyped
    # coefficient dump, the finding truthfully reports zero defensibly scoped
    # tests instead of inventing a run-wide family.
    mt_findings = [f for f in manifest["findings"] if f["validator"] == "multiple_testing"]
    assert len(mt_findings) >= 1
    assert mt_findings[-1]["detail"]["n_tests"] >= 0


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
