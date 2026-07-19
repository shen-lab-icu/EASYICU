"""`table == reality` plausibility gate over primary-result artefacts.

The value-level numeric auditor only checks manuscript-number == table-number.
These tests lock the complementary check: a physically-impossible PRIMARY table
(events > sample, a proportion outside [0,1], a non-positive ratio, an inverted
CI) fails closed — and, crucially, a healthy result and unrelated count dicts do
NOT trip it (case-neutral: no question-specific direction/threshold).
"""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.reporting.readiness import (
    primary_result_plausibility_errors,
)


def _summary(run_dir: Path, step_id: str, payload: dict) -> None:
    out = run_dir / "steps" / step_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_summary.json").write_text(json.dumps(payload))


def test_events_exceeding_sample_is_flagged(tmp_path: Path):
    # the fix3i failure mode: a column swap made "events" the sum of ages
    _summary(
        tmp_path,
        "01_survival_analysis",
        {
            "primary_model": {
                "estimate": 0.93,
                "p_value": 1e-25,
                "n": 73083,
                "events": 4635208,
            }
        },
    )
    errs = primary_result_plausibility_errors(tmp_path)
    assert errs, "events (4.6M) > n (73k) must be flagged"
    assert any("events" in e and "exceed" in e for e in errs)


def test_healthy_survival_result_is_clean(tmp_path: Path):
    _summary(
        tmp_path,
        "01_survival_analysis",
        {
            "analysis_family": "time_to_event",
            "hazard_ratio": 1.82,
            "hazard_ratio_ci_low": 1.74,
            "hazard_ratio_ci_high": 1.91,
            "n_events": 7063,
            "n_analysis": 74454,
            "median_followup_hours": 168.0,
        },
    )
    assert primary_result_plausibility_errors(tmp_path) == []


def test_event_rate_above_one_is_flagged(tmp_path: Path):
    _summary(tmp_path, "01_x", {"hazard_ratio": 1.2, "event_rate": 1.4})
    errs = primary_result_plausibility_errors(tmp_path)
    assert any("event rate" in e for e in errs)


def test_nonpositive_ratio_is_flagged(tmp_path: Path):
    _summary(tmp_path, "01_x", {"odds_ratio": -0.3, "p_value": 0.01})
    errs = primary_result_plausibility_errors(tmp_path)
    assert any("ratio estimate" in e for e in errs)


def test_inverted_ci_is_flagged(tmp_path: Path):
    _summary(tmp_path, "01_x", {"hazard_ratio": 1.1, "ci_low": 1.5, "ci_high": 1.2})
    errs = primary_result_plausibility_errors(tmp_path)
    assert any("inverted confidence interval" in e for e in errs)


def test_unrelated_count_dict_without_result_markers_is_not_flagged(tmp_path: Path):
    # a cohort-accounting dict may carry both an events-like and n-like key but
    # is NOT a model-result row; the events<=n check must not fire here.
    _summary(
        tmp_path,
        "01_cohort",
        {
            "cohort_counts": {"n_events": 900, "n_stays": 500}
        },  # no estimate/p_value marker
    )
    assert primary_result_plausibility_errors(tmp_path) == []


def test_result_csv_row_is_scanned(tmp_path: Path):
    out = tmp_path / "steps" / "01_survival_analysis" / "outputs"
    out.mkdir(parents=True)
    (out / "cox_summary.csv").write_text(
        "variable,hazard_ratio,p_value,n_model,n_events_model\n"
        "vent_24h_any,0.93,1e-25,73083,4635208\n"
    )
    errs = primary_result_plausibility_errors(tmp_path)
    assert any("cox_summary.csv" in e and "exceed" in e for e in errs)


def test_missing_steps_dir_returns_empty(tmp_path: Path):
    assert primary_result_plausibility_errors(tmp_path) == []
