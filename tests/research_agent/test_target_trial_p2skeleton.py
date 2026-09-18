"""Target-trial skeleton: denominator conservation, fail-closed time-zero, review scaffold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.methods.target_trial import (
    MANUAL_ASSUMPTIONS_TITLE,
    CloneCensorWeightChecklist,
    TargetTrialError,
    check_time_zero,
    clone_censor_weight_checklist,
    reconcile_eligibility,
    render_review_sheet,
)


def _eligibility_frame() -> pd.DataFrame:
    return pd.DataFrame({"stay_id": range(10)})


# ---------------------------------------------------------------------------
# Eligibility: denominator-chain conservation with hand-checked counts.
# ---------------------------------------------------------------------------


def test_eligibility_denominator_chain_conserves() -> None:
    inclusions = {
        "adult": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        "icu": [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    }
    exclusions = {"shock": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]}
    report = reconcile_eligibility(
        _eligibility_frame(), inclusions=inclusions, exclusions=exclusions
    )

    assert report.n_start == 10
    assert [s.name for s in report.steps] == ["adult", "icu", "shock"]
    assert [s.kind for s in report.steps] == [
        "inclusion",
        "inclusion",
        "exclusion",
    ]
    adult, icu, shock = report.steps
    assert (adult.n_before, adult.n_marginal, adult.n_meeting) == (10, 8, 8)
    assert (adult.n_removed, adult.n_remaining) == (2, 8)
    assert (icu.n_before, icu.n_marginal, icu.n_meeting) == (8, 9, 7)
    assert (icu.n_removed, icu.n_remaining) == (1, 7)
    assert (shock.n_before, shock.n_marginal, shock.n_meeting) == (7, 3, 2)
    assert (shock.n_removed, shock.n_remaining) == (2, 5)
    # Chain continuity: each step starts where the previous one ended.
    for prev, cur in zip(report.steps, report.steps[1:]):
        assert cur.n_before == prev.n_remaining
    # Conservation invariant: start - excluded == analysis.
    assert report.n_start - report.n_excluded_total == report.n_analysis == 5
    assert report.analysis_positions == (0, 1, 3, 4, 6)
    assert report.evidence_ceiling == "analysis_only"


def test_eligibility_fail_closed() -> None:
    frame = _eligibility_frame()
    with pytest.raises(TargetTrialError):  # empty analysis set
        reconcile_eligibility(frame, inclusions={"none": [0] * 10})
    with pytest.raises(TargetTrialError):  # NaN mask is not silently dropped
        reconcile_eligibility(frame, inclusions={"x": [1.0] * 9 + [float("nan")]})
    with pytest.raises(TargetTrialError):  # length mismatch
        reconcile_eligibility(frame, inclusions={"x": [1, 1, 1]})
    with pytest.raises(TargetTrialError):  # non-binary values
        reconcile_eligibility(frame, exclusions={"x": [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]})
    with pytest.raises(TargetTrialError):  # empty frame
        reconcile_eligibility(pd.DataFrame({"stay_id": []}))


# ---------------------------------------------------------------------------
# Time-zero alignment: clean pass plus every fail-closed violation.
# ---------------------------------------------------------------------------


def _time_zero_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t0": [0.0, 1.0, 2.0, 0.5],
            "event": [5.0, 6.0, 7.0, 8.0],
            "group": ["early", "deferred", "early", "deferred"],
            "treated_at": [1.0, float("nan"), 2.5, float("nan")],
        }
    )


def test_time_zero_clean_pass() -> None:
    report = check_time_zero(
        _time_zero_frame(),
        time_zero_col="t0",
        event_time_col="event",
        group_col="group",
        group_known_at_time_zero=[True, True, True, True],
        treatment_time_col="treated_at",
        grace_period_days=2.0,
    )
    assert report.n == 4
    assert report.n_analysis == 4
    assert report.grace_period_days == 2.0
    assert report.evidence_ceiling == "analysis_only"


def test_time_zero_violations_fail_closed() -> None:
    base = {"time_zero_col": "t0", "event_time_col": "event", "group_col": "group"}
    frame = _time_zero_frame()
    with pytest.raises(TargetTrialError, match="knowable"):
        check_time_zero(
            frame, **base, group_known_at_time_zero=[True, False, True, True]
        )
    early = frame.copy()
    early.loc[0, "event"] = -1.0
    with pytest.raises(TargetTrialError, match="precedes"):
        check_time_zero(
            early, **base, group_known_at_time_zero=[True] * 4,
        )
    late = frame.copy()
    late.loc[0, "treated_at"] = 30.0
    with pytest.raises(TargetTrialError, match="grace"):
        check_time_zero(
            late,
            **base,
            group_known_at_time_zero=[True] * 4,
            treatment_time_col="treated_at",
            grace_period_days=2.0,
        )
    missing_group = frame.copy()
    missing_group.loc[1, "group"] = None
    with pytest.raises(TargetTrialError, match="missing"):
        check_time_zero(
            missing_group, **base, group_known_at_time_zero=[True] * 4
        )
    with pytest.raises(TargetTrialError, match="grace"):
        check_time_zero(
            frame,
            **base,
            group_known_at_time_zero=[True] * 4,
            grace_period_days=-1.0,
        )


# ---------------------------------------------------------------------------
# Clone-censor-weight scaffold: review aid only, never a causal conclusion.
# ---------------------------------------------------------------------------


def _checklist() -> CloneCensorWeightChecklist:
    return clone_censor_weight_checklist(
        strategies=["early", "deferred"],
        grace_period_days=3.0,
        n_analysis=120,
        censoring_rule_summary="deviation from the assigned strategy censors at deviation",
        weight_model_spec="pooled logistic IPCW on baseline covariates, seed 0, no truncation",
    )


def test_checklist_labels_manual_assumptions_and_estimates_nothing() -> None:
    sheet = _checklist()
    assert sheet.manual_assumptions_title == "需人工确认的因果假设清单"
    assert MANUAL_ASSUMPTIONS_TITLE == "需人工确认的因果假设清单"
    assert len(sheet.manual_assumptions) >= 5
    joined = "\n".join(sheet.manual_assumptions)
    for keyword in ("可交换", "正性", "一致性", "删失"):
        assert keyword in joined
    # Scaffold, not estimator: no causal conclusion may escape.
    assert sheet.causal_effect_estimate is None
    assert sheet.n_clones_mechanical == 240
    assert sheet.evidence_ceiling == "analysis_only"
    assert len(sheet.machine_gaps) >= 1

    text = render_review_sheet(sheet)
    assert "需人工确认的因果假设清单" in text
    assert "analysis_only" in text
    assert "不下因果结论" in text

    # Deterministic: identical inputs give identical sheets.
    assert _checklist() == sheet


def test_checklist_rejects_bad_spec() -> None:
    with pytest.raises(TargetTrialError):
        clone_censor_weight_checklist(
            strategies=["only-one"],
            grace_period_days=3.0,
            n_analysis=10,
            censoring_rule_summary="rule",
            weight_model_spec="spec",
        )
    with pytest.raises(TargetTrialError):
        clone_censor_weight_checklist(
            strategies=["a", "b"],
            grace_period_days=3.0,
            n_analysis=0,
            censoring_rule_summary="rule",
            weight_model_spec="spec",
        )
    with pytest.raises(TargetTrialError):
        render_review_sheet(object())
