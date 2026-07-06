"""Regression: the survival KM table renderer recognises both canonical and
reconciled-reporting column names.

H1 01d root cause (2026-07-06): a downstream repair renderer failed with
"Could not identify time/survival columns in reporting_km_curve" because the
reconciled reporting table uses descriptive names (``time_hours_post_landmark``,
``survival_probability``) rather than the deterministic runner's canonical
``time``/``survival``. The robust SOURCE renderer already handled it via
substring matching; this locks explicit, exact-match recognition of both
families so a KM points table is never missed for a naming reason.

NOTE: this is defensive-in-depth on the source renderer. It does NOT by itself
fix 01d, which runs LLM-generated repair code that bypasses this renderer — that
architectural routing is left to the #47 session (see task log).
"""

from __future__ import annotations

import pandas as pd

from easyicu.research_agent.figures.survival import _km_groups_from_table


def _curve_points(groups):
    return {name: len(payload["time"]) for name, payload in groups}


def test_km_table_reconciled_reporting_columns():
    frame = pd.DataFrame(
        {
            "stratum": ["exposed"] * 3 + ["control"] * 3,
            "time_hours_post_landmark": [0.0, 24.0, 48.0, 0.0, 24.0, 48.0],
            "survival_probability": [1.0, 0.8, 0.6, 1.0, 0.9, 0.85],
            "ci_low": [1.0, 0.7, 0.5, 1.0, 0.85, 0.8],
            "ci_high": [1.0, 0.9, 0.7, 1.0, 0.95, 0.9],
        }
    )
    groups = _km_groups_from_table(frame)
    assert groups is not None
    assert _curve_points(groups) == {"exposed": 3, "control": 3}


def test_km_table_canonical_deterministic_columns():
    # The deterministic Cox runner emits time/survival/group/at_risk.
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "time": [0.0, 12.0, 0.0, 12.0],
            "survival": [1.0, 0.7, 1.0, 0.9],
            "at_risk": [100, 70, 100, 90],
        }
    )
    groups = _km_groups_from_table(frame)
    assert groups is not None
    assert _curve_points(groups) == {"a": 2, "b": 2}


def test_km_table_exact_name_wins_over_decoy_substring():
    # A decoy column containing "time" must not shadow the real points column.
    frame = pd.DataFrame(
        {
            "stratum": ["x", "x"],
            "enrollment_time_note": ["n/a", "n/a"],
            "time_hours_post_landmark": [0.0, 24.0],
            "survival_probability": [1.0, 0.75],
        }
    )
    groups = _km_groups_from_table(frame)
    assert groups is not None
    # 2 time points recovered -> the numeric points column was selected, not the note
    assert list(_curve_points(groups).values()) == [2]


def test_km_table_without_time_or_survival_returns_none():
    # Gate integrity: a table with neither a time nor a survival column is unrenderable.
    frame = pd.DataFrame({"stratum": ["a", "b"], "n_group": [10, 20]})
    assert _km_groups_from_table(frame) is None
