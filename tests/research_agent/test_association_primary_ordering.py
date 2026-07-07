"""The base association forest marks row 0 as the primary estimand; row 0 must
be the true primary exposure, not whatever coefficient a long per-coefficient
table happened to list first (false-pass audit #15/#16).
"""

from __future__ import annotations

import pandas as pd

from easyicu.research_agent.figure_skill import (
    _normalise_association_frame,
    _primary_match_rank,
    _order_primary_first,
)


def _assoc(variables, ors):
    return pd.DataFrame(
        {
            "variable": variables,
            "odds_ratio": ors,
            "ci_low": [o * 0.95 for o in ors],
            "ci_high": [o * 1.05 for o in ors],
        }
    )


def test_exposure_moved_to_row_zero():
    df = _assoc(
        ["age", "charlson", "vasopressor_use", "sex_M"], [1.02, 1.13, 3.04, 0.88]
    )
    out = _normalise_association_frame(df, primary_exposure="vasopressor use")
    assert out.iloc[0]["label"].lower().startswith("vasopressor")
    assert round(float(out.iloc[0]["estimate"]), 2) == 3.04


def test_no_exposure_match_preserves_order():
    df = _assoc(["age", "charlson"], [1.02, 1.13])
    out = _normalise_association_frame(df, primary_exposure="vasopressor")
    assert out.iloc[0]["label"].lower() == "age"


def test_exact_match_beats_substring():
    # 'aki' exposure: exact 'aki' term must beat 'aki_stage_3' derived term
    df = _assoc(["aki_stage_3", "aki", "age"], [2.5, 1.9, 1.02])
    out = _normalise_association_frame(df, primary_exposure="aki")
    assert out.iloc[0]["label"].lower() == "aki"


def test_main_effect_beats_interaction_by_length():
    # no exact match; the closest-length substring (main effect) should win over
    # a longer interaction term
    r_main = _primary_match_rank("vasopressor", "vasopressoruse")
    r_inter = _primary_match_rank("vasopressor_x_age", "vasopressoruse")
    assert r_main < r_inter


def test_order_primary_first_is_stable_for_ties():
    df = pd.DataFrame(
        {
            "label": ["age", "charlson", "sex"],
            "estimate": [1.0, 1.1, 0.9],
            "lower": [0.9, 1.0, 0.8],
            "upper": [1.1, 1.2, 1.0],
        }
    )
    # token that matches nothing -> unchanged order
    out = _order_primary_first(df, "vasopressor")
    assert list(out["label"]) == ["age", "charlson", "sex"]
