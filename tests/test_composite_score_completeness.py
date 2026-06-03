"""Pre-analysis, outcome-blind completeness QC for composite scores.

Locks two invariants:
  1. ``sofa2_score`` emits an outcome-blind ``sofa2_n_components`` count without
     changing the standard (MIMIC-IV/ricu-faithful) score value.
  2. ``composite_score_completeness`` is GENERIC — it works on any composite
     score given its components, not just SOFA, and never looks at an outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.sofa2 import sofa2_score
from easyicu.data_quality import composite_score_completeness

_COMPS = [
    "sofa2_resp", "sofa2_coag", "sofa2_liver",
    "sofa2_cardio", "sofa2_cns", "sofa2_renal",
]


def _component_dict(values_by_comp):
    return {
        c: pd.DataFrame({"stay_id": [1, 2, 3], c: values_by_comp[c]})
        for c in _COMPS
    }


def test_sofa2_score_emits_outcome_blind_component_count():
    vals = {
        "sofa2_resp":   [2, 1, np.nan],
        "sofa2_coag":   [0, np.nan, np.nan],
        "sofa2_liver":  [1, 0, np.nan],
        "sofa2_cardio": [0, np.nan, np.nan],
        "sofa2_cns":    [0, 2, np.nan],
        "sofa2_renal":  [0, np.nan, np.nan],
    }
    out = sofa2_score(_component_dict(vals)).set_index("stay_id")
    # completeness count (0-6), outcome-blind
    assert out.loc[1, "sofa2_n_components"] == 6
    assert out.loc[2, "sofa2_n_components"] == 3
    assert out.loc[3, "sofa2_n_components"] == 0
    # score value unchanged: missing -> 0 (standard), so the all-missing stay
    # collapses to 0 and is indistinguishable from a true zero by score alone
    assert out.loc[1, "sofa2"] == 3
    assert out.loc[2, "sofa2"] == 3
    assert out.loc[3, "sofa2"] == 0


def test_composite_score_completeness_is_generic_non_sofa():
    # A wholly made-up score with four components — no SOFA anywhere.
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "frailty_index": [0, 0, 3, 0],
        "c1": [1, np.nan, 2, np.nan],
        "c2": [0, np.nan, 1, np.nan],
        "c3": [0, np.nan, 0, np.nan],
        "c4": [0, np.nan, 1, np.nan],
    })
    rep = composite_score_completeness(df, "frailty_index", ["c1", "c2", "c3", "c4"])
    assert rep["n_total_components"] == 4
    assert rep["min_components"] == 4
    assert rep["n_low_completeness"] == 2
    assert rep["frac_low_completeness"] == 0.5
    assert rep["n_complete_components"] == 2
    assert rep["frac_complete_components"] == 0.5
    # outcome-blind: report carries no outcome/label key
    assert not any("death" in k or "mortality" in k or "outcome" in k for k in rep)


def test_composite_score_completeness_accepts_precomputed_count():
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sofa2": [0, 0, 5],
        "sofa2_n_components": [6, 0, 6],
    })
    rep = composite_score_completeness(
        df, "sofa2", n_components_col="sofa2_n_components"
    )
    assert rep["n_low_completeness"] == 1
    assert rep["frac_low_completeness"] == 1 / 3
    assert rep["n_complete_components"] == 2
