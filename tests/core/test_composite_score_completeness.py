"""Pre-analysis, outcome-blind completeness QC for composite scores.

Locks two invariants:
  1. ``sofa2_score`` emits explicit observed/available counts; the deprecated
     ``sofa2_n_components`` alias remains equal to available.
  2. ``composite_score_completeness`` is GENERIC — it works on any composite
     score given its components, not just SOFA, and never looks at an outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.scores.sofa2 import sofa2_score
from easyicu.scores import sirs_score, qsofa_score, news_score, mews_score
from easyicu.io.data_quality import composite_score_completeness


def _ts(name, values, ids=(1, 2, 3), t="2020-01-01"):
    """Single-timepoint per-patient input frame for an early-warning score."""
    return pd.DataFrame({
        "stay_id": list(ids),
        "charttime": [pd.Timestamp(t)] * len(ids),
        name: list(values),
    })

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
    for column in (
        "sofa2_n_observed_components",
        "sofa2_n_available_components",
        "sofa2_n_components",
    ):
        assert out.loc[1, column] == 6
        assert out.loc[2, column] == 3
        assert out.loc[3, column] == 0
    # The component count remains available for outcome-blind QC, while the
    # aggregate score fails closed unless all six components are available.
    assert out.loc[1, "sofa2"] == 3
    assert pd.isna(out.loc[2, "sofa2"])
    assert pd.isna(out.loc[3, "sofa2"])


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


# --- Early-warning scores: outcome-blind completeness signal (SIRS/qSOFA/NEWS/MEWS).
# Each score coalesces an unmeasured component to 0, so an all-missing row scores 0
# just like a fully-measured negative; the *_n_components column tells them apart
# without changing the score. Per-patient single timepoint keeps LOCF/windowing trivial.

def test_sirs_score_emits_outcome_blind_component_count():
    out = sirs_score(
        temp=_ts("temp", [37, 39, np.nan]),
        hr=_ts("hr", [100, np.nan, np.nan]),
        resp=_ts("resp", [25, np.nan, np.nan]),
        wbc=_ts("wbc", [15, np.nan, np.nan]),
        pco2=_ts("pco2", [30, np.nan, np.nan]),
        bnd=_ts("bnd", [12, np.nan, np.nan]),
        id_cols=["stay_id"], index_col="charttime",
    ).set_index("stay_id")
    # 4 components: temp, hr, (resp|pco2), (wbc|bnd)
    assert out.loc[1, "sirs_n_components"] == 4
    assert out.loc[2, "sirs_n_components"] == 1
    assert out.loc[3, "sirs_n_components"] == 0
    # score unchanged: missing -> 0
    assert out.loc[1, "sirs"] == 3   # hr>90, resp>20, wbc>12
    assert out.loc[2, "sirs"] == 1   # temp 39 only
    assert out.loc[3, "sirs"] == 0


def test_qsofa_score_emits_outcome_blind_component_count():
    out = qsofa_score(
        gcs=_ts("gcs", [10, 15, np.nan]),
        sbp=_ts("sbp", [90, np.nan, np.nan]),
        resp=_ts("resp", [25, np.nan, np.nan]),
        id_cols=["stay_id"], index_col="charttime",
    ).set_index("stay_id")
    assert out.loc[1, "qsofa_n_components"] == 3
    assert out.loc[2, "qsofa_n_components"] == 1   # gcs measured (but 15 -> 0pt)
    assert out.loc[3, "qsofa_n_components"] == 0
    assert out.loc[1, "qsofa"] == 3
    assert out.loc[2, "qsofa"] == 0
    assert out.loc[3, "qsofa"] == 0


def test_news_score_completeness_excludes_defaulted_supp_o2():
    out = news_score(
        hr=_ts("hr", [80, np.nan, np.nan]),
        avpu=_ts("avpu", ["A", np.nan, np.nan]),
        supp_o2=_ts("supp_o2", [True, np.nan, np.nan]),
        o2sat=_ts("o2sat", [98, np.nan, np.nan]),
        temp=_ts("temp", [37, 36.5, np.nan]),
        sbp=_ts("sbp", [120, np.nan, np.nan]),
        resp=_ts("resp", [18, np.nan, np.nan]),
        id_cols=["stay_id"], index_col="charttime",
    ).set_index("stay_id")
    # 6 measured physiological components; supp_o2 excluded (defaults to False)
    assert out.loc[1, "news_n_components"] == 6
    assert out.loc[2, "news_n_components"] == 1   # temp only
    assert out.loc[3, "news_n_components"] == 0
    # all-missing row scores 0 like a fully-measured normal patient
    assert out.loc[3, "news"] == 0


def test_mews_score_emits_outcome_blind_component_count():
    out = mews_score(
        hr=_ts("hr", [80, np.nan, np.nan]),
        avpu=_ts("avpu", ["A", np.nan, np.nan]),
        temp=_ts("temp", [37, 36.5, np.nan]),
        sbp=_ts("sbp", [120, np.nan, np.nan]),
        resp=_ts("resp", [18, np.nan, np.nan]),
        id_cols=["stay_id"], index_col="charttime",
    ).set_index("stay_id")
    assert out.loc[1, "mews_n_components"] == 5
    assert out.loc[2, "mews_n_components"] == 1   # temp only
    assert out.loc[3, "mews_n_components"] == 0
    assert out.loc[3, "mews"] == 0
