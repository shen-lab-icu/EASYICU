"""Sepsis SI window tests.

Contract under test: when ``index_col`` is numeric, ``susp_inf`` interprets
those values as **hours since ICU admission** — the post
``concept._align_time_to_admission`` invariant used everywhere else (notably
``sepsis_sofa2.sep3_sofa2``).  A 24h ABX window therefore means
``|abx_t - samp_t| <= 24`` when the times are numeric.
"""

import pandas as pd

from easyicu.sepsis import susp_inf


def test_susp_inf_numeric_hour_offsets_match_within_abx_window():
    # samp 10h after abx → inside 24h abx_win
    abx = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "abx": [1]})
    samp = pd.DataFrame({"stay_id": [1], "charttime": [10.0], "samp": [1]})

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result["susp_inf"].tolist() == [True]
    assert result["charttime"].tolist() == [0.0]


def test_susp_inf_numeric_hour_offsets_match_within_samp_window():
    # abx 50h after samp → inside 72h samp_win, outside 24h abx_win
    abx = pd.DataFrame({"stay_id": [1], "charttime": [50.0], "abx": [1]})
    samp = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "samp": [1]})

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result["susp_inf"].tolist() == [True]
    # SI time is the earlier of the pair (samp_time here)
    assert result["charttime"].tolist() == [0.0]


def test_susp_inf_numeric_hour_offsets_outside_both_windows():
    # samp 100h after abx → outside both 24h abx_win and 72h samp_win
    abx = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "abx": [1]})
    samp = pd.DataFrame({"stay_id": [1], "charttime": [100.0], "samp": [1]})

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result.empty


def test_susp_inf_datetime_offsets_use_timedelta_windows():
    # The datetime path is unaffected by the numeric refactor; sanity check.
    abx = pd.DataFrame(
        {"stay_id": [1], "charttime": [pd.Timestamp("2024-01-01 00:00")], "abx": [1]}
    )
    samp = pd.DataFrame(
        {"stay_id": [1], "charttime": [pd.Timestamp("2024-01-01 10:00")], "samp": [1]}
    )

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result["susp_inf"].tolist() == [True]
