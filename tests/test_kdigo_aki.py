import pandas as pd

from easyicu.kdigo_aki import _calculate_uo_rates_simple


def test_kdigo_uo_requires_minimum_documented_window_hours():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 60],
            "urine": [10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime")

    assert result["uo_rt_6hr"].isna().all()
    assert result["uo_rt_12hr"].isna().all()
    assert result["uo_rt_24hr"].isna().all()


def test_kdigo_uo_matches_mit_lcp_window_rules_once_six_hours_are_present():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime")

    assert result["uo_rt_6hr"].iloc[-1] == 0.1
    assert pd.isna(result["uo_rt_12hr"].iloc[-1])
    assert pd.isna(result["uo_rt_24hr"].iloc[-1])
