import pandas as pd
import pytest

from easyicu.kdigo_aki import _calculate_uo_rates_simple, kdigo_uo


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


def test_kdigo_uo_missing_patient_weight_leaves_rates_missing():
    urine = pd.DataFrame(
        {
            "stay_id": [2, 2, 2, 2, 2, 2],
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime")

    assert result["uo_rt_6hr"].isna().all()
    assert result["uo_rt_12hr"].isna().all()
    assert result["uo_rt_24hr"].isna().all()

    staged = kdigo_uo(urine, weight, "stay_id", "charttime")
    assert staged["uo_rt_6hr"].isna().all()
    assert staged["aki_stage_uo"].eq(0).all()


def test_kdigo_uo_invalid_weight_does_not_fall_back_to_70kg():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [0.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime")

    assert result["uo_rt_6hr"].isna().all()


def test_kdigo_uo_global_weight_without_id_applies_to_all_rows():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"weight": [50.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime")

    assert result["uo_rt_6hr"].iloc[-1] == pytest.approx(0.2)
