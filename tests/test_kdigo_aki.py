import pandas as pd
import pytest

from easyicu.kdigo_aki import _calculate_uo_rates_simple, kdigo_stages, kdigo_uo


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


def test_kdigo_uo_large_minute_offsets_are_not_misclassified_as_seconds():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "charttime": [60000, 60060, 60120, 60180, 60240, 60300],
            "urine": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime")

    assert result["uo_rt_6hr"].iloc[-1] == pytest.approx(0.1)


def test_kdigo_uo_hour_numeric_time_axis_is_supported():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 4, 5],
            "urine": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "time")

    assert result["uo_rt_6hr"].iloc[-1] == pytest.approx(0.1)


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


def test_kdigo_rrt_stage_applies_from_first_active_rrt_time():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0, 120, 240],
            "crea": [1.0, 1.1, 1.2],
        }
    )
    rrt = pd.DataFrame({"stay_id": [1], "charttime": [180], "rrt": [1]})

    result = kdigo_stages(creatinine, rrt_df=rrt, id_col="stay_id", time_col="charttime")

    assert result["aki_stage_rrt"].tolist() == [0, 0, 3]
    assert result["aki_stage"].tolist() == [0, 0, 3]
    assert result["aki"].tolist() == [False, False, True]


def test_kdigo_rrt_exact_timestamp_match_is_not_required():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [100, 200],
            "crea": [1.0, 1.1],
        }
    )
    rrt = pd.DataFrame({"stay_id": [1], "charttime": [150], "rrt": [True]})

    result = kdigo_stages(creatinine, rrt_df=rrt, id_col="stay_id", time_col="charttime")

    assert result.loc[result["charttime"] == 200, "aki_stage_rrt"].item() == 3
