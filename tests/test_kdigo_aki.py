import pandas as pd
import pytest

import easyicu.scores.kdigo_aki as kdigo_aki
from easyicu.scores.kdigo_aki import (
    _calculate_uo_rates_simple,
    kdigo_stages,
    kdigo_uo,
)


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
    assert staged["aki_stage_uo"].isna().all()
    assert not staged["uo_assessable"].any()
    assert staged["uo_assessment_reason"].eq(
        "uo_window_or_weight_unavailable"
    ).all()


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


def test_kdigo_missing_baseline_is_unknown_not_stage_zero():
    creatinine = pd.DataFrame(
        {"stay_id": [1], "charttime": [0], "crea": [1.2]}
    )

    result = kdigo_stages(creatinine, id_col="stay_id", time_col="charttime")

    assert pd.isna(result["aki_stage_creat"].iloc[0])
    assert pd.isna(result["aki_stage_uo"].iloc[0])
    assert pd.isna(result["aki_stage"].iloc[0])
    assert pd.isna(result["aki"].iloc[0])
    assert result["uo_assessment_reason"].iloc[0] == "urine_or_weight_unavailable"


def test_kdigo_uo_calculation_error_is_unknown_not_stage_zero(monkeypatch):
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.1]}
    )
    urine = pd.DataFrame(
        {"stay_id": [1], "charttime": [0], "urine": [10.0]}
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [70.0]})

    def broken_uo(*args, **kwargs):
        raise RuntimeError("source decoding failed")

    monkeypatch.setattr(kdigo_aki, "kdigo_uo", broken_uo)
    result = kdigo_stages(
        creatinine,
        urine_df=urine,
        weight_df=weight,
        id_col="stay_id",
        time_col="charttime",
    )

    assert result["aki_stage_uo"].isna().all()
    assert not result["uo_assessable"].any()
    assert result["uo_assessment_reason"].eq("uo_calculation_error").all()


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


def test_kdigo_uo_refuses_unkeyed_weight_for_multiple_urine_entities():
    urine = pd.DataFrame(
        {
            "stay_id": [1] * 6 + [2] * 6,
            "charttime": list(range(0, 360, 60)) * 2,
            "urine": [10.0] * 12,
        }
    )
    # A different identifier namespace is not a join key.  Before this guard,
    # the first 50 kg row was broadcast to both stays.
    weight = pd.DataFrame({"patientid": [10, 20], "weight": [50.0, 100.0]})

    result = _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime")

    assert result["uo_rt_6hr"].isna().all()


def test_kdigo_hirid_direct_rate_uses_full_clock_time_coverage():
    urine = pd.DataFrame(
        {
            "patientid": [1, 1, 1],
            "datetime": [
                pd.Timedelta(hours=0),
                pd.Timedelta(hours=2),
                pd.Timedelta(hours=6),
            ],
            "urine": [50.0, 100.0, 50.0],
        }
    )
    weight = pd.DataFrame({"patientid": [1], "weight": [50.0]})

    result = kdigo_uo(
        urine,
        weight,
        id_col="patientid",
        time_col="datetime",
        source_is_rate=True,
        interval=pd.Timedelta(hours=1),
    )

    # Direct rates cover [-1,0], [0,2], and [2,6]. At t=6 the exact
    # six-hour window is therefore (100*2 + 50*4) / (50 kg * 6 h).
    assert result["uo_rt_6hr"].iloc[-1] == pytest.approx(
        400.0 / 50.0 / 6.0
    )
    assert result["uo_rt_12hr"].isna().all()
    assert result["uo_rt_24hr"].isna().all()


def test_kdigo_hirid_direct_rate_covers_only_observed_chart_span():
    urine = pd.DataFrame(
        {
            "patientid": [1, 1],
            "datetime": [
                pd.Timedelta(hours=0),
                pd.Timedelta(hours=10),
            ],
            "urine": [50.0, 500.0],
        }
    )
    weight = pd.DataFrame({"patientid": [1], "weight": [50.0]})

    result = kdigo_uo(
        urine,
        weight,
        id_col="patientid",
        time_col="datetime",
        source_is_rate=True,
        interval=pd.Timedelta(hours=1),
    )

    # The second recorded rate covers the preceding ten-hour chart interval:
    # the six-hour KDIGO window is evaluable, but no 12-hour span exists.
    assert result["uo_rt_6hr"].iloc[-1] == pytest.approx(10.0)
    assert result["uo_rt_12hr"].isna().all()
    assert result["uo_rt_24hr"].isna().all()
    assert pd.isna(result["aki_stage_uo"].iloc[0])
    assert result["aki_stage_uo"].iloc[1] == 0


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
    # Before a comparison baseline exists, the documented creatinine values do
    # not prove no AKI.  RRT still establishes stage 3 from its initiation.
    assert pd.isna(result["aki_stage"].iloc[0])
    assert result["aki_stage"].iloc[1] == 0
    assert result["aki_stage"].iloc[2] == 3
    assert pd.isna(result["aki"].iloc[0])
    assert bool(result["aki"].iloc[1]) is False
    assert bool(result["aki"].iloc[2]) is True


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
