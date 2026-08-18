import pandas as pd
import pytest

import easyicu.scores.kdigo_aki as kdigo_aki
from easyicu.scores.kdigo_aki import (
    KDIGOComponentCalculationError,
    KDIGOComponentLoadError,
    KDIGOComponentSchemaError,
    _calc_aki_stage_creat,
    _calculate_uo_rates_simple,
    get_aki_incidence,
    kdigo_stages,
    kdigo_uo,
    load_kdigo_aki,
    summarize_aki,
)


@pytest.mark.clinical_conformance
def test_kdigo_stage3_absolute_threshold_uses_each_current_creatinine_value() -> None:
    """MIT-LCP KDIGO SQL: current >= 4 and >= 0.3 above 48-hour low."""
    stage = _calc_aki_stage_creat(
        pd.Series([4.5, 4.0, 4.0]),
        pd.Series([4.0, 3.7, 3.71]),
        pd.Series([4.0, 3.0, 3.0]),
    )

    assert stage.tolist() == [3, 3, 0]


def test_kdigo_uo_requires_minimum_documented_window_hours():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 60],
            "urine": [10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

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

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

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

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

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

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "time", time_unit="hours"
    )

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

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

    assert result["uo_rt_6hr"].isna().all()
    assert result["uo_rt_12hr"].isna().all()
    assert result["uo_rt_24hr"].isna().all()

    staged = kdigo_uo(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )
    assert staged["uo_rt_6hr"].isna().all()
    assert staged["aki_stage_uo"].isna().all()
    assert not staged["uo_assessable"].any()
    assert staged["uo_assessment_reason"].eq(
        "missing_weight"
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

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

    assert result["uo_rt_6hr"].isna().all()


@pytest.mark.parametrize("weights", ([50.0, 100.0], [100.0, 50.0]))
def test_kdigo_uo_conflicting_keyed_weights_fail_closed(weights):
    urine = pd.DataFrame(
        {
            "stay_id": [1] * 6,
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [30.0] * 6,
        }
    )
    weight = pd.DataFrame({"stay_id": [1, 1], "weight": weights})

    with pytest.raises(KDIGOComponentCalculationError) as caught:
        _calculate_uo_rates_simple(
            urine, weight, "stay_id", "charttime", time_unit="minutes"
        )

    assert caught.value.reason_code == "kdigo_weight_values_conflict"


def test_kdigo_rate_source_conflicting_keyed_weights_fail_closed():
    urine = pd.DataFrame(
        {
            "patientid": [1, 1, 1],
            "datetime": [
                pd.Timedelta(hours=0),
                pd.Timedelta(hours=2),
                pd.Timedelta(hours=6),
            ],
            "urine": [30.0, 30.0, 30.0],
        }
    )
    weight = pd.DataFrame(
        {"patientid": [1, 1], "weight": [50.0, 100.0]}
    )

    with pytest.raises(KDIGOComponentCalculationError) as caught:
        _calculate_uo_rates_simple(
            urine,
            weight,
            "patientid",
            "datetime",
            source_is_rate=True,
            interval=pd.Timedelta(hours=1),
        )

    assert caught.value.reason_code == "kdigo_weight_values_conflict"


def test_kdigo_uo_duplicate_identical_keyed_weights_are_accepted():
    urine = pd.DataFrame(
        {
            "stay_id": [1] * 6,
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [30.0] * 6,
        }
    )
    weight = pd.DataFrame({"stay_id": [1, 1], "weight": [100.0, 100.0]})

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

    assert result["uo_rt_6hr"].iloc[-1] == pytest.approx(0.3)


def test_empty_weight_source_is_missingness_not_a_calculation_crash():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 60],
            "urine": [10.0, 10.0],
        }
    )

    result = kdigo_stages(
        None,
        urine_df=urine,
        weight_df=pd.DataFrame(),
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result["urine_ascertainment"].eq("indeterminate").all()
    assert result["urine_ascertainment_reason"].eq("missing_weight").all()
    assert result["aki_ascertainment"].eq("indeterminate").all()


def test_kdigo_missing_baseline_is_unknown_not_stage_zero():
    creatinine = pd.DataFrame(
        {"stay_id": [1], "charttime": [0], "crea": [1.2]}
    )

    result = kdigo_stages(
        creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert pd.isna(result["aki_stage_creat"].iloc[0])
    assert pd.isna(result["aki_stage_uo"].iloc[0])
    assert pd.isna(result["aki_stage"].iloc[0])
    assert pd.isna(result["aki"].iloc[0])
    assert not bool(result["aki_assessable"].iloc[0])
    assert result["aki_ascertainment"].iloc[0] == "indeterminate"
    assert result["aki_assessment_reason"].iloc[0] == "indeterminate"
    assert result["uo_assessment_reason"].iloc[0] == "source_absent"
    assert result["creatinine_ascertainment_reason"].iloc[0] == (
        "insufficient_baseline"
    )
    assert pd.isna(result["aki_severe"].iloc[0])
    assert not bool(result["aki_severe_assessable"].iloc[0])
    assert result["aki_severe_ascertainment"].iloc[0] == "indeterminate"


def test_kdigo_uses_urine_timeline_when_creatinine_is_unavailable():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [10.0] * 6,
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = kdigo_stages(
        None,
        urine_df=urine,
        weight_df=weight,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert len(result) == len(urine)
    assert result["aki_stage_creat"].isna().all()
    assert result["aki_stage_uo"].iloc[-1] == 1
    assert bool(result["aki_assessable"].iloc[-1]) is True
    assert bool(result["aki"].iloc[-1]) is True


def test_kdigo_uses_rrt_timeline_when_no_laboratory_or_urine_exists():
    rrt = pd.DataFrame({"stay_id": [1], "charttime": [180], "rrt": [1]})

    result = kdigo_stages(
        None,
        rrt_df=rrt,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result[["stay_id", "charttime"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 180}
    ]
    assert result["aki_stage"].iloc[0] == 3
    assert bool(result["rrt_observed"].iloc[0]) is True
    assert bool(result["aki_assessable"].iloc[0]) is True


def test_load_kdigo_keeps_urine_only_patients_in_the_public_api():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "charttime": [0, 1, 2, 3, 4, 5],
            "urine": [10.0] * 6,
        }
    )
    result = load_kdigo_aki(
        "synthetic",
        verbose=False,
        preloaded_data={
            "crea": pd.DataFrame(),
            "urine": urine,
            "weight": pd.DataFrame({"stay_id": [1], "weight": [100.0]}),
            "rrt": pd.DataFrame(),
        },
    )

    assert not result.empty
    assert result["aki_stage_creat"].isna().all()
    assert result["aki_assessable"].any()


def test_summarize_aki_separates_definitive_partial_and_indeterminate_patients():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "aki": pd.Series([True, False, pd.NA, pd.NA], dtype="boolean"),
            "aki_stage": pd.Series([1, 0, 0, pd.NA], dtype="Int64"),
            "aki_ascertainment": [
                "positive",
                "negative_complete",
                "partial_no_observed_positive",
                "indeterminate",
            ],
        }
    )

    summary = summarize_aki(frame, id_col="stay_id")

    assert summary["aki_positive_patients"] == 1
    assert summary["aki_negative_complete_patients"] == 1
    assert summary["aki_partial_no_observed_positive_patients"] == 1
    assert summary["aki_indeterminate_patients"] == 1
    assert summary["n_definitive_phenotype_patients"] == 2
    assert summary["aki_prevalence_among_definitive_phenotypes"] == pytest.approx(0.5)
    assert summary["definitive_phenotype_coverage"] == pytest.approx(0.5)
    assert summary["aki_rate_denominator"] == "definitive_phenotype_patients"


def test_kdigo_uo_calculation_error_fails_closed_with_stable_reason(monkeypatch):
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
    with pytest.raises(KDIGOComponentCalculationError) as error:
        kdigo_stages(
            creatinine,
            urine_df=urine,
            weight_df=weight,
            id_col="stay_id",
            time_col="charttime",
            time_unit="minutes",
        )

    assert error.value.component == "urine_output"
    assert error.value.reason_code == "kdigo_urine_output_calculation_failed"


def test_nonnumeric_urine_is_not_silently_converted_to_zero_output():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 60],
            "urine": [10.0, "bad-source-value"],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [70.0]})

    with pytest.raises(KDIGOComponentSchemaError) as error:
        kdigo_uo(
            urine,
            weight,
            id_col="stay_id",
            time_col="charttime",
            time_unit="minutes",
        )

    assert error.value.component == "urine_output"
    assert error.value.reason_code == "kdigo_urine_output_numeric_encoding_invalid"


def test_nonnumeric_creatinine_is_not_downgraded_to_patient_data_absent():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 60],
            "crea": [1.0, "bad-source-value"],
        }
    )

    with pytest.raises(KDIGOComponentSchemaError) as error:
        kdigo_stages(
            creatinine,
            id_col="stay_id",
            time_col="charttime",
            time_unit="minutes",
        )

    assert error.value.component == "creatinine"
    assert error.value.reason_code == "kdigo_creatinine_numeric_encoding_invalid"


def test_sparse_hourly_creatinine_is_not_misread_as_minutes():
    # Hour-valued charttime sampled once daily: median spacing 24 h. The old
    # spacing-only heuristic classified this as minutes and inflated the 48 h
    # window to 2,880 h, so the whole-stay minimum became the "48 h baseline".
    creatinine = pd.DataFrame(
        {
            "stay_id": [1] * 6,
            "charttime": [0, 24, 48, 72, 96, 120],
            "crea": [1.0, 0.4, 0.5, 0.6, 0.7, 1.5],
        }
    )

    result = kdigo_aki.kdigo_creatinine(
        creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="hours",
    )

    row = result.loc[result["charttime"] == 120]
    # True 48 h baseline is 0.6 at t=96; the true 7-day baseline is 0.4 at
    # t=24, so stage 3 is the correct KDIGO call. If the axis is misread as
    # minutes both windows inflate 60x and the 48 h baseline would be the
    # whole-stay minimum 0.4 as well -- losing the distinction the window
    # contract exists to enforce.
    assert row["creat_low_past_48hr"].item() == 0.6
    assert row["creat_low_past_7day"].item() == 0.4
    assert row["aki_stage_creat"].item() == 3


def test_numeric_kdigo_charttime_requires_explicit_unit() -> None:
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0, 24, 120],
            "crea": [1.0, 1.0, 1.4],
        }
    )

    with pytest.raises(ValueError, match="time_unit"):
        kdigo_aki.kdigo_creatinine(
            creatinine,
            id_col="stay_id",
            time_col="charttime",
        )


def test_out_of_range_creatinine_is_dropped_without_losing_valid_kdigo_series():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0, 60, 120],
            "crea": [1.0, 999.0, 1.5],
        }
    )

    result = kdigo_aki.kdigo_creatinine(
        creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result["charttime"].tolist() == [0, 120]
    assert result.loc[result["charttime"] == 120, "aki_stage_creat"].item() == 1


def test_negative_urine_observation_is_dropped_without_invalidating_valid_window():
    urine = pd.DataFrame(
        {
            "stay_id": [1] * 7,
            "charttime": [0, 60, 120, 180, 240, 300, 360],
            "urine": [10.0, 10.0, -1.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

    assert 120 not in result["charttime"].tolist()
    assert result["uo_rt_6hr"].notna().any()


def test_kdigo_uo_global_weight_without_id_applies_to_all_rows():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1],
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"weight": [50.0]})

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

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

    result = _calculate_uo_rates_simple(
        urine, weight, "stay_id", "charttime", time_unit="minutes"
    )

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

    result = kdigo_stages(
        creatinine,
        rrt_df=rrt,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    # The component-neutral spine retains the RRT initiation itself instead
    # of waiting for the next creatinine measurement.
    assert result["charttime"].tolist() == [0, 120, 180, 240]
    assert result["aki_stage_rrt"].tolist() == [0, 0, 3, 3]
    # Before a comparison baseline exists, the documented creatinine values do
    # not prove no AKI.  RRT still establishes stage 3 from its initiation.
    assert pd.isna(result["aki_stage"].iloc[0])
    assert result["aki_stage"].iloc[1] == 0
    assert result["aki_stage"].iloc[2] == 3
    assert result["aki_stage"].iloc[3] == 3
    assert pd.isna(result["aki"].iloc[0])
    assert pd.isna(result["aki"].iloc[1])
    assert result["aki_ascertainment"].iloc[1] == (
        "partial_no_observed_positive"
    )
    assert bool(result["aki"].iloc[2]) is True
    assert bool(result["aki"].iloc[3]) is True


def test_kdigo_rrt_exact_timestamp_match_is_not_required():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [100, 200],
            "crea": [1.0, 1.1],
        }
    )
    rrt = pd.DataFrame({"stay_id": [1], "charttime": [150], "rrt": [True]})

    result = kdigo_stages(
        creatinine,
        rrt_df=rrt,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result.loc[result["charttime"] == 200, "aki_stage_rrt"].item() == 3


def test_component_negative_without_complete_window_is_partial_not_negative():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 300],
            "crea": [1.0, 1.0],
        }
    )

    result = kdigo_stages(
        creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    last = result.iloc[-1]
    assert last["aki_stage_creat"] == 0
    assert last["creatinine_ascertainment"] == "negative"
    assert last["creatinine_ascertainment_reason"] == "criterion_negative"
    assert last["urine_ascertainment"] == "indeterminate"
    assert last["urine_ascertainment_reason"] == "source_absent"
    assert last["rrt_ascertainment"] == "indeterminate"
    assert last["rrt_ascertainment_reason"] == "source_absent"
    assert last["observation_window_coverage"] == "partial"
    assert last["aki_ascertainment"] == "partial_no_observed_positive"
    assert pd.isna(last["aki"])
    assert not bool(last["aki_assessable"])


def test_complete_window_and_three_negative_components_prove_negative_aki():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 300],
            "crea": [1.0, 1.0],
        }
    )
    urine = pd.DataFrame(
        {
            "stay_id": [1] * 6,
            "charttime": [0, 60, 120, 180, 240, 300],
            "urine": [100.0] * 6,
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})
    rrt = pd.DataFrame({"stay_id": [1], "charttime": [0], "rrt": [0]})

    result = kdigo_stages(
        creatinine,
        urine_df=urine,
        weight_df=weight,
        rrt_df=rrt,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
        observation_window_coverage={1: "complete"},
    )

    last = result.iloc[-1]
    assert last["creatinine_ascertainment"] == "negative"
    assert last["urine_ascertainment"] == "negative"
    assert last["rrt_ascertainment"] == "negative"
    assert last["rrt_ascertainment_reason"] == "criterion_negative"
    assert last["aki_ascertainment"] == "negative_complete"
    assert bool(last["aki"]) is False
    assert bool(last["aki_assessable"]) is True
    assert bool(last["aki_severe_creat"]) is False
    assert bool(last["aki_severe_uo"]) is False
    assert bool(last["aki_severe_rrt"]) is False
    assert bool(last["aki_severe"]) is False
    assert bool(last["aki_severe_assessable"]) is True
    assert last["aki_severe_ascertainment"] == "negative_complete"


def test_positive_component_overrides_partial_coverage():
    rrt = pd.DataFrame({"stay_id": [1], "charttime": [30], "rrt": [1]})

    result = kdigo_stages(
        None,
        rrt_df=rrt,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result["observation_window_coverage"].iloc[0] == "partial"
    assert result["aki_ascertainment"].iloc[0] == "positive"
    assert bool(result["aki"].iloc[0]) is True
    assert bool(result["aki_severe_rrt"].iloc[0]) is True
    assert bool(result["aki_severe"].iloc[0]) is True
    assert bool(result["aki_severe_assessable"].iloc[0]) is True
    assert result["aki_severe_ascertainment"].iloc[0] == "positive"


def test_stage_one_is_not_misclassified_as_confirmed_no_severe_aki():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 60],
            "crea": [1.0, 1.6],
        }
    )

    result = kdigo_stages(
        creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    last = result.iloc[-1]
    assert last["aki_stage_creat"] == 1
    assert bool(last["aki_severe_creat"]) is False
    assert pd.isna(last["aki_severe"])
    assert not bool(last["aki_severe_assessable"])
    assert last["aki_severe_ascertainment"] == (
        "partial_no_observed_positive"
    )


def test_stage_two_component_confirms_severe_aki_with_partial_coverage():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 60],
            "crea": [1.0, 2.1],
        }
    )

    result = kdigo_stages(
        creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    last = result.iloc[-1]
    assert last["aki_stage_creat"] == 2
    assert bool(last["aki_severe_creat"]) is True
    assert bool(last["aki_severe"]) is True
    assert bool(last["aki_severe_assessable"]) is True
    assert last["aki_severe_ascertainment"] == "positive"


def test_nonempty_rrt_with_unresolved_schema_fails_closed():
    with pytest.raises(KDIGOComponentSchemaError) as error:
        kdigo_stages(
            pd.DataFrame({"stay_id": [1], "charttime": [0], "crea": [1.0]}),
            rrt_df=pd.DataFrame({"wrong": [1]}),
            id_col="stay_id",
            time_col="charttime",
            time_unit="minutes",
        )

    assert error.value.reason_code == "kdigo_rrt_timeline_schema_invalid"


def test_load_kdigo_source_exception_is_not_downgraded_to_missing(monkeypatch):
    import easyicu.api

    def broken_load(**_kwargs):
        raise OSError("corrupt source")

    monkeypatch.setattr(easyicu.api, "load_concepts", broken_load)

    with pytest.raises(KDIGOComponentLoadError) as error:
        load_kdigo_aki("synthetic", verbose=False)

    assert error.value.reason_code == "kdigo_component_load_failed"
    assert error.value.component == "crea"


def test_public_aki_helpers_reject_missing_identity_before_indexing():
    frame = pd.DataFrame(
        {
            "aki": pd.Series([True], dtype="boolean"),
            "aki_stage": pd.Series([1], dtype="Int64"),
            "aki_ascertainment": ["positive"],
        }
    )

    with pytest.raises(KDIGOComponentSchemaError) as incidence_error:
        get_aki_incidence(frame)
    with pytest.raises(KDIGOComponentSchemaError) as summary_error:
        summarize_aki(frame)

    assert incidence_error.value.reason_code == "kdigo_public_api_keys_unresolved"
    assert summary_error.value.reason_code == "kdigo_public_api_id_unresolved"
