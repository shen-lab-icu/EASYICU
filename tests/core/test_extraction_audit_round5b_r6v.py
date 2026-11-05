"""Round-6 unverified claims: small, invented, same-input owner replays.

These characterize the audited version; passing a synthetic replay does not
close a discrepancy in an unavailable clinical cohort. See the lane receipt
for the exact outstanding source-level comparisons.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver, callbacks
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.eicu_io import deduplicate_io, systemic_intake
from easyicu.concept.hirid_fluid_counters import counter_increments
from easyicu.concept.schema import ConceptSource
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.io.ts_utils import change_interval
from easyicu.resources import load_dictionary
from easyicu.scores.aki_profiles import (
    apply_reference_aki,
    apply_source_native_aki,
    build_renal_aki_bundle,
    load_renal_aki_bundle,
)
from easyicu.scores.kdigo_aki import kdigo_uo
from easyicu.scores.sofa2 import sofa2_renal, sofa2_resp
from easyicu.scores.urine_output import collection_windows
from easyicu.table import ICUTable
from easyicu.utils.callback_utils import mimic_dur_incv, mimic_rate_cv, mimic_rate_mv
from tests.support.urine_raw_source import _load, _source, _stays, _urine_source  # noqa: F401


def _table(name, times, values, *, id_col="stay_id"):
    return ICUTable(
        pd.DataFrame({id_col: 1, "charttime": times, name: values}),
        id_columns=[id_col], index_column="charttime", value_column=name,
    )


def _context(name, database="miiv", **kwargs):
    return callbacks.ConceptCallbackContext(
        name, None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name=database)), None,
        kwargs=kwargs,
    )








@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "sic"])
def test_r6_f005_rejected_gap_has_no_volume_but_prior_overlap_can_remain(database):
    # Every accepted hourly collection through 20.25 contains 10 mL.
    # The 14 h collection ending 34.25 is rejected in its entirety.
    times = np.r_[np.arange(21) + .25, 34.25, 40.25, 46.25, 52.25, 58.25]
    columns = ["urine24", "urine24_est", "urine24_observed_ml",
               "urine24_covered_h", "urine24_available_at",
               "urine24_est_available_at", "urine24_uses_future"]
    outputs = []
    for rejected_amount in (1., 9999.):
        amounts = [10.] * 21 + [rejected_amount] + [60.] * 4
        actual = callbacks._callback_urine24(
            {"urine": _table("urine", times, amounts)},
            _context("urine24", database),
        ).data.set_index("charttime")
        outputs.append(actual[columns])
        # Independently enumerate the accepted right-labelled collection bins.
        # No cell in 21..34 receives a share of the rejected amount.
        cells = {h: (10., h + .25) for h in range(21)}
        cells.update({h: (10., end + .25) for end in (40, 46, 52, 58)
                      for h in range(end - 5, end + 1)})
        for hour, row in actual.iterrows():
            segment_start = 34 if hour >= 34 else 0
            overlap = [cells[h] for h in cells
                       if max(hour - 23, segment_start) <= h <= hour]
            covered = len(overlap)
            volume = sum(v for v, _ in overlap)
            assert row.urine24_covered_h == covered
            if covered:
                assert row.urine24_observed_ml == pytest.approx(volume)
            else:
                assert pd.isna(row.urine24_observed_ml)
            assert pd.notna(row.urine24) == (covered == 24)
            assert pd.notna(row.urine24_est) == (18 <= covered < 24)
            if 18 <= covered < 24:
                assert row.urine24_est == pytest.approx(volume * 24 / covered)
                assert row.urine24_est_available_at == max(t for _, t in overlap)
            if covered == 24:
                assert row.urine24 == pytest.approx(volume)
        # This is legal residual coverage, not a copied rolling sum: it decays.
        assert actual.loc[24., "urine24_covered_h"] == 20
        assert actual.loc[26., "urine24_covered_h"] == 18
        assert actual.loc[26., "urine24_est_available_at"] == 20.25
        assert not actual.loc[26., "urine24_uses_future"]
        assert actual.loc[34., "urine24_covered_h"] == 0
    pd.testing.assert_frame_equal(*outputs)


def test_r6_f005_left_edge_invalid_channel_same_time_and_stage_evidence():
    ends = np.array([0., 4., 8., 12., 16., 20., 24.])
    amounts = np.array([0., 40., 80., 120., 160., 200., 240.])
    volume, covered, available = collection_windows(ends, amounts, [6])[6]
    assert (volume[-1], covered[-1], available[-1]) == (340., 6., 24.)
    for invalid_amount in (np.nan, -1.):
        invalid = amounts.copy()
        invalid[5] = invalid_amount
        volume, covered, _ = collection_windows(ends, invalid, [24])[24]
        assert (volume[-1], covered[-1]) == (240., 4.)
    # Two channels with the same end add volume, never duplicate duration.
    urine = pd.DataFrame({"stay_id": 1, "charttime": np.repeat(ends, 2),
                          "urine": np.repeat(amounts / 2, 2)})
    weight = pd.DataFrame({"stay_id": [1], "weight": [50.]})
    result = kdigo_uo(urine, weight, time_unit="hours").iloc[-1]
    assert result.uo_rt_6hr == pytest.approx(340 / 6 / 50)
    assert result.uo_rt_24hr == pytest.approx(840 / 24 / 50)
    assert result.uo_rt_24hr_covered_h == 24
    assert result.aki_stage_uo == 0


def test_r6_f027_chart_state_machine_and_explicit_intervals_keep_distinct_evidence():
    # An 8 h setting gap continues, an 8.01 h gap splits; oxygen closes at
    # the preceding setting, extubation at its own timestamp. No guessed tail.
    chart = pd.DataFrame({
        "icustay_id": 1,
        "charttime": [.25, 8.25, 9., 10., 18.01, 19., 20.],
        "itemid": [60, 60, 467, 60, 60, 60, 640],
        "value": ["5", "5", "Cannula", "5", "5", "5", "Extubated"],
        "error": 0,
    })
    source = ConceptSource(
        table="chartevents", sub_var="itemid", value_var="value",
        index_var="charttime", callback="mimiciii_chart_ventilation_intervals",
    )
    result = _apply_callback(chart, source, concept_name="mech_vent")
    assert result.charttime.tolist() == [.25, 18.01]
    assert result.dur_var.tolist() == pytest.approx([480., 119.4])
    procedure = pd.DataFrame({"icustay_id": [1, 1], "charttime": [8.5, 18.5],
                              "mech_vent": [225794, 225792], "dur_var": [60., 60.]})
    explicit = _apply_callback(procedure, ConceptSource(
        table="procedureevents_mv", value_var="itemid",
        callback="mimiciii_explicit_ventilation_interval",
    ), concept_name="mech_vent")
    assert explicit.mech_vent.tolist() == ["noninvasive", "invasive"]
    # The exact IMV [.25,8.25) and NIV [8.5,9.5) both touch hour 8.
    # An independent replay must allow that mixed hour instead of XOR labels.
    assert result.charttime.iloc[0] + result.dur_var.iloc[0] / 60 == 8.25
    assert explicit.charttime.iloc[0] == 8.5
    assert explicit.charttime.iloc[0] + explicit.dur_var.iloc[0] / 60 == 9.5


@pytest.mark.parametrize("reverse", [False, True])
def test_r6_f028_real_selector_resolves_same_time_native_modes_before_hour_pooling(tmp_path, reverse):
    origin = pd.Timestamp("2180-01-01 00:17")
    chart = pd.DataFrame({
        "stay_id": 1, "subject_id": 1, "hadm_id": 1,
        "charttime": origin + pd.to_timedelta([.25, .25, .75, 1.25], unit="h"),
        "itemid": [223849, 229314, 223849, 223849],
        "value": ["PCV+", "VOL/AC", "VOL/AC", "Standby"], "valuenum": np.nan,
        "valueuom": None,
    })
    source = _source(tmp_path, "miiv", chartevents=chart.iloc[::-1] if reverse else chart,
                          icustays=pd.DataFrame({"stay_id": [1], "subject_id": [1],
                                                "hadm_id": [1], "intime": [origin],
                                                "outtime": [origin + pd.Timedelta(days=1)]}))
    results = ConceptResolver(load_dictionary()).load_concepts(
        ["vent_mode", "vent_breath_seq"], source, merge=False,
        interval=pd.Timedelta(hours=1), r_compatible=False, verbose=False,
        concept_workers=1, patient_ids={"stay_id": [1]},
    )
    mode, sequence = results["vent_mode"].data, results["vent_breath_seq"].data
    assert mode.vent_mode.tolist() == ["pressure"]
    assert sequence.vent_breath_seq.tolist() == ["controlled"]
    assert mode.charttime.tolist() == [0.]
    assert sequence.charttime.tolist() == [0.]


def test_r6_f029_duration_helper_is_not_rate_owner_and_cv_mv_use_different_pooling():
    origin = pd.Timestamp("2180-01-01 00:17")
    bounds = pd.DataFrame({"icustay_id": [1], "intime": [origin],
                           "outtime": [origin + pd.Timedelta(hours=12)]})
    frame = pd.DataFrame({"icustay_id": 1, "linkorderid": [1, 1, 2, 2],
                          "charttime": origin + pd.to_timedelta([.25, 2.25, .25, 2.25], unit="h"),
                          "rate": [.2, 0., .6, 0.], "stopped": [None, "Stopped"] * 2})
    duration = mimic_dur_incv(frame, val_col="norepi_dur", icu_stays=bounds)
    assert duration.norepi_dur.tolist() == [2.]
    cv = mimic_rate_cv(frame, val_col="rate", grp_var="linkorderid",
                       id_cols=["icustay_id"], admission_times=bounds)
    cv["charttime"] = (cv.charttime - origin) / pd.Timedelta(hours=1)
    # ricu's CV owner floors to the ICU origin and takes the median across
    # simultaneous infusion groups. This cannot be replayed as an additive MV rate.
    assert cv.set_index("charttime").rate.to_dict() == {0.: .4, 1.: .4, 2.: 0.}
    mv = pd.DataFrame({"icustay_id": [1, 1], "orderid": [1, 2], "starttime": [.25, .75],
                       "endtime": [1., 2.], "rate": [.2, .6]})
    expanded = mimic_rate_mv(mv, val_col="rate", stop_var="endtime",
                             id_cols=["icustay_id", "orderid"])
    assert expanded.starttime.tolist() == [0., 0., 1.]
    assert expanded.rate.tolist() == [.2, .6, .6]
    # Actual dose conserves the precise intervals, not rate * published cells.
    raw = mv.set_index("orderid")
    starts = expanded.orderid.map(raw.starttime)
    ends = expanded.orderid.map(raw.endtime)
    overlap = np.minimum(ends, expanded.starttime + 1) - np.maximum(starts, expanded.starttime)
    assert (expanded.rate * overlap * 60).sum() == pytest.approx(54.)
    assert ((mv.endtime - mv.starttime) * mv.rate * 60).sum() == pytest.approx(54.)
    rates = {name: _table(name, [0.], [value]) for name, value in {
        "norepi_rate": .2, "epi_rate": .1, "dopa_rate": 15., "phn_rate": 1.,
    }.items()}
    equivalent = callbacks._callback_norepi_equiv(rates, _context("norepi_equiv", "mimic"))
    assert equivalent.data.norepi_equiv.item() == pytest.approx(.5)


def test_r6_f030_owner_point_scores_differ_from_twenty_four_cell_worst_values():
    times = list(map(float, range(27)))
    pafi = _table("pafi", times, [80.] + [500.] * 26)
    vent = _table("vent_ind", times, [True] * 27)
    point = callbacks._callback_sofa_resp({"pafi": pafi, "vent_ind": vent},
                                         _context("sofa_resp", "mimic"))
    assert point.data.sofa_resp.tolist() == [4] + [0] * 26
    rolled = callbacks._callback_sofa_score({"sofa_resp": point},
                                           _context("sofa", "mimic", keep_components=True)).data
    expected = [4] * 24 + [0] * 3
    assert rolled.sofa_resp.tolist() == expected
    assert rolled.sofa.tolist() == expected
    point2 = callbacks.CALLBACK_REGISTRY["sofa2_resp"](
        {"pafi": pafi, "adv_resp": _table("adv_resp", times, [True] * 27)},
        _context("sofa2_resp", "mimic"),
    )
    assert point2.data.sofa2_resp.tolist() == [3] + [0] * 26
    rolled2 = callbacks._callback_sofa2_score({"sofa2_resp": point2},
                                             _context("sofa2", "mimic", keep_components=True)).data
    assert rolled2.sofa2_resp.tolist() == [3] * 24 + [0] * 3


@pytest.mark.parametrize("values,expected", [
    (["invasive", "noninvasive"], "invasive"),
    (["noninvasive", "invasive"], "noninvasive"),
])
def test_r6_f039_hourly_first_preserves_chronology_instead_of_forcing_imv(tmp_path, values, expected):
    points = _table("mech_vent", [.25, .75], values)
    actual = change_interval(points, interval=pd.Timedelta(hours=1),
                             aggregation="first", time_unit="hours").data
    assert actual.mech_vent.tolist() == [expected]
    # Exact-time conflict has a deterministic value tie-break (not evidence
    # that every NIV hour with settings should be reclassified as IMV).
    tied = _table("mech_vent", [.25, .25], ["noninvasive", "invasive"])
    assert change_interval(tied, interval=pd.Timedelta(hours=1),
                           aggregation="first", time_unit="hours").data.mech_vent.item() == "invasive"
    raw = pd.DataFrame({"patientunitstayid": [1, 1], "respchartoffset": [15., 30.],
                        "respchartvaluelabel": ["B1: IPAP", "PEEP"],
                        "mech_vent": ["12", "5"]})
    evidence = _apply_callback(raw, ConceptSource(
        table="respiratorycharting", sub_var="respchartvaluelabel",
        value_var="respchartvalue", index_var="respchartoffset",
        callback="eicu_ventilator_setting_evidence",
    ), concept_name="mech_vent")
    assert evidence.mech_vent.tolist() == ["noninvasive", "noninvasive"]

    patient = pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                            "unitdischargeoffset": [600], "hospitaladmitoffset": [0]})
    for label, offsets, modes, answer in [
        ("chronology", [15., 45.], values, expected),
        ("tie", [15., 15.], ["noninvasive", "invasive"], "invasive"),
        ("niv", [15., 30.], ["noninvasive", "peep"], "noninvasive"),
    ]:
        chart = pd.DataFrame({"patientunitstayid": 1, "respchartoffset": offsets,
            "respchartvaluelabel": ["PEEP" if m == "peep" else "O2 Device" for m in modes],
            "respchartvalue": [{"invasive": "Ventilator", "noninvasive": "BiPAP", "peep": "5"}[m]
                               for m in modes]})
        source = _source(tmp_path / label, "eicu", patient=patient, respiratorycharting=chart)
        published = _load(source, ["mech_vent", "vent_ind"])["mech_vent"].data
        if label == "chronology":
            # The public window concept retains both categories in a mixed
            # hour. A second first() here would be a different request.
            assert published.mech_vent.tolist() == values
        elif label == "tie":
            assert set(published.mech_vent) == {"invasive", "noninvasive"}
        else:
            assert set(published.mech_vent) == {answer}



@pytest.mark.parametrize("database", ["eicu", "sic"])
def test_r6_f040_support_metabolic_crisis_and_owner_availability_are_separate(database):
    owner = callbacks.CALLBACK_REGISTRY["sofa2_resp"]({
        "pafi": _table("pafi", [0., 1., 2., 3.], [60., 60., np.nan, 500.]),
        "adv_resp": _table("adv_resp", [0., 1., 2., 3.], [False, True, True, False]),
        "oxygenation_sustained_1h": _table("oxygenation_sustained_1h", [0., 1., 2., 3.], [True, False, True, True]),
    }, _context("sofa2_resp", database))
    resp = owner.data
    assert resp.sofa2_resp.tolist() == [2, 0, 0, 0]
    assert resp.sofa2_resp_available.tolist() == [1, 0, 0, 1]
    rolled = callbacks._callback_sofa2_score(
        {"sofa2_resp": owner}, _context("sofa2", database, keep_components=True),
    ).data
    # An unavailable synthetic zero does not erase the preceding valid state.
    # A recovered normal observation is still subject to the 24-cell worst rule.
    assert rolled.sofa2_resp.tolist() == [2] * 4
    assert rolled.sofa2_resp_available.tolist() == [1] * 4
    # Creatinine + potassium establish metabolic RRT criteria independently
    # of charted treatment; a non-renal-only treatment must not score 4.
    renal = sofa2_renal(
        pd.Series([1., 1.3, np.nan, 1.]),
        potassium=pd.Series([6., 6., np.nan, 4.]),
        ph=pd.Series([7.4] * 4), bicarb=pd.Series([24.] * 4),
        uo_6h=pd.Series([np.nan] * 4), uo_6h_covered_h=pd.Series([0.] * 4),
        rrt=pd.Series([False, False, True, True]),
        rrt_nonrenal_only=pd.Series([False, False, True, False]),
    )
    assert renal.tolist() == [0, 4, 0, 4]
    assert sofa2_resp(pd.Series([500.]), ecmo=pd.Series([True])).item() == 4


def test_r6_f041_public_smoothing_is_trailing_and_mimic3_has_no_native_smoother():
    # Both t=7 and its t=1 left edge are included. t=7.01 excludes t=1.
    times = [0., 1., 2., 7., 7.01, 20.]
    crea = pd.DataFrame({"stay_id": 1, "charttime": times,
                         "crea": [1., 1.4, 1., 1., 1., 1.]})
    result = apply_reference_aki(crea_df=crea, time_unit="hours")
    stage = result.aki_stage_reference.to_numpy(dtype=float)
    expected = [max(stage[j] for j, other in enumerate(times) if t - 6 <= other <= t)
                for t in times]
    assert stage.tolist() == [0., 1., 0., 0., 0., 0.]
    assert result.aki_stage_reference_smoothed_6h.tolist() == expected
    assert not result.aki_reference_uses_future.any()
    native = apply_source_native_aki("miiv", crea_df=crea, time_unit="hours")
    assert native.aki_stage_source_native_smoothed.tolist() == expected
    # III omits this field by contract; the renderer must not invent zero.
    bundle = build_renal_aki_bundle("mimic", crea_df=crea, time_unit="hours")
    assert "aki_stage_source_native_smoothed" not in bundle


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("time_unit", ["hours", "minutes", "datetime"])
def test_r6_f041_all_null_components_remain_unavailable_to_smoothing(native, time_unit):
    # Stay 2 has no surviving measurements. Stay 1 recovers, loses evidence
    # for more than six hours, then recovers again. No future backfill.
    crea = pd.DataFrame({
        "stay_id": [1] * 6 + [2] * 2,
        "charttime": [0., 1., 2., 3., 10., 11., 0., 1.],
        "crea": [np.nan, 1., 1.4, np.nan, np.nan, 1., np.nan, np.nan],
    })
    if time_unit == "minutes":
        crea.charttime *= 60
    elif time_unit == "datetime":
        crea.charttime = pd.Timestamp("2180-01-01") + pd.to_timedelta(crea.charttime, unit="h")
    kwargs = dict(crea_df=crea, time_unit=time_unit)
    if native:
        result = apply_source_native_aki("miiv", **kwargs)
        name = "aki_stage_source_native_smoothed"
    else:
        result = apply_reference_aki(**kwargs)
        name = "aki_stage_reference_smoothed_6h"
    first = result.loc[result.stay_id == 1, name]
    assert first.isna().tolist() == [True, False, False, False, True, False]
    assert first.dropna().tolist() == [0, 1, 1, 0]
    assert result.loc[result.stay_id == 2, name].isna().all()
    # Entirely empty component calculations used to lose their key columns.
    empty = crea.loc[crea.stay_id == 2]
    kwargs["crea_df"] = empty
    actual = (apply_source_native_aki("miiv", **kwargs) if native
              else apply_reference_aki(**kwargs))
    assert actual[name].isna().all()


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("component", ["crea", "urine"])
def test_r6_f041_all_null_source_calculation_keeps_keyed_missing_smoothing(database, component):
    empty = pd.DataFrame({"stay_id": [1, 1], "charttime": [0., 1.],
                          component: [np.nan, np.nan]})
    kwargs = {f"{component}_df": empty, "time_unit": "hours"}
    if component == "urine":
        kwargs["weight_df"] = pd.DataFrame({"stay_id": [1], "weight": [70.]})
    reference = apply_reference_aki(**kwargs)
    assert reference.charttime.tolist() == [0., 1.]
    assert reference.aki_stage_reference_smoothed_6h.isna().all()
    native = apply_source_native_aki(database, **kwargs)
    assert native.charttime.tolist() == [0., 1.]
    if database == "miiv":
        assert native.aki_stage_source_native_smoothed.isna().all()
    else:
        assert "aki_stage_source_native_smoothed" not in native


@pytest.mark.parametrize("native", [False, True])
def test_r6_f041_active_rrt_supports_smoothing_without_creatinine_or_urine(native):
    rrt = pd.DataFrame({"stay_id": 1, "charttime": [0., 6., 6.01],
                        "rrt": [True, False, False]})
    if native:
        result = apply_source_native_aki("miiv", rrt_df=rrt, time_unit="hours")
        name = "aki_stage_source_native_smoothed"
    else:
        result = apply_reference_aki(rrt_df=rrt, time_unit="hours")
        name = "aki_stage_reference_smoothed_6h"
    assert result[name].iloc[:2].tolist() == [3, 3]
    assert pd.isna(result[name].iloc[2])


@pytest.mark.parametrize("entry", ["reference", "native", "export"])
@pytest.mark.parametrize("weight", [70., np.nan])
def test_r6_f041_unassessable_urine_cannot_renew_normal_smoothing(entry, weight):
    urine = pd.DataFrame({"stay_id": 1, "charttime": [0., 6., 20.], "urine": [10., 60., 900.]})
    weights = pd.DataFrame({"stay_id": [1], "weight": [weight]})
    if entry == "export":
        # This is the native-v2 renal loader's public preloaded-input path.
        empty = pd.DataFrame(columns=["stay_id", "charttime", "rrt"])
        result = load_renal_aki_bundle("miiv", preloaded_data={
            "crea": empty.rename(columns={"rrt": "crea"}), "urine": urine,
            "weight": weights, "rrt": empty, "crrt": empty,
        })
    else:
        kwargs = dict(urine_df=urine, weight_df=weights, time_unit="hours")
        result = (apply_reference_aki(**kwargs) if entry == "reference"
                  else apply_source_native_aki("miiv", **kwargs))
    name = ("aki_stage_source_native_smoothed" if entry == "native"
            else "aki_stage_reference_smoothed_6h")
    # First collection is too short; the last collection is rejected (>12h).
    # Surviving calculation keys alone do not establish a stage or a normal 0.
    assert pd.isna(result[name].iloc[0])
    assert pd.isna(result[name].iloc[2])
    if np.isfinite(weight):
        assert result[name].iloc[1] == 1
    else:
        assert result[name].isna().all()


def test_r6_f042_charted_plateau_checks_pip_peep_without_using_niv_as_a_veto(tmp_path):
    tables = {
        "plateau_pres_raw": _table("plateau_pres_raw", [0., 1., 2., 3.], [20., 40., 4., 22.]),
        "pip": _table("pip", [0., 1., 2., 3.], [25., 25., 25., np.nan]),
        "peep": _table("peep", [0., 1., 2., 3.], [5., 5., 5., np.nan]),
        "mech_vent": _table("mech_vent", [0., 1., 2., 3.], ["noninvasive"] * 4),
    }
    actual = callbacks._callback_plateau_pressure(tables, _context("plateau_pres", "hirid")).data
    assert actual.charttime.tolist() == [0., 3.]
    assert actual.plateau_pres.tolist() == [20., 22.]
    definition = load_dictionary().get("plateau_pres_raw")
    assert definition.sources["hirid"][0].ids == [3200]

    origin = pd.Timestamp("2180-01-01 12:00")
    rows = [(h, item, value) for item, values in {
        3200: [20., 40., 4., 22.], 3110: [25., 25., 25., np.nan],
        2600: [5., 5., 5., np.nan], 15001552: [3., 3., 3., 3.],
    }.items() for h, value in enumerate(values)]
    observation = pd.DataFrame(rows, columns=["datetime", "variableid", "value"])
    observation["patientid"] = 1
    observation["datetime"] = origin + pd.to_timedelta(observation.datetime, unit="h")
    source = _source(tmp_path, "hirid", observations=observation,
                     general=pd.DataFrame({"patientid": [1], "admissiontime": [origin]}))
    loaded = _load(source, ["plateau_pres", "mech_vent"])
    assert loaded["plateau_pres"].data.plateau_pres.tolist() == [20., 22.]
    assert set(loaded["mech_vent"].data.mech_vent) == {"noninvasive"}


def test_r6_f043_sic_hpm_369_cannot_enter_creatinine_selector_or_reference_stage(tmp_path):
    definition = load_dictionary().get("crea")
    # R6-F023 adds enzymatic creatinine 368; HPM 369 stays out.
    assert [s.ids for s in definition.sources["sic"]] == [[367, 368]]
    source = _source(tmp_path, "sic", laboratory=pd.DataFrame({
        "CaseID": [1] * 6, "LaboratoryID": [367, 369, 368, 369, 367, 369],
        # Both candidate values are inside the main concept's numeric bounds:
        # source selection, rather than value clipping, must exclude them.
        "Offset": [0., 0., 3600., 3600., 7200., 7200.],
        "LaboratoryValue": [1., 3., 1.4, 12., 1., 4.],
    }), cases=pd.DataFrame({"CaseID": [1], "PatientID": [1], "ICUOffset": [0.],
                           "TimeOfStay": [86400.], "OffsetOfDeath": [np.nan]}))
    loaded = ConceptResolver(ConceptDictionary({"crea": definition})).load_concepts(
        ["crea"], source, merge=False, r_compatible=False, verbose=False,
        interval=pd.Timedelta(hours=1), concept_workers=1, patient_ids={"CaseID": [1]},
    )["crea"].data
    assert loaded.crea.tolist() == [1., 1.4, 1.]
    result = apply_reference_aki(crea_df=loaded, id_col="CaseID", time_col="Offset",
                                 time_unit="hours")
    assert result.aki_stage_reference.tolist() == [0, 1, 0]


@pytest.mark.parametrize("values,total", [
    ([1000., 10000., 1100., 1200.], 1200.),  # established isolated spike
    ([1000., 10000., 990., 1200.], 11200.),  # spike OR real rise + restart
    ([0., 10000., 0., 100.], 10100.),        # zero baseline is ambiguous too
    ([1000., 10000., 11000., 12000.], 12000.),  # sustained large input
    ([1000., 10000., 100., 200.], 10200.),    # restart must survive
])
def test_r6_f044_counter_trajectory_alone_cannot_disambiguate_near_return_restart(tmp_path, values, total):
    raw = pd.DataFrame({"patientid": 1,
                        "datetime": pd.date_range("2180-01-01 13:00", periods=4, freq="h"),
                        "value": values})
    result = counter_increments(raw, id_col="patientid", time_col="datetime", value_col="value")
    assert result.volume.sum() == total
    if total == 1200.:
        assert result.end.tolist() == raw.datetime.iloc[[0, 2, 3]].tolist()
        assert result.start.iloc[1] == raw.datetime.iloc[0]

    source = _source(tmp_path, "hirid", observations=raw.assign(variableid=30005010),
                     general=pd.DataFrame({"patientid": [1],
                         "admissiontime": [pd.Timestamp("2180-01-01 12:00")]}))
    published = _load(source, ["total_input_ml"])["total_input_ml"].data
    assert published.total_input_ml.sum() == pytest.approx(total)


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f045_io_raw_identity_classification_and_rate_overlap_conserve_volume(tmp_path, layout):
    prefix = "flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|"
    raw = pd.DataFrame({
        "patientunitstayid": 1,
        "intakeoutputid": range(9),
        "intakeoutputoffset": [30, 30, 30, 30, 30, 30, 30, 30, 90],
        "intakeoutputentryoffset": [31, 31, 32, 31, 31, 31, 31, 31, 91],
        "cellpath": [prefix + name for name in ["IV", "IV", "IV", "Daily total", "Bladder irrigation",
                                                 "PCA Amount (mcg)", "Nutrition Total", "IV (mL/hr)", "IV (mL/hr)"]],
        "cellvaluenumeric": [100., 100., 100., 900., 900., 900., 20., 60., 0.],
    })
    deduplicated = deduplicate_io(raw)
    assert len(deduplicated) == 8
    assert deduplicated.attrs["eicu_io_dedup_audit"]["duplicates_removed"] == 1
    actual = systemic_intake(deduplicated, "cellvaluenumeric")
    actual["hour"] = np.floor(actual.intakeoutputoffset / 60)
    assert actual.groupby("hour").cellvaluenumeric.sum().to_dict() == {0.: 250., 1.: 30.}
    assert actual.cellvaluenumeric.sum() == 280.
    assert actual.attrs["eicu_intake_audit"]["nonsystemic_or_cumulative_dropped"] == 3

    source = _source(tmp_path, "eicu", layout=layout, intakeoutput=raw.assign(celllabel="IV"),
                     patient=pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                         "unitdischargeoffset": [600], "hospitaladmitoffset": [0]}))
    for names in [["total_input_ml"], ["total_input_ml", "urine", "other_output_ml", "fluid_balance"]]:
        loaded = _load(source, names)["total_input_ml"]
        assert loaded.data.set_index(loaded.index_column).total_input_ml.to_dict() == {0.: 250., 1.: 30.}




@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "sic"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f005_public_selector_rejected_collection_is_inert(tmp_path, database, layout):
    times = np.r_[np.arange(21) + .25, 34.25, 40.25, 46.25, 52.25, 58.25]
    results = []
    for amount in [1., 4999.]:  # both survive the raw concept's 5000 mL bound
        source = _urine_source(tmp_path / str(amount), database, times,
                               [10.] * 21 + [amount] + [60.] * 4, layout=layout)
        names = ["urine24", "urine24_est", "uo_6h", "uo_24h"]
        batch = _load(source, names)
        single = _load(source, ["urine24"])["urine24"]
        pd.testing.assert_frame_equal(single.data, batch["urine24"].data)
        daily = single.data.set_index(single.index_column)
        assert daily.loc[24., "urine24_covered_h"] == 20
        assert daily.loc[26., "urine24_covered_h"] == 18
        assert daily.loc[34., "urine24_covered_h"] == 0
        assert daily.loc[26., "urine24_est_available_at"] < 26
        results.append({name: table.data for name, table in batch.items()})
    for name in names:
        pd.testing.assert_frame_equal(results[0][name], results[1][name])


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "sic"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
@pytest.mark.parametrize("name", ["urine24", "uo_6h"])
@pytest.mark.parametrize("compatible", [False, True])
def test_r6_f005_partial_wide_request_keeps_urine_availability(tmp_path, monkeypatch, database, layout, name, compatible):
    from easyicu.scores.urine_output import urine_availability_columns

    times = np.r_[np.arange(25) + .25, 38.25, 44.25]
    source = _urine_source(tmp_path, database, times, [10.] * 25 + [900., 60.], layout=layout)
    names = ["k", "na", "cl", {"eicu": "phos", "sic": "bili"}.get(database, "mg")]
    dictionary = load_dictionary()
    definitions = [dictionary.get(n).sources[database][0] for n in names]
    assert len({d.table for d in definitions}) == 1
    assert all(d.callback is None for d in definitions)
    ids = [d.ids[0] for d in definitions]
    values = [4., 140., 100., 2.]
    if database in {"miiv", "mimic"}:
        # Real lab rows have admission identifiers, not ICU stay identifiers.
        labs = pd.DataFrame({"subject_id": 1, "hadm_id": 1,
            "charttime": pd.Timestamp("2180-01-01 00:17"), "itemid": ids,
            "valuenum": values, "valueuom": ["mEq/L"] * 3 + ["mg/dL"]})
    elif database == "eicu":
        labs = pd.DataFrame({"patientunitstayid": 1, "labresultoffset": 0.,
                             "labname": ids, "labresult": values,
                             "labmeasurenameinterface": ["mEq/L"] * 3 + ["mg/dL"]})
    else:
        labs = pd.DataFrame({"CaseID": 1, "Offset": 0., "LaboratoryID": ids,
                             "LaboratoryValue": values})
    source = _source(tmp_path, database, layout=layout, **{definitions[0].table: labs})

    calls = []
    original = ConceptResolver._merge_partial_wide_result

    def record_partial(resolver, merged, tables, requested, covered, **kwargs):
        calls.append(set(covered))
        return original(resolver, merged, tables, requested, covered, **kwargs)

    monkeypatch.setattr(ConceptResolver, "_merge_partial_wide_result", record_partial)

    def publish(requested):
        frame = ConceptResolver(dictionary).load_concepts(
            requested, source, merge=True, r_compatible=compatible,
            interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1,
        )
        from easyicu.scores.kdigo_aki import detect_time_col

        return frame.rename(columns={detect_time_col(frame): "charttime"}).set_index(
            [source.config.id_configs["icustay"].id, "charttime"],
        )

    single = publish([name])
    batch = publish([*names, name])
    if layout == "bucket" and compatible:
        assert set(names) in calls  # prove that the optimized branch was reached
    assert batch.loc[(1, 0.), names].tolist() == pytest.approx(values)
    receipts = urine_availability_columns(name)
    assert single[receipts].notna().any().all()
    assert set(receipts) <= set(batch.columns)
    columns = [name, *receipts] if compatible else list(single.columns)
    pd.testing.assert_frame_equal(
        batch.reindex(single.index)[columns], single[columns],
        check_dtype=False,
    )


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f041_public_resolver_does_not_publish_rejected_urine_as_normal(tmp_path, database, layout):
    source = _urine_source(tmp_path, database, [0., 6., 20.], [10., 60., 900.], layout=layout)
    result = _load(source, ["kdigo_aki", "urine", "weight"])["kdigo_aki"].data
    assert result.charttime.tolist() == [0., 6., 20.]
    # A valid six-hour window exists at 6; no supported stage exists at 20.
    assert result.loc[result.charttime == 6, "aki_stage_reference_smoothed_6h"].item() == 1
    assert result.loc[result.charttime == 20, "aki_stage_reference_smoothed_6h"].isna().all()


@pytest.mark.parametrize("database", ["miiv", "mimic"])
def test_r6_f027_public_selector_preserves_distinct_ventilation_intervals(tmp_path, database):
    origin = pd.Timestamp("2180-01-01 00:17")
    key = "stay_id" if database == "miiv" else "icustay_id"
    procedures = pd.DataFrame({key: 1, "subject_id": 1, "hadm_id": 1,
        "starttime": origin + pd.to_timedelta([8.5, 18.5], unit="h"),
        "endtime": origin + pd.to_timedelta([9.5, 19.5], unit="h"),
        "itemid": [225794, 225792], "value": [60., 60.], "valueuom": "min",
        "cancelreason": 0, "statusdescription": "FinishedRunning"})
    frames = {"icustays": _stays(database, origin),
              "procedureevents" if database == "miiv" else "procedureevents_mv": procedures}
    if database == "mimic":
        frames["chartevents"] = pd.DataFrame({key: 1, "subject_id": 1, "hadm_id": 1,
            "charttime": origin + pd.to_timedelta([.25, 8.25, 9., 10., (18. + 1 / 60), 19., 20.], unit="h"),
            "itemid": [60, 60, 467, 60, 60, 60, 640],
            "value": ["5", "5", "Cannula", "5", "5", "5", "Extubated"],
            "valuenum": [5., 5., np.nan, 5., 5., 5., np.nan], "valueuom": None, "error": 0})
    source = _source(tmp_path, database, **frames)
    loaded = ConceptResolver(load_dictionary()).load_concepts(
        ["mech_vent"], source, merge=False, interval=None, aggregate=False,
        r_compatible=False, verbose=False, concept_workers=1,
    )["mech_vent"]
    frame = loaded.data
    # The procedure selector must preserve both categories and its exact start.
    assert set(frame.mech_vent) == {"invasive", "noninvasive"}
    assert frame.loc[frame.mech_vent.eq("noninvasive"), loaded.index_column].tolist() == [8.5]
    if database == "mimic":
        assert set(frame.loc[frame.mech_vent.eq("invasive"), loaded.index_column]) == {.25, (18. + 1 / 60), 18.5}


@pytest.mark.parametrize("system", ["cv", "mv"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f029_public_rate_selectors_and_equivalent_share_source_clock(tmp_path, system, layout):
    origin = pd.Timestamp("2180-01-01 00:17")
    channels = {"norepi_rate": (30120, 221906, .2), "epi_rate": (30119, 221289, .1),
                "dopa_rate": (30043, 221662, 15.), "phn_rate": (30128, 221749, 1.)}
    rows = []
    for cv, mv, rate in channels.values():
        if system == "cv":
            for group, scale in [(1, .5), (2, 1.5)]:
                for hour, value in [(.25, rate * scale), (2.25, 0.)]:
                    rows.append({"icustay_id": 1, "subject_id": 1, "hadm_id": 1,
                        "charttime": origin + pd.Timedelta(hours=hour), "itemid": cv,
                        "rate": value, "rateuom": "mcgkgmin", "linkorderid": cv * 10 + group,
                        "stopped": "Stopped" if value == 0 else None})
        else:
            rows.append({"icustay_id": 1, "subject_id": 1, "hadm_id": 1,
                "starttime": origin + pd.Timedelta(hours=.25),
                "endtime": origin + pd.Timedelta(hours=2), "itemid": mv,
                "rate": rate, "rateuom": "mcg/kg/min", "orderid": mv,
                "linkorderid": mv, "patientweight": 70., "statusdescription": "FinishedRunning"})
    source = _source(tmp_path, "mimic", layout=layout, **{
        "icustays": _stays("mimic", origin), f"inputevents_{system}": pd.DataFrame(rows)})
    loaded = _load(source, [*channels, "norepi_equiv"])
    for name, (_, _, rate) in channels.items():
        table = loaded[name]
        expected = {0.: rate, 1.: rate, **({2.: 0.} if system == "cv" else {})}
        assert table.data.set_index(table.index_column)[name].to_dict() == pytest.approx(expected)
        pd.testing.assert_frame_equal(_load(source, [name])[name].data, table.data)
    equiv = loaded["norepi_equiv"]
    assert equiv.data.set_index(equiv.index_column).norepi_equiv.to_dict() == pytest.approx(
        {0.: .5, 1.: .5, **({2.: 0.} if system == "cv" else {})})


def test_r6_f030_public_parent_request_uses_twenty_four_cell_worst(tmp_path):
    origin = pd.Timestamp("2180-01-01 00:17")
    times = list(range(27))
    rows = []
    for h in times:
        for item, value in [(779, 40. if h == 0 else 250.), (223835, 50.), (220339, 5.)]:
            rows.append({"icustay_id": 1, "subject_id": 1, "hadm_id": 1,
                         "charttime": origin + pd.Timedelta(hours=h), "itemid": item,
                         "valuenum": value, "value": str(value), "valueuom": None, "error": 0})
    source = _source(tmp_path, "mimic", icustays=_stays("mimic", origin),
        chartevents=pd.DataFrame(rows), procedureevents_mv=pd.DataFrame({
            "icustay_id": [1], "subject_id": [1], "hadm_id": [1], "itemid": [225792],
            "starttime": [origin], "endtime": [origin + pd.Timedelta(hours=27)],
            "value": [27 * 60.], "valueuom": ["min"], "cancelreason": [0],
            "statusdescription": ["FinishedRunning"]}))
    # Supply the parent's other-vasopressor owner with a real, scoped order;
    # an absent table is not an observed negative cardiovascular component.
    pd.DataFrame({"icustay_id": [1], "subject_id": [1], "hadm_id": [1], "itemid": [222315],
        "starttime": [origin], "endtime": [origin + pd.Timedelta(hours=27)],
        "rate": [.03], "rateuom": ["units/min"], "amount": [48.6], "amountuom": ["units"],
        "linkorderid": [1], "orderid": [1], "patientweight": [70.],
        "statusdescription": ["FinishedRunning"], "cancelreason": [0]}).to_parquet(tmp_path / "inputevents_mv.parquet", index=False)
    for score, component, peak in [("sofa", "sofa_resp", 4), ("sofa2", "sofa2_resp", 3)]:
        point = _load(source, [component])[component].data
        assert point.loc[point.charttime.between(0, 26), component].tolist() == [peak] + [0] * 26
        parent = _load(source, [score], keep_components=True)[score].data
        # Other unavailable organs are outside this comparison; use the exact
        # published respiratory owner's values and its parent's retained column.
        assert parent.loc[parent.charttime.between(0, 26), component].tolist() == [peak] * 24 + [0] * 3


def _sofa2_source(tmp_path, database, layout):
    times = [0., 1., 3.]
    if database == "eicu":
        rows = [(h * 60, name, value) for name, values in {
            "paO2": [30., 30., 250.], "creatinine": [1., 1.3, 1.],
            "potassium": [6., 6., 4.], "bicarbonate": [24., 24., 24.],
        }.items() for h, value in zip(times, values)]
        lab = pd.DataFrame(rows, columns=["labresultoffset", "labname", "labresult"])
        lab["patientunitstayid"] = 1
        lab["labmeasurenameinterface"] = ""
        chart = pd.DataFrame({"patientunitstayid": 1,
            "respchartoffset": [0., 0., 60., 60., 180., 180.],
            "respchartvaluelabel": ["FiO2", "O2 Device"] * 3,
            "respchartvalue": ["50", "Nasal cannula", "50", "Ventilator", "50", "Nasal cannula"]})
        source = _source(tmp_path, database, lab=lab, respiratorycharting=chart,
            customlab=pd.DataFrame({"patientunitstayid": 1, "labotheroffset": [0., 60., 180.],
                "labothername": "Sample Type", "labothervaluetext": "Arterial", "labotherresult": np.nan}),
            patient=pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                "unitdischargeoffset": [600], "hospitaladmitoffset": [0], "admissionweight": [70.]}))
    else:
        rows = [(h * 3600, item, value) for item, values in {
            689: [30., 30., 250.], 367: [1., 1.3, 1.], 463: [6., 6., 4.], 456: [24., 24., 24.],
        }.items() for h, value in zip(times, values)]
        lab = pd.DataFrame(rows, columns=["Offset", "LaboratoryID", "LaboratoryValue"])
        lab["CaseID"], lab["LaboratoryType"] = 1, 2296
        source = _source(tmp_path, database, layout=layout, laboratory=lab,
            data_float_h=pd.DataFrame({"CaseID": 1, "Offset": [0., 3600., 10800.],
                "DataID": 2283, "Val": 50.}),
            data_range=pd.DataFrame({"CaseID": [1], "DataID": [720], "Offset": [3600.], "OffsetEnd": [10800.]}),
            cases=pd.DataFrame({"CaseID": [1], "PatientID": [1], "ICUOffset": [0.],
                "TimeOfStay": [36000.], "WeightOnAdmission": [70000.], "AdmissionYear": [2018],
                "HospitalDischargeType": [2027], "OffsetOfDeath": [np.nan],
                "EstimatedSurvivalObservationTime": [3077]}))
    return source


@pytest.mark.parametrize("database", ["eicu", "sic"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f040_public_owners_resolve_support_and_metabolic_inputs(tmp_path, database, layout):
    source = _sofa2_source(tmp_path, database, layout)
    times = [0., 1., 3.]
    for component, expected in [("sofa2_resp", [2, 4, 0]), ("sofa2_renal", [0, 4, 0])]:
        table = _load(source, [component])[component]
        keyed = table.data.set_index(table.index_column)
        assert keyed.loc[times, component].tolist() == expected
        assert keyed.loc[times, component + "_available"].eq(1).all()

    if database == "sic":
        combined = _load(source, ["mech_vent", "adv_resp", "sofa2_resp"])
        for name in ["mech_vent", "adv_resp"]:
            table = combined[name]
            assert table.data[table.index_column].tolist() == [1., 2.]
        assert combined["sofa2_resp"].data.set_index("charttime").loc[1., "sofa2_resp"] == 4


@pytest.mark.parametrize("layout", ["flat", "bucket"])
@pytest.mark.parametrize("settings", ["unrelated", "empty", "missing"])
def test_r6_f040_native_respiratory_retains_explicit_sic_airway(tmp_path, monkeypatch, layout, settings):
    import shutil
    import tempfile
    from easyicu.api.extraction import extract_database

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path / "duckdb"))
    source = _sofa2_source(tmp_path / "source", "sic", layout)
    table = (source.base_path / "data_float_h.parquet" if layout == "flat"
             else source.base_path / "data_float_h_bucket/bucket_id=0/part.parquet")
    if settings == "empty":
        pd.read_parquet(table).iloc[:0].to_parquet(table, index=False)
    elif settings == "missing":
        if layout == "flat":
            table.unlink()
        else:
            shutil.rmtree(source.base_path / "data_float_h_bucket")
    result = extract_database("sic", data_path=str(source.base_path), output_dir=str(tmp_path / "out"),
        modules=["outcome", "respiratory"], patient_ids={"CaseID": [1]}, native_export_v2=True, verbose=False)
    assert not result["modules"]["respiratory"].get("errors")
    published = pd.read_parquet(tmp_path / "out/respiratory.parquet").set_index("charttime")
    assert published.loc[[1., 2.], "vent_ind"].fillna(False).eq(True).all()  # noqa: E712
    assert published.loc[[1., 2.], "adv_resp"].fillna(False).eq(True).all()  # noqa: E712
    assert published.loc[[1., 2.], "mech_vent"].eq("invasive").fillna(False).all()
