"""Extraction audit round 4 (2026-11-04) regression tests."""

import numpy as np
import pandas as pd
import pytest


def _urine24(records, *, window=24, min_steps=18, cap=12):
    from easyicu.concept.callbacks import _urine24_batch

    df = pd.DataFrame(records, columns=["stay", "t", "urine"])
    out = _urine24_batch(df, "stay", "t", "urine", 1.0, window, min_steps, cap).set_index("t")
    # R5C-IV/III-003, R5-EICU-033, R5-SIC-024: estimates now have their own name.
    return out.urine24.combine_first(out.urine24_est)


def test_r4_001_uncharted_hours_are_missing_not_zero():
    # 100 ml every 4 h covers every hour: 24 h holds 600 ml.
    out = _urine24([(1, h, 100.0) for h in range(0, 49, 4)])
    assert out.loc[47.0] == pytest.approx(600.0)
    # A single 500 ml record after 3 uncharted hours covers 4 hours only.
    sparse = _urine24([(1, 0, 50.0), (1, 30, 500.0)])
    assert sparse.isna().all()  # the 30 h gap exceeds 12 h: not assessable


def test_r4_001_window_needs_18_covered_hours_and_scales_to_24():
    # Hours 0..19 covered at 10 ml/h, then nothing: 20 covered hours.
    out = _urine24([(1, 0, 10.0)] + [(1, h, 10.0) for h in range(1, 20)])
    assert np.isnan(out.loc[16.0])  # 17 covered hours < 18
    assert out.loc[19.0] == pytest.approx(240.0)  # 200 ml over 20 h -> 24 h


def test_r4_010_window_is_24_cells():
    # 1 ml every hour for 48 hours: a 24-cell window sums exactly 24 ml.
    out = _urine24([(1, h, 1.0) for h in range(48)])
    assert out.loc[30.0] == pytest.approx(24.0)


def test_r4_104_six_hour_grid_scales_by_cells_not_by_24():
    from easyicu.concept.callbacks import _urine24_batch

    df = pd.DataFrame({"stay": 1, "t": [0.0, 6.0, 12.0, 18.0, 24.0], "urine": 300.0})
    out = _urine24_batch(df, "stay", "t", "urine", 6.0, 4, 3, 2).set_index("t")["urine24"]
    assert out.loc[24.0] == pytest.approx(1200.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_r4_001_vectorized_kernel_matches_plain_reference(seed):
    from easyicu.concept.callbacks import _urine24_batch
    from tests.support.urine_window_reference import _urine24_batch as reference

    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(1, 6):
        t = np.cumsum(rng.choice([0.4, 1.0, 2.0, 5.0, 13.0], size=40)) - rng.uniform(0, 10)
        rows += [(stay, float(x), float(v)) for x, v in zip(t, rng.uniform(0, 300, size=40))]
    df = pd.DataFrame(rows, columns=["stay", "t", "urine"])
    got = _urine24_batch(df, "stay", "t", "urine", 1.0, 24, 18, 12).reset_index(drop=True)
    want = reference(df, "stay", "t", "urine", 1.0, 24, 18, 12)
    # R5C-IV/III-003: the retrospective numeric input is preserved where
    # no rejected >12 h interval splits the evaluation window.
    got["urine24"] = got.urine24.combine_first(got.urine24_est)
    comparable = got.urine24.notna()
    pd.testing.assert_frame_equal(got.loc[comparable, want.columns], want.loc[comparable],
                                  check_dtype=False, atol=1e-9)
    # The exact R4 coverage gate remains enforced, including windows dropped
    # after a rejected interval starts a new segment (R5C-HIRID-001).
    assert got.urine24_covered_h.between(0, 24).all()
    assert got.loc[got.urine24_covered_h < 18, "urine24"].isna().all()


def test_r4_001_kdigo_uo_does_not_spread_a_record_over_a_long_gap():
    from easyicu.scores.kdigo_aki import _calculate_uo_rates_simple

    # 50 ml/h hourly for 6 h, a 20 h charting gap, then a 100 ml record.
    hours = list(range(6)) + [25]
    urine = pd.DataFrame({"stay": 1, "t": hours, "urine": [50.0] * 6 + [100.0]})
    weight = pd.DataFrame({"stay": [1], "weight": [50.0]})
    out = _calculate_uo_rates_simple(urine, weight, "stay", "t", time_unit="hours")
    after_gap = out.loc[out["t"] == 25]
    # MIT-LCP spread 100 ml over 20 h (0.1 ml/kg/h, stage 3 at 24 h); the gap
    # is not assessable, so no window ending at 25 h can be evaluated.
    assert after_gap[["uo_rt_6hr", "uo_rt_12hr", "uo_rt_24hr"]].isna().all(axis=None)
    assert out.loc[out["t"] == 5, "uo_rt_6hr"].item() == pytest.approx(1.0)


def test_r4_061_kdigo_anuria_is_zero_output():
    from easyicu.scores.kdigo_aki import _calc_aki_stage_uo

    s = lambda v: pd.Series([v])  # noqa: E731
    assert _calc_aki_stage_uo(s(0.6), s(0.005), s(np.nan)).item() == 2
    assert _calc_aki_stage_uo(s(0.6), s(0.0), s(np.nan)).item() == 3


def _dict(name):
    import json
    from pathlib import Path

    import easyicu

    return json.loads((Path(easyicu.__file__).parent / "data" / name).read_text())


def test_r4_002_carevue_phenylephrine_is_another_vasopressor():
    for db in ("mimic", "mimic_demo"):
        cv = [s for s in _dict("sofa2-dict.json")["other_vaso"]["sources"][db] if s["table"] == "inputevents_cv"]
        ids = {i for s in cv for i in s["ids"]}
        assert {30051, 30125, 30127, 30128} <= ids
        phn = [s for s in cv if 30127 in s["ids"]]
        # R5C-III-012: positive rates now support observed infusion intervals.
        assert phn[0]["callback"] == "mimic_other_vaso"


@pytest.mark.parametrize("db", ["eicu", "eicu_demo"])
def test_r4_003_eicu_other_output_never_selects_a_urine_row(db):
    import re

    d = _dict("concept-dict.json")
    urine = d["urine"]["sources"][db][0]["regex"]
    exclude = d["other_output_ml"]["sources"][db][0]["exclude_regex"]["cellpath"]
    assert urine in exclude  # the urine selector is shared, not re-listed
    prefix = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"
    labels = ["Indwelling Catheter Output", "Urethral Catheter", "Voided Amount", "Ileal Conduit",
              "URINE CATHETER", "Urine Output-Foley", "Texas Cath", "Stool"]
    selected = re.compile(urine, re.IGNORECASE)  # how rgx_itm matches
    for label in labels:
        path = prefix + label
        assert label == "Stool" or selected.search(path)
        assert re.search(exclude, path), label
    for label in ["Stool (ml)", "Emesis", "Chest Tube #1", "NG tube"]:
        assert not re.search(exclude, prefix + label), label


def test_r4_005_aumc_ventilator_module_loads_whole(tmp_path):
    from easyicu.api import concepts as api
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.synthetic_sources import generate_source

    generate_source(tmp_path / "aumc", "aumc", stays=12)
    try:
        out = api.load_concepts(
            concepts=list(CONCEPT_GROUPS_INTERNAL["ventilator"]), database="aumc",
            data_path=str(tmp_path / "aumc"), verbose=False, concept_workers=1, parallel_workers=1,
        )
    finally:
        api.clear_global_loader()
    # plateau and PEEP arrive under different time-column names; the module
    # crashed before driving_pressure() aligned them (f19742099).
    assert out["driving_pres"].notna().any()
    assert out["peep"].notna().any() and out["plateau_pres"].notna().any()


def test_r4_006_mimic_delirium_proxy_keeps_prescription_points():
    from easyicu.utils.compat import MIXED_POINT_WINDOW_CONCEPTS

    # MetaVision rows have an end time; CareVue prescriptions do not and were
    # dropped by the window expansion.
    assert "delirium_tx_proxy" in MIXED_POINT_WINDOW_CONCEPTS


def test_r4_007_aumc_insulin_rate_covers_each_infusion_hour(tmp_path):
    import glob

    from easyicu.api import concepts as api
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "aumc"
    generate_source(root, "aumc", stays=12)
    path = [p for p in glob.glob(str(root / "drugitems*")) if p.endswith(".parquet")][0]
    drugs = pd.read_parquet(path)
    insulin = drugs.index[drugs.itemid.eq(7624) & drugs.admissionid.eq(3)]
    first, later = insulin[0], insulin[1:]
    drugs.loc[first, "stop"] = drugs.loc[first, "start"] + 10 * 3_600_000  # ms
    drugs.loc[first, "duration"] = 600
    drugs = drugs.drop(index=later)
    drugs.to_parquet(path, index=False)
    try:
        out = api.load_concepts(concepts=["ins"], database="aumc", data_path=str(root),
                                verbose=False, concept_workers=1, parallel_workers=1)
    finally:
        api.clear_global_loader()
    # A 10-hour infusion gives 10 hourly rows; a one-hour one gives one.
    assert out.loc[out.admissionid.eq(3), "ins"].notna().sum() == 10
    per_stay = out.loc[out.admissionid.ne(3)].groupby("admissionid")["ins"].count()
    assert per_stay.eq(2).all()  # two one-hour infusions each


def test_r4_056_aumc_spread_bag_stops_inside_its_last_hour():
    from easyicu.concept.callback_apply import _apply_callback
    from easyicu.concept.schema import ConceptSource

    # R5C-AUMC-004 supersedes stop stretching: 400 mL stays in the true hour.
    mapping = _dict("concept-dict.json")["dex"]["sources"]["aumc"][0]
    source = ConceptSource.from_mapping({**mapping, "val_var": "dose", "unit_var": "doseunit"})
    frame = pd.DataFrame({"admissionid": [1], "itemid": [7254], "start": [30.0], "stop": [50.0],
                          "dose": [1200.0], "doseunit": ["ml"], "doserateunit": ["uur"],
                          "iscontinuous": [0], "action": ["Toedienen"]})
    out = _apply_callback(frame, source, "dex", "doseunit")
    assert out["dose"].item() == pytest.approx(400.0)
    assert out["start"].item() == pytest.approx(0.0)


def _cvp(cvp_rows, dbp_rows, database="miiv"):
    from types import SimpleNamespace

    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_cvp_artefact_filter
    from easyicu.table import ICUTable

    def table(rows, name):
        frame = pd.DataFrame(rows, columns=["stay_id", "charttime", name])
        return ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column=name)

    ctx = ConceptCallbackContext(
        concept_name="cvp", target="ts_tbl", interval=pd.Timedelta(hours=1), resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name=database)), patient_ids=None,
    )
    out = _callback_cvp_artefact_filter({"cvp_raw": table(cvp_rows, "cvp_raw"), "dbp": table(dbp_rows, "dbp")}, ctx)
    data = out.data if hasattr(out, "data") else out
    return dict(zip(data["charttime"], data["cvp"]))


def test_r4_013_cvp_keeps_real_30_to_50_values():
    # A sustained 35-38 mmHg run with a diastolic of 60 is real congestion.
    out = _cvp([(1, h, v) for h, v in enumerate([20, 35, 36, 38, 37])], [(1, h, 60.0) for h in range(5)])
    assert out == {0: 20, 1: 35, 2: 36, 3: 38, 4: 37}


def test_r4_013_cvp_artefact_candidates_are_retained_under_r6_r2():
    rows = [(1, 0, 12), (1, 1, 45), (1, 2, 12), (1, 3, 40), (1, 4, 14), (1, 5, 13)]
    dbp = [(1, 3, 44.0)]
    out = _cvp(rows, dbp)
    # Hour 1: 45 between 12 and 12 is an isolated spike. Hour 3: 40 is within
    # 10 mmHg of the 44 mmHg diastolic arterial pressure.
    # R6-F008: these candidates are flagged, never hard-deleted within R2.
    assert out == {0: 12, 1: 45, 2: 12, 3: 40, 4: 14, 5: 13}


def test_r4_013_hirid_uses_the_shared_r6_r2_bounds():
    out = _cvp([(1, 0, 20), (1, 1, 35), (1, 2, 36)], [], database="hirid")
    assert out == {0: 20, 1: 35, 2: 36}


def test_r4_013_cvp_bounds_and_sicdb_source_cap():
    d = _dict("concept-dict.json")
    # R5-SIC-001: the SIC-specific dictionary lifts only its raw bound.
    assert d["cvp"]["max"] == 50 and d["cvp_raw"]["max"] == 50
    assert d["cvp"]["concepts"] == ["cvp_raw", "dbp"]
    assert d["cvp_raw"]["sources"]["sic"][0]["callback"] == "sicdb_cvp_raw"


def test_r4_029_hirid_short_episode_across_an_hour_is_not_one_hour():
    from easyicu.utils.callback_utils import hirid_duration

    frame = pd.DataFrame({
        "patientid": [1, 1, 2, 2], "infusionid": [10, 10, 20, 20],
        "givenat": [2 + 40 / 60, 3.5, 2 + 50 / 60, 17 + 5 / 60],
        "givendose": [0.0, 5.0, 0.0, 5.0],
        "recordstatus": 0,
    })
    out = hirid_duration(frame, concept_name="norepi_dur", val_col="givendose",
                         index_col="givenat", grp_var="infusionid").set_index("patientid")
    # R5C-HIRID-003: all episode lengths use the exact clock, including long episodes.
    assert out.loc[1, "norepi_dur"] == pytest.approx(50 / 60)
    assert out.loc[2, "norepi_dur"] == pytest.approx(14.25)


def test_r4_030_hirid_floating_point_residue_is_not_a_dose():
    from easyicu.utils.callback_utils import hirid_duration

    frame = pd.DataFrame({
        "patientid": [1, 1], "infusionid": [10, 10], "givenat": [1.0, 3.0], "givendose": [0.0, 1e-9],
        "recordstatus": 0,
    })
    out = hirid_duration(frame, concept_name="norepi_dur", val_col="givendose",
                         index_col="givenat", grp_var="infusionid")
    assert out.empty


@pytest.mark.parametrize("levels,expected", [
    (["invasive", "noninvasive"], [True]),
    ([None, None], []),  # a batch of unlevelled (no airway) windows only
])
def test_r4_014_022_vent_ind_takes_invasive_windows_only(levels, expected):
    from types import SimpleNamespace

    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vent_ind
    from easyicu.table import WinTbl

    mech = WinTbl(
        pd.DataFrame({"stay_id": [1, 2], "charttime": [1.0, 1.0], "dur_var": [3.0, 3.0],
                      "mech_vent": levels}),
        id_vars=["stay_id"], index_var="charttime", dur_var="dur_var", dur_unit="hours",
    )
    ctx = ConceptCallbackContext(
        concept_name="vent_ind", target="win_tbl", interval=pd.Timedelta(hours=1), resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="miiv")), patient_ids=None,
    )
    out = _callback_vent_ind({"mech_vent": mech}, ctx)
    rows = [] if out is None or out.data.empty else out.data.query("vent_ind == True")["stay_id"].tolist()
    assert rows == ([1] if expected else [])


def test_r4_019_mimic_vaso60_does_not_carry_a_down_titrated_rate():
    from types import SimpleNamespace

    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vaso60
    from easyicu.table import ICUTable

    def table(frame, value):
        return ICUTable(frame, id_columns=["icustay_id"], index_column="charttime", value_column=value)

    # 0.2 at 10:10, down-titrated to 0.05 at 10:50, charted again at 12:00.
    rate = table(pd.DataFrame({"icustay_id": 1, "charttime": [10 + 10 / 60, 10 + 50 / 60, 12.0],
                               "norepi_rate": [0.2, 0.05, 0.05]}), "norepi_rate")
    dur = table(pd.DataFrame({"icustay_id": [1], "charttime": [10 + 10 / 60], "norepi_dur": [4.0]}),
                "norepi_dur")
    ctx = ConceptCallbackContext(
        concept_name="norepi60", target=None, interval=pd.Timedelta(hours=1), resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="mimic")),
        patient_ids={"icustay_id": [1]},
    )
    out = _callback_vaso60({"norepi_rate": rate, "norepi_dur": dur}, ctx).data
    by_hour = dict(zip(out["charttime"], out["norepi60"]))
    # The 10:50 rate runs into 11:00; the 10:10 rate stopped at 10:50.
    assert by_hour[10.0] == pytest.approx(0.2)
    assert by_hour[11.0] == pytest.approx(0.05)


def test_r4_020_021_mimic_gas_without_a_specimen_record_is_not_arterial():
    from easyicu.concept.blood_gas_specimen import SPECIMEN_RULES

    for db in ("mimic", "mimic_demo", "miiv"):
        assert SPECIMEN_RULES[db].keep_unrecorded is False


def test_r4_024_eicu_shared_settings_take_the_stays_ventilator_kind():
    from easyicu.concept.eicu_ventilation_windows import setting_evidence

    charting = pd.DataFrame({
        "patientunitstayid": [1, 1, 2, 2, 3],
        "respchartoffset": [0, 30, 0, 30, 0],
        # Stay 1: BiPAP with PEEP; stay 2: set tidal volume with PEEP; stay 3:
        # pressure support alone.
        "respchartvaluelabel": ["B1: IPAP", "PEEP", "Tidal Volume (set)", "PEEP", "Pressure Support"],
        "respchartvalue": ["12", "5", "450", "5", "10"],
    })
    kept = setting_evidence(charting, label_column="respchartvaluelabel", value_column="respchartvalue",
                            id_column="patientunitstayid", time_column="respchartoffset")
    levels = {(r.patientunitstayid, r.respchartvaluelabel): r.respchartvalue for r in kept.itertuples()}
    assert levels == {
        (1, "B1: IPAP"): "noninvasive", (1, "PEEP"): "noninvasive",
        (2, "Tidal Volume (set)"): "invasive", (2, "PEEP"): "invasive",
    }


def test_r4_027_aumc_weight_falls_back_to_measured_weight():
    sources = _dict("concept-dict.json")["weight"]["sources"]["aumc"]
    channels = [s.get("source_channel") for s in sources]
    assert channels == ["admission", "charted"]
    assert set(sources[1]["ids"]) == {20081, 6638, 6637, 10442}
    priority = _dict("concept-dict.json")["weight"]["source_channel_priority"]
    assert priority.index("admission") < priority.index("charted")


def test_r4_028_hirid_dose_is_split_over_the_hours_it_was_given_in():
    from easyicu.utils.callback_utils import hirid_duration, hirid_rate_kg

    frame = pd.DataFrame({
        "patientid": 1, "infusionid": 7, "doseunit": "µg", "weight": 50.0,
        # Start marker 10:50, 25 ug by 11:05, 60 ug by 12:05.
        "givenat": [10 + 50 / 60, 11 + 5 / 60, 12 + 5 / 60], "givendose": [0.0, 25.0, 60.0],
        "recordstatus": 0,
    })
    rate = hirid_rate_kg(frame, concept_name="norepi_rate", val_col="givendose", unit_col="doseunit",
                         grp_var="infusionid", index_col="givenat")
    mass = dict(zip(rate["givenat"], rate["norepi_rate"] * 60 * 50))
    assert mass == pytest.approx({10: 25 * 10 / 15, 11: 25 * 5 / 15 + 60 * 55 / 60, 12: 60 * 5 / 60})
    assert sum(mass.values()) == pytest.approx(85.0)  # the dose is conserved
    dur = hirid_duration(frame, concept_name="norepi_dur", val_col="givendose",
                         index_col="givenat", grp_var="infusionid")
    # R5C-HIRID-003: preserve the true start separately from its occupied hour.
    assert dur["givenat"].tolist() == pytest.approx([10 + 50 / 60])


def test_r4_016_chart_transcription_within_6h_of_a_lab_result_is_dropped():
    from easyicu.concept.source_channel import RANK_COLUMN, select_preferred_channel

    frame = pd.DataFrame({
        "icustay_id": [1, 1, 1, 2],
        "charttime": [10.0, 13.0, 30.0, 13.0],
        "crea": [1.2, 1.2, 1.5, 0.9],
        RANK_COLUMN: [0, 1, 1, 1],  # lab, chart, chart, chart
    })
    out = select_preferred_channel(frame, value_column="crea", id_columns=["icustay_id"],
                                   time_column="charttime", window_hours=6)
    # The 13 h copy of the 10 h lab result goes; a chart value 20 h later and
    # one of a stay without labs stay.
    assert list(zip(out.icustay_id, out.charttime)) == [(1, 10.0), (1, 30.0), (2, 13.0)]


def test_r4_016_mimic_lab_channels_are_declared():
    d = _dict("concept-dict.json")
    for name in ("crea", "hgb", "plt", "wbc", "ph", "po2", "pco2", "sao2"):
        assert d[name]["source_channel_priority"] == ["lab", "chart"]
        assert d[name]["source_channel_window_hours"] == 6
        for src in d[name]["sources"]["mimic"]:
            assert src["source_channel"] == ("lab" if src["table"] == "labevents" else "chart")


def test_r4_031_hirid_vent_ind_bridges_ventilator_evidence_from_an_airway():
    from types import SimpleNamespace

    from easyicu.concept.callbacks import _hirid_bridge_ventilator_evidence
    from easyicu.table import ICUTable

    def table(rows, name):
        return ICUTable(pd.DataFrame(rows, columns=["patientid", "charttime", name]),
                        id_columns=["patientid"], index_column="charttime", value_column=name)

    evidence = {
        # stay 1: PEEP with etCO2 at 5 h (4 h after the airway) and at 30 h;
        # stay 2: PEEP and etCO2 without any airway record.
        "peep": table([(1, 5.0, 8.0), (1, 30.0, 8.0), (2, 3.0, 8.0)], "peep"),
        "etco2": table([(1, 5.0, 35.0), (1, 30.0, 35.0), (2, 3.0, 35.0)], "etco2"),
        "tidal_vol": table([], "tidal_vol"),
        "vent_mode": table([], "vent_mode"),  # R5C-HIRID-004: explicit stop evidence
    }
    resolver = SimpleNamespace(load_concepts=lambda names, ds, **kw: {n: evidence[n] for n in names})
    ctx = SimpleNamespace(resolver=resolver, data_source=None, patient_ids=None)
    airway = table([(1, 0.0, True), (1, 1.0, True)], "vent_ind")
    out = _hirid_bridge_ventilator_evidence(airway, ctx, "patientid", "charttime").data
    assert list(zip(out.patientid, out.charttime)) == [(1, float(h)) for h in range(6)]


def test_r4_033_036_distribute_amount_conserves_the_amount():
    from easyicu.utils.callback_utils import distribute_amount

    data = pd.DataFrame({
        "stay_id": [1, 2], "time": [10.5, 2.0], "endtime": [10.5, 6.0],
        "value": [10.0, 20.0], "unit": ["units", "units"],
    })
    out = distribute_amount(data, val_col="value", unit_col="unit", end_col="endtime", index_col="time")
    push = out[out.stay_id == 1]
    assert push["time"].tolist() == [10] and push["value"].tolist() == [10.0]
    drip = out[out.stay_id == 2]
    assert drip["time"].tolist() == [2, 3, 4, 5]
    assert drip["value"].sum() == pytest.approx(20.0)


def test_r4_043_eicu_coded_gcs_item_scores_are_parsed():
    from easyicu.concept.callback_apply import _eicu_gcs_score

    values = pd.Series(["4-->(E4) spontaneous", "3", "2-->(E3) to pain", "5-->(M5) localizes",
                        "1-->", "Unable to score due to medication", "7-->(E7) x"])
    out = _eicu_gcs_score(values, "E")
    assert out.tolist()[:2] == [4.0, 3.0]
    assert out.iloc[2:4].isna().all()  # score disagrees with the code / wrong item
    assert out.iloc[4] == 1.0
    assert out.iloc[5:].isna().all()


@pytest.mark.parametrize("db", ["eicu", "eicu_demo"])
def test_r4_043_048_055_eicu_coded_gcs_rows_reach_the_concepts(tmp_path, db):
    from pathlib import Path

    import easyicu
    from easyicu.concept import ConceptDictionary, ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    tmp_path.joinpath("nursecharting").mkdir()
    pd.DataFrame({
        "patientunitstayid": [1], "patienthealthsystemstayid": [11], "uniquepid": ["p1"],
        "age": ["60"], "gender": ["Male"], "hospitalid": [1], "hospitaladmitoffset": [0],
        "hospitaldischargeoffset": [72 * 60], "unitdischargeoffset": [48 * 60],
        "unitdischargestatus": ["Alive"], "hospitaldischargestatus": ["Alive"], "admissionweight": [80.0],
    }).to_parquet(tmp_path / "patient.parquet", index=False)
    rows = [
        (5, "Best Eye Response", "4-->(E4) spontaneous"),
        (5, "Best Motor Response", "6-->(M6) obeys commands"),
        (5, "Best Verbal Response", "5-->(V5) oriented"),
        (65, "Best Eye Response", "2"),
        (65, "Best Motor Response", "3-->(M4) withdraws"),
        (65, "Best Verbal Response", "1-->(V1) none"),
    ]
    frame = pd.DataFrame(rows, columns=["nursingchartoffset", "nursingchartcelltypevallabel", "nursingchartvalue"])
    frame["patientunitstayid"] = 1
    frame["nursingchartentryoffset"] = frame["nursingchartoffset"]
    frame["nursingchartcelltypecat"] = "Scores"
    frame["nursingchartcelltypevalname"] = "Value"
    frame["nursingchartid"] = np.arange(len(frame), dtype=float)
    frame.to_parquet(tmp_path / "nursecharting/part.parquet", index=False)
    source = ICUDataSource(load_src_cfg(db), base_path=tmp_path, enable_cache=False)
    loaded = ConceptResolver(ConceptDictionary.from_json(Path(easyicu.__file__).parent / "data/concept-dict.json")).load_concepts(
        ["egcs", "mgcs", "vgcs"], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": [1]},
        concept_workers=1, verbose=False,
    )

    def hourly(name):
        data = loaded[name].data
        time = data[loaded[name].index_column]
        if pd.api.types.is_timedelta64_dtype(time):
            time = time / pd.Timedelta(hours=1)
        return {float(t): float(v) for t, v in zip(time, data[name]) if pd.notna(v)}

    assert hourly("egcs") == {0.0: 4.0, 1.0: 2.0}
    assert hourly("mgcs") == {0.0: 6.0}  # 3-->(M4) disagrees with its own code
    assert hourly("vgcs") == {0.0: 5.0, 1.0: 1.0}


@pytest.mark.parametrize("db", ["mimic", "mimic_demo"])
def test_r4_034_mimic3_admission_type_is_the_service_active_at_icu_intime(tmp_path, db):
    from pathlib import Path

    import easyicu
    from easyicu.concept import ConceptDictionary, ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    pd.DataFrame({
        "row_id": [1, 2], "subject_id": [100, 100], "hadm_id": [10, 10], "icustay_id": [1, 2],
        "dbsource": "metavision", "first_careunit": "MICU", "last_careunit": "MICU",
        "first_wardid": 1, "last_wardid": 1,
        "intime": pd.to_datetime(["2150-01-05", "2150-01-20"]),
        "outtime": pd.to_datetime(["2150-01-07", "2150-01-22"]), "los": 2.0,
    }).to_parquet(tmp_path / "icustays.parquet", index=False)
    pd.DataFrame({
        "row_id": [1, 2], "subject_id": 100, "hadm_id": 10,
        # Chosen four days before the first stay; changed between the stays.
        "transfertime": pd.to_datetime(["2150-01-01 08:00", "2150-01-12 09:00"]),
        "prev_service": [None, "MED"], "curr_service": ["MED", "CSURG"],
    }).to_parquet(tmp_path / "services.parquet", index=False)
    source = ICUDataSource(load_src_cfg(db), base_path=tmp_path, enable_cache=False)
    dictionary = ConceptDictionary.from_json(Path(easyicu.__file__).parent / "data/concept-dict.json")
    loaded = ConceptResolver(dictionary).load_concepts(
        ["adm"], source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
        patient_ids={"icustay_id": [1, 2]}, concept_workers=1, verbose=False,
    )["adm"].data
    assert loaded.set_index("icustay_id")["adm"].to_dict() == {1: "med", 2: "surg"}


def test_r4_046_eicu_fentanyl_weight_indexed_rates_become_mass_rates():
    from easyicu.utils.callback_utils import eicu_rate_mass_callback

    frame = pd.DataFrame({
        "patientunitstayid": [1, 1, 1, 2, 3],
        "infusionoffset": [0, 120, 240, 0, 0],
        "drugname": ["Fentanyl (mcg/kg/hr)", "Fentanyl (mcg/kg/min)", "Fentanyl (mg/hr)",
                     "Fentanyl (mg/hr)", "Fentanyl (mcg/kg/hr)"],
        "drugrate": ["1", "0.02", "50", "0.1", "1"],
        "patientweight": [80.0, 80.0, 80.0, 70.0, None],
    })
    out = eicu_rate_mass_callback("mcg/hour", mg_label_mcg_above=10)(
        frame, val_var="drugrate", sub_var="drugname", concept_name="fentanyl_rate",
    )
    got = {(s, t): v for s, t, v in zip(out["patientunitstayid"], out["infusionoffset"], out["fentanyl_rate"])}
    assert got[(1, 0)] == pytest.approx(80.0)  # 1 mcg/kg/hr x 80 kg
    assert got[(1, 120)] == pytest.approx(96.0)  # 0.02 mcg/kg/min x 80 kg x 60
    assert got[(1, 240)] == pytest.approx(50.0)  # '(mg/hr)' 50 is a mcg/hr rate
    assert got[(2, 0)] == pytest.approx(100.0)  # 0.1 mg/hr is a real mg rate
    assert 3 not in set(out["patientunitstayid"])  # no weight: not a dose


def _filtered(spec, frame, concept=None):
    import types

    from easyicu.concept.callback_apply import _drop_declared_row_exclusions

    source = types.SimpleNamespace(params=spec, value_var=spec.get("val_var"))
    return _drop_declared_row_exclusions(frame, source, concept)


def _src(d, concept, db, **match):
    return next(s for s in d[concept]["sources"][db] if all(s.get(k) == v for k, v in match.items()))


def test_r4_032_mimic3_d_dimer_is_on_the_miiv_feu_scale():
    d = _dict("concept-dict.json")
    for db in ("miiv", "mimic", "mimic_demo"):
        assert all("callback" not in s for s in d["d_dimer"]["sources"][db])
    assert "FEU" in d["d_dimer"]["_provenance"] and "FEU / 2" not in d["d_dimer"]["_provenance"]


def test_r4_035_or_pacu_intake_is_not_an_icu_transfusion():
    d = _dict("concept-dict.json")
    or_pacu = {226367, 226368, 226369, 227070, 227071, 227072, 30103, 30104, 30105}
    for concept, kept in (("ffp", 30005), ("packed_rbc", 225168), ("platelets", 30006)):
        for db in ("mimic", "mimic_demo", "miiv"):
            ids = {i for s in d[concept]["sources"][db] for i in s.get("ids", [])}
            assert not ids & or_pacu
        assert kept in {i for s in d[concept]["sources"]["mimic"] for i in s.get("ids", [])}


def test_r4_037_057_058_059_iv_concepts_drop_enteral_orders():
    d = _dict("concept-dict.json")
    rx = _src(d, "levetiracetam", "mimic", table="prescriptions")
    routes = pd.DataFrame({"route": ["IV", "IV DRIP", "IVPB", "PO", "PO/NG", "NG", "ORAL", "J TUBE", None]})
    assert _filtered(rx, routes)["route"].tolist() == ["IV", "IV DRIP", "IVPB"]
    categories = pd.DataFrame({"ordercategory": [
        "Injecties CZS/Sedatie/Analgetica", "Niet iv CZS/Sedatie/Analgetica", "Niet iv Tractus Digestivus",
        "Infuus - Crystalloid", "Spuitpompen"]})
    for concept in ("levetiracetam", "pantoprazole"):
        kept = _filtered(d[concept]["sources"]["aumc"][0], categories)["ordercategory"].tolist()
        assert kept == ["Injecties CZS/Sedatie/Analgetica", "Infuus - Crystalloid", "Spuitpompen"]
    assert d["levetiracetam"]["sources"]["hirid"][0]["ids"] == [1000756, 1001175]


def test_r4_038_eicu_meropenem_is_an_antibiotic():
    import re

    d = _dict("concept-dict.json")
    for db in ("eicu", "eicu_demo"):
        for source in d["abx"]["sources"][db]:
            if source.get("class_name") == "rgx_itm":
                assert re.search(source["regex"], "MEROPENEM 1 G IV SOLR", re.IGNORECASE)
                assert re.search(source["regex"], "Merrem", re.IGNORECASE)
        hicl = _src(d, "abx", db, sub_var="drughiclseqno")
        assert 11254 in hicl["ids"]
        named = pd.DataFrame({"drugname": ["MEROPENEM 1 G IV SOLR", None, "warfarin"], "routeadmin": "IV"})
        assert _filtered(hicl, named)["drugname"].isna().tolist() == [True, False]


def test_r4_041_eicu_delirium_scale_score_yes_no_is_cam_icu():
    from easyicu.concept.callback_apply import apply_callback

    d = _dict("sofa2-dict.json")
    spec = _src(d, "delirium_positive", "eicu", regex="^Delirium Scale/Score$")

    class _Source:
        callback = spec["callback"]
        params = spec
        value_var = spec["val_var"]
        sub_var = spec["sub_var"]
        index_var = None
        unit_var = None

    frame = pd.DataFrame({"nursingchartvalue": ["Yes", "No", "NO", "CAM-ICU", "ICDSC", "3"]})
    out = apply_callback(frame, _Source(), "delirium_positive")
    values = out["nursingchartvalue"].tolist()
    # Scale names and ICDSC scores are not results: missing, never text.
    assert values[:3] == [True, False, False] and all(pd.isna(v) for v in values[3:])
    assert "YES" in d["delirium_positive"]["sources"]["eicu"][0]["callback"]


def test_r4_042_eicu_haloperidol_and_olanzapine_orders_are_delirium_proxy_candidates():
    import re

    d = _dict("sofa2-dict.json")
    name = d["delirium_tx_proxy"]["sources"]["eicu"][0]
    assert re.search(name["regex"], "HALDOL 5 MG/ML INJ", re.IGNORECASE)
    assert {1662, 11814} <= set(d["delirium_tx_proxy"]["sources"]["eicu"][1]["ids"])


def test_r4_045_050_eicu_opioids_keep_iv_orders_only():
    d = _dict("concept-dict.json")
    fentanyl = _src(d, "fentanyl", "eicu", table="medication", sub_var="drugname")
    orders = pd.DataFrame({
        "drugname": ["FENTANYL CITRATE 0.05 MG/ML IJ SOLN", "fentanyl 25 mcg/hr patch", "Fentanyl",
                     "fentanyl/bupivacaine epidural", "Fentanyl", "Fentanyl"],
        "routeadmin": ["IV PUSH", "TD", "Transdermal", "EPIDURAL", "Intranasal", "IV"],
    })
    kept = _filtered(fentanyl, orders, "fentanyl")
    assert kept["routeadmin"].tolist() == ["IV PUSH", "IV"]
    drip = _src(d, "fentanyl", "eicu", table="infusiondrug")
    mixes = pd.DataFrame({"drugname": ["Fentanyl (mcg/hr)", "Fentanyl/Bupivacaine (ml/hr)"]})
    assert _filtered(drip, mixes, "fentanyl")["drugname"].tolist() == ["Fentanyl (mcg/hr)"]
    morphine = _src(d, "morphine", "eicu", table="medication", sub_var="drugname")
    orders = pd.DataFrame({
        "drugname": ["MORPHINE SULFATE 2 MG/ML IJ SOLN", "MORPHINE SULFATE ER 15 MG PO TBCR", "morphine", "morphine"],
        "routeadmin": ["IV", "PO", "Per NG tube", "IV Push"],
    })
    assert _filtered(morphine, orders, "morphine")["routeadmin"].tolist() == ["IV", "IV Push"]


def test_r4_053_eicu_potassium_hicl_549_551_by_iv_route():
    d = _dict("concept-dict.json")
    hicl = _src(d, "potassium_iv", "eicu", sub_var="drughiclseqno")
    assert {549, 551} <= set(hicl["ids"])
    rows = pd.DataFrame({"drugname": [None, None, "KCL 20 MEQ PO"], "routeadmin": ["IVPB", "PO", "IV"]})
    assert _filtered(hicl, rows)["routeadmin"].tolist() == ["IVPB"]


def test_r4_060_sicdb_antibiotics_are_systemic():
    ids = set(_dict("concept-dict.json")["abx"]["sources"]["sic"][0]["ids"])
    assert {1451, 1439, 1462, 1455, 1628, 1813, 1605, 1446, 1948, 1606, 1693, 1819, 1997, 1441} <= ids
    assert not ids & {1927, 1809, 1486, 1543, 1942, 1925, 1924}


def test_r4_039_batch_lab_path_keeps_declared_row_filters(tmp_path):
    # eICU lab read through the batch multi-concept loader (flat parquet
    # directory): a basos row reported in K/uL is an absolute count.
    from easyicu.concept import ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_dictionary

    pd.DataFrame({
        "patientunitstayid": [1], "patienthealthsystemstayid": [11], "uniquepid": ["p1"],
        "age": ["60"], "gender": ["Male"], "hospitalid": [1], "hospitaladmitoffset": [0],
        "hospitaldischargeoffset": [72 * 60], "unitdischargeoffset": [48 * 60],
        "unitdischargestatus": ["Alive"], "hospitaldischargestatus": ["Alive"], "admissionweight": [80.0],
    }).to_parquet(tmp_path / "patient.parquet", index=False)
    (tmp_path / "lab").mkdir()
    pd.DataFrame({
        "labid": [1, 2, 3, 4], "patientunitstayid": 1, "labresultoffset": [10, 70, 130, 130], "labtypeid": 4,
        "labname": ["-basos", "-basos", "-lymphs", "Hgb"], "labresult": [1.0, 0.05, 20.0, 9.0],
        "labresulttext": ["1", "0.05", "20", "9"], "labmeasurenamesystem": ["%", "K/mcL", "%", "g/dL"],
        "labmeasurenameinterface": ["%", "K/uL", "%", "g/dL"], "labresultrevisedoffset": [10, 70, 130, 130],
    }).to_parquet(tmp_path / "lab" / "part.parquet", index=False)
    source = ICUDataSource(load_src_cfg("eicu"), base_path=tmp_path, enable_cache=False)
    loaded = ConceptResolver(load_dictionary(source_name="eicu")).load_concepts(
        ["basos", "lymph", "hgb"], source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
        patient_ids={"patientunitstayid": [1]}, concept_workers=1, verbose=False,
    )
    assert loaded["basos"].data["basos"].tolist() == [1.0]
    assert loaded["lymph"].data["lymph"].tolist() == [20.0]
    assert loaded["hgb"].data["hgb"].tolist() == [9.0]


@pytest.mark.parametrize("score,prefix,callback", [("sofa", "sofa", "sofa_score"), ("sofa2", "sofa2", "sofa2_score")])
def test_r4_009_sofa_worst_value_window_is_24_hourly_cells(score, prefix, callback):
    from easyicu.concept import ConceptResolver
    from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
    from easyicu.config import DataSourceConfig
    from easyicu.table import ICUTable

    components = [f"{prefix}_{organ}" for organ in ("resp", "coag", "liver", "cardio", "cns", "renal")]
    frame = pd.DataFrame({"stay_id": 1, "charttime": range(30), **{c: [0.0] * 30 for c in components}})
    frame.loc[0, f"{prefix}_cns"] = 4.0  # one bad hour at t=0
    definitions = {c: ConceptDefinition(name=c, sources={"unit": [ConceptSource(table=c, value_var="value")]})
                   for c in components}
    definitions[score] = ConceptDefinition(name=score, sources={}, sub_concepts=components, callback=callback)

    class Source:
        config = DataSourceConfig(name="unit", tables={
            c: {"defaults": {"id_var": "stay_id", "index_var": "charttime", "val_var": "value"}} for c in components})

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            data = frame[["stay_id", "charttime", table_name]].rename(columns={table_name: "value"})
            return ICUTable(data.copy(), id_columns=["stay_id"], index_column="charttime", value_column="value")

        def clear_cache(self):
            pass

    out = ConceptResolver(ConceptDictionary(definitions)).load_concepts(
        [score], Source(), merge=True, interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1,
    )
    by_hour = out.set_index("charttime")[score]
    assert by_hour.loc[23] == 4  # hours 0..23: 24 cells include t=0
    assert by_hour.loc[24] == 0  # hours 1..24 no longer do


def test_r4_011_015_adv_resp_o2_device_labels():
    import re

    d = _dict("sofa2-dict.json")
    for db in ("miiv", "mimic", "mimic_demo"):
        device = _src(d, "adv_resp", db, ids=[226732])
        hit = lambda label: re.search(device["regex"], label, re.IGNORECASE) is not None  # noqa: E731
        assert hit("High flow nasal cannula") and hit("Bipap mask") and hit("Endotracheal tube")
        assert not hit("High flow neb") and not hit("Tracheostomy tube") and not hit("Trach mask")


def test_r4_026_aumc_tracheostoma_is_not_a_ventilation_episode():
    d = _dict("concept-dict.json")
    source = next(s for s in d["mech_vent"]["sources"]["aumc"] if 9328 in s.get("ids", []))
    assert 12635 not in source["ids"] and "Tracheostoma" not in source["callback"]


def _eicu_tables(tmp_path, **tables):
    pd.DataFrame({
        "patientunitstayid": [1], "patienthealthsystemstayid": [11], "uniquepid": ["p1"],
        "age": ["60"], "gender": ["Male"], "hospitalid": [1], "hospitaladmitoffset": [0],
        "hospitaldischargeoffset": [72 * 60], "unitdischargeoffset": [48 * 60],
        "unitdischargestatus": ["Alive"], "hospitaldischargestatus": ["Alive"], "admissionweight": [80.0],
    }).to_parquet(tmp_path / "patient.parquet", index=False)
    for name, frame in tables.items():
        frame = frame.copy()
        frame["patientunitstayid"] = 1
        (tmp_path / name).mkdir()
        frame.to_parquet(tmp_path / name / "part.parquet", index=False)


def test_r4_152_eicu_fahrenheit_temperatures_are_converted(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_dictionary

    vital = pd.DataFrame({"vitalperiodicid": [1, 2, 3], "observationoffset": [5, 65, 125],
                          "temperature": [37.0, 98.6, 120.0]})
    nurse = pd.DataFrame({
        "nursingchartid": [1.0], "nursingchartoffset": [185], "nursingchartentryoffset": [185],
        "nursingchartcelltypecat": ["Vital Signs"], "nursingchartcelltypevallabel": ["Temperature"],
        "nursingchartcelltypevalname": ["Temperature (C)"], "nursingchartvalue": ["101.3"],
    })
    _eicu_tables(tmp_path, vitalperiodic=vital, nursecharting=nurse)
    source = ICUDataSource(load_src_cfg("eicu"), base_path=tmp_path, enable_cache=False)
    loaded = ConceptResolver(load_dictionary(source_name="eicu")).load_concepts(
        ["temp"], source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
        patient_ids={"patientunitstayid": [1]}, concept_workers=1, verbose=False,
    )["temp"]
    got = dict(zip(loaded.data[loaded.index_column], loaded.data["temp"].round(2)))
    assert got == {0.0: 37.0, 1.0: 37.0, 3.0: 38.5}  # 120 is neither scale


def test_r4_l_items_dictionary_contracts():
    import re

    d, s2 = _dict("concept-dict.json"), _dict("sofa2-dict.json")

    def excluded(spec, column, value):
        return re.search(spec["exclude_regex"][column], value) is not None

    # R4-068 / R4-072 / R4-085 / R4-113: inactive device states are not support
    for concept in ("vent_mode", "vent_breath_seq"):
        assert excluded(d[concept]["sources"]["miiv"][0], "value", "Ambient")
    assert excluded(_src(s2, "adv_resp", "mimic", table="chartevents", ids=[223848, 223849, 220339, 224700, 225949]), "value", "Not applicable")
    # R5C-AUMC-016: original-time device arbitration owns standby now.
    assert any(s.get("callback") == "aumc_respiratory_modes" for s in s2["adv_resp"]["sources"]["aumc"])
    for db in ("miiv", "mimic"):
        mcs = next(s for s in s2["mech_circ_support"]["sources"][db] if 227980 in s.get("ids", []))
        assert excluded(mcs, "value", "Off") and not excluded(mcs, "value", "1:1")
    # R4-074 / R4-079 / R4-081 / R4-126 / R4-153: unit channels that are not the concept
    assert excluded(d["cvp_raw"]["sources"]["mimic"][0], "valueuom", "%")
    assert excluded(_src(d, "fio2", "eicu", table="lab"), "labmeasurenameinterface", "lpm")
    assert excluded(_src(d, "pip", "eicu", table="lab"), "labmeasurenameinterface", "cwp")
    assert excluded(d["ca"]["sources"]["eicu"][0], "labmeasurenameinterface", "mmol/L")
    assert excluded(d["tibc"]["sources"]["eicu"][0], "labmeasurenameinterface", "ng/dL")
    # R4-088: HiRID boolean drugs need a dose above numerical noise
    hirid = [s for c in d.values() for s in (c.get("sources") or {}).get("hirid", []) if s.get("table") == "pharma"]
    assert not any(s.get("callback") == "transform_fun(comp_na(`>`, 0))" for s in hirid)
    # R4-093 / R4-094 / R4-097: SICdb values
    assert _src(d, "fio2", "sic", ids=2283)["callback"] == "transform_fun(percent_as_numeric)"
    for concept in ("furosemide", "loop_diuretic"):
        assert d[concept]["sources"]["sic"][0]["callback"] == "transform_fun(comp_na(`>`, 0))"
    for dictionary in (d, s2):
        assert dictionary["rrt"]["sources"]["sic"][0]["callback"] == "transform_fun(comp_na(`>`, 0))"
    # dead sources and items removed (R4-096/098/109/114/122/143/147/161/168/171-181)
    ids = lambda c, db: {i for s in d[c]["sources"][db] for i in (s.get("ids") if isinstance(s.get("ids"), list) else [s.get("ids")])}  # noqa: E731
    assert 184 not in ids("o2sat", "sic") | ids("sao2", "sic")
    assert 451 not in ids("bicar", "sic") and 454 not in ids("lact", "sic") and 1871 not in ids("cisatracurium", "sic")
    assert 1686 not in ids("thiazide_diuretic", "sic") | ids("potassium_sparing_diuretic", "sic")
    assert 2086 in ids("potassium_iv", "sic")
    assert 229617 not in ids("epi_rate", "miiv") and 224027 not in ids("temp", "miiv")
    assert not {650, 1883, 3049} & ids("minute_vol", "mimic")
    assert "Minute Volume, Spontaneous" not in ids("minute_vol", "eicu")
    assert 1000932 not in ids("morphine", "hirid") and 9113 not in ids("nitroglycerin", "aumc")
    assert 6815 not in ids("cai", "aumc") and "aumc" not in d["sedated_gcs"]["sources"]
    # R4-107 / R4-117 / R4-123 / R4-125..151: non-systemic and non-IV routes
    assert excluded(_src(d, "abx", "miiv", table="prescriptions"), "route", "OU")
    assert excluded(_src(d, "vancomycin", "mimic", table="prescriptions"), "route", "IRR")
    abx = _src(d, "abx", "eicu", table="medication", sub_var="drughiclseqno")
    assert excluded(abx, "routeadmin", "Ophthalmic") and not excluded(abx, "routeadmin", "IM")
    for concept in ("bicarbonate", "calcium_iv", "dextrose50", "magnesium_iv", "midazolam", "nicardipine", "pantoprazole", "potassium_iv"):
        for spec in (s for s in d[concept]["sources"]["eicu"] if s["table"] == "medication"):
            assert excluded(spec, "routeadmin", "PO") and excluded(spec, "routeadmin", "Topical"), concept
            assert not excluded(spec, "routeadmin", "IV Push"), concept
    assert excluded(_src(d, "heparin", "eicu", table="medication", sub_var="drugname"), "drugname", "HEPARIN LOCK FLUSH")
    for concept in ("amiodarone", "calcium_iv", "morphine"):
        assert excluded(d[concept]["sources"]["aumc"][0], "ordercategory", "Niet iv CZS/Sedatie/Analgetica")
    for concept in ("fentanyl", "potassium_iv"):
        spec = _src(d, concept, "hirid", table="pharma")
        assert excluded(spec, "route", "enteral") and excluded(spec, "route", "p os") and not excluded(spec, "route", "iv-inf")
    # R4-089 / R4-154 / R4-169: descriptions carry no other database's counts
    for concept in ("adv_resp", "ecmo", "mech_circ_support", "other_vaso", "delirium_positive"):
        assert "validated" not in s2[concept]["description"] and not re.search(r"\d{3,}", s2[concept]["description"])
    assert d["icp"]["min"] == -10 and d["plt"]["min"] == 1  # R4-112 / R4-158
    assert (d["hbco"]["min"], d["hbco"]["max"]) == (0, 100)  # R4-135 / R4-174
    # Pooled TRUE/FALSE rows: an hour is positive when any row is (a median gave 0.5).
    assert s2["delirium_positive"]["aggregate"] == "max" and s2["rrt"]["aggregate"] == d["rrt"]["aggregate"] == "max"


def test_r4_110_miiv_gcs_does_not_read_a_verbal_score_of_1_as_intubation():
    import types

    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_gcs
    from easyicu.table import ICUTable

    def table(name, values):
        frame = pd.DataFrame({"stay_id": 1, "charttime": [0.0, 1.0], name: values})
        return ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column=name)

    source = types.SimpleNamespace(config=types.SimpleNamespace(name="miiv"))
    ctx = ConceptCallbackContext(concept_name="gcs", target=None, interval=pd.Timedelta(hours=1),
                                 resolver=None, data_source=source, patient_ids=None)
    # No ETT evidence anywhere in the batch: vgcs 1 is a real verbal score.
    out = _callback_gcs({"egcs": table("egcs", [3.0, 3.0]), "mgcs": table("mgcs", [5.0, 5.0]),
                         "vgcs": table("vgcs", [1.0, 1.0])}, ctx)
    assert out.data["gcs"].tolist() == [9.0, 9.0]


def test_r4_085_row_filters_on_a_default_value_column_declare_it():
    # The loader renames a table's default value column to the concept name;
    # a filter on it finds the column only through a declared val_var (AUMC
    # adv_resp 'Stand By' broke the respiratory and SOFA-2 modules otherwise).
    import json
    from pathlib import Path

    import easyicu

    data = Path(easyicu.__file__).parent / "data"
    tables = {s["name"]: s["tables"] for s in json.loads((data / "data-sources.json").read_text())}
    offenders = []
    for name in ("concept-dict.json", "sofa2-dict.json"):
        for concept, spec in _dict(name).items():
            for db, sources in (spec.get("sources") or {}).items() if isinstance(spec, dict) else ():
                for source in sources:
                    default = ((tables.get(db) or {}).get(source.get("table"), {}).get("defaults") or {}).get("val_var")
                    filtered = set(source.get("exclude_regex") or {}) | set(source.get("require_regex") or {})
                    if default in filtered and not (source.get("val_var") or source.get("value_var")):
                        offenders.append((name, concept, db))
    assert offenders == []
