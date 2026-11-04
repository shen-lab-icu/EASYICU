"""R5 residue regressions: invented intervals and small source fixtures only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_sofa2_score,
    _callback_vaso60,
    _callback_vaso_ind,
    _callback_vent_ind,
    _expand_wintbl_to_hourly,
)
from easyicu.concept.eicu_ventilation_windows import bridge_hours
from easyicu.concept.window_expansion import expand_wintbl_vectorized
from easyicu.resources import load_dictionary
from easyicu.scores.sofa2_aggregate import SOFA2_COMPONENT_NAMES
from easyicu.table import ICUTable, WinTbl
from easyicu.utils.callback_utils import locb, locf
from tests.support.blood_gas_sources import _source, _write


def _table(name, rows):
    return ICUTable(
        pd.DataFrame(rows, columns=["stay_id", "charttime", name]),
        id_columns=["stay_id"], index_column="charttime", value_column=name,
    )


def _context(name, database="miiv", **kwargs):
    return ConceptCallbackContext(
        concept_name=name, target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=SimpleNamespace(config=SimpleNamespace(name=database)),
        patient_ids=None, kwargs=kwargs,
    )


@pytest.mark.parametrize("mode", ["raw", "floored_clamped"])
@pytest.mark.parametrize("legacy_zero_flag", [False, True])
def test_ri_01_wintbl_publishes_only_positive_overlap(mode, legacy_zero_flag):
    frame = pd.DataFrame({
        "stay_id": range(1, 8), "time": [0., .8, -1.8, 3., 4., np.nan, 6.],
        "dur": [2., 2.4, 1., 0., -1., 2., np.nan], "value": 7.,
    })
    before = frame.copy(deep=True)
    out = expand_wintbl_vectorized(
        frame, idx_col="time", dur_col="dur", id_cols=["stay_id"],
        value_columns=["value"], interval_hours=1., end_mode=mode,
        duration_zero_single=legacy_zero_flag,
    )
    assert out.groupby("stay_id").time.apply(list).to_dict() == {
        1: [0., 1.], 2: [0., 1., 2., 3.], 3: [-2., -1.],
    }
    assert out.value.eq(7).all()
    pd.testing.assert_frame_equal(frame, before)


def _mv_source(tmp_path, database, rows, *, amount="amount"):
    identifier = "stay_id" if database == "miiv" else "icustay_id"
    origin = pd.Timestamp("2180-01-01 00:17:15")
    frame = pd.DataFrame(rows)
    frame[identifier] = 1
    frame["subject_id"], frame["hadm_id"] = 101, 201
    for column in ("starttime", "endtime"):
        frame[column] = origin + pd.to_timedelta(frame[column], unit="h")
    frame["rateuom"], frame["amountuom"] = "mL/hour", "mL" if amount == "dex" else "mg"
    frame["statusdescription"] = "FinishedRunning"
    frame["linkorderid"] = range(1, len(frame) + 1)
    frame["orderid"] = frame["linkorderid"]
    _write(frame, tmp_path / ("inputevents.parquet" if database == "miiv" else "inputevents_mv.parquet"))
    _write(pd.DataFrame({
        identifier: [1], "subject_id": [101], "hadm_id": [201],
        "intime": [origin], "outtime": [origin + pd.Timedelta(hours=48)],
    }), tmp_path / "icustays.parquet")
    return _source(tmp_path, database)


def _load_one(source, name, *, target=None):
    dictionary = load_dictionary()
    definition = dictionary[name]
    # Isolate the relevant interval source; eMAR and CareVue are point sources.
    definition.sources[source.config.name] = definition.sources[source.config.name][:1]
    if target:
        definition.target = target
    result = ConceptResolver(dictionary).load_concepts(
        [name], source, merge=False, interval=pd.Timedelta(hours=1),
        r_compatible=False, verbose=False, concept_workers=1,
    )[name]
    return result.data if hasattr(result, "data") else result


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("target", ["ts_tbl", "win_tbl"])
@pytest.mark.parametrize("start,stop,expected", [(0., 2., [0., 1.]), (.8, 3.2, [0., 1., 2., 3.])])
def test_ri_02_source_expansion_keeps_final_partial_hour_without_stop_hour(tmp_path, database, target, start, stop, expected):
    source = _mv_source(tmp_path, database, {
        "starttime": [start], "endtime": [stop], "itemid": [220950],
        "amount": [20. * (stop - start)], "rate": [20.],
    }, amount="dex")
    out = _load_one(source, "dex", target=target)
    clock = next(c for c in ("starttime", "charttime") if c in out)
    assert out[clock].tolist() == expected
    assert out.dex.tolist() == pytest.approx([20.] * len(expected))


@pytest.mark.parametrize("clock", ["numeric", "duration", "datetime", "mixed"])
def test_ri_02_expander_aligns_both_clocks_to_the_admission_grid(clock):
    from easyicu.io.ts_utils import expand

    origin = pd.Timestamp("2020-01-01 00:17:15")
    frame = pd.DataFrame({"stay_id": [1, 2, 1, 1], "starttime": [.8, 4.8, 4., 6.], "endtime": [3.2, 5.2, 4., 5.], "rate": [10., 20., 99., 99.]})
    origins = pd.DataFrame({"stay_id": [1, 2], "intime": [origin, origin + pd.Timedelta(minutes=20)]})
    anchors = frame.stay_id.map(origins.set_index("stay_id").intime)
    if clock == "duration":
        frame.endtime = pd.to_timedelta(frame.endtime - frame.starttime, unit="h")
    elif clock in ("datetime", "mixed"):
        frame.endtime = anchors + pd.to_timedelta(frame.endtime, unit="h")
        if clock == "datetime":
            frame.starttime = anchors + pd.to_timedelta(frame.starttime, unit="h")
    # Concatenated sources may reuse row labels; labels must not multiply rows.
    frame.index = [0, 0, 1, 2]
    out = expand(frame, "starttime", "endtime", pd.Timedelta(hours=1), id_cols=["stay_id"], keep_vars=["rate"], admission_times=origins, end_inclusive=False)
    expected = [0., 1., 2., 3., 4., 5.]
    if clock == "datetime":
        expected = list(pd.DatetimeIndex([origin] * 4 + [origins.intime.iloc[1]] * 2) + pd.to_timedelta(expected, unit="h"))
    assert out.starttime.tolist() == expected
    assert out.stay_id.tolist() == [1] * 4 + [2] * 2
    assert out.rate.tolist() == [10.] * 4 + [20.] * 2


@pytest.mark.parametrize("datetime_clock", [False, True, "utc"])
@pytest.mark.parametrize("time_name", ["charttime", "infusionoffset"])
def test_ri_02_public_numeric_window_adapter_uses_positive_overlap(datetime_clock, time_name):
    from easyicu.api.concepts import _expand_public_numeric_win_tbl_output, MAX_WINDOW_EXPANSION_POINTS
    from easyicu.table.duration import set_dur_var_unit

    origin = pd.Timestamp("2020-01-01", tz="UTC" if datetime_clock == "utc" else None)
    frame = pd.DataFrame({"stay_id": [1, 2, 3, 4], "charttime": [0., .5, .8, 3.], "dur_var": [1., .5, 2.4, 0.], "dex": 10.})
    if datetime_clock:
        frame.charttime = origin + pd.to_timedelta(frame.charttime, unit="h")
    frame = frame.rename(columns={"charttime": time_name})
    set_dur_var_unit(frame, "hours")
    out = _expand_public_numeric_win_tbl_output(frame, "dex", "1h").rename(columns={time_name: "charttime"})
    if datetime_clock:
        out.charttime = (out.charttime - origin) / pd.Timedelta(hours=1)
    assert out.groupby("stay_id").charttime.apply(list).to_dict() == {1: [0.], 2: [0.], 3: [0., 1., 2., 3.]}
    # Sparse single-stay loads must not rely on other stays' grid spacing to
    # recognize an already-normalized hour axis with an offset column name.
    single = _expand_public_numeric_win_tbl_output(frame.loc[frame.stay_id.eq(3)], "dex", "1h")
    single_clock = single[time_name]
    if datetime_clock:
        single_clock = (single_clock - origin) / pd.Timedelta(hours=1)
    assert single_clock.tolist() == [0., 1., 2., 3.]
    frame.loc[0, "dur_var"] = MAX_WINDOW_EXPANSION_POINTS + 1
    with pytest.raises(ValueError, match="expand|points"):
        _expand_public_numeric_win_tbl_output(frame, "dex", "1h")


_INDICATORS = [(kind, database) for kind in ("vaso", "vaso60") for database in ("miiv", "mimic", "eicu", "sic")]
_INDICATORS += [(kind, database) for kind in ("vent_events", "vent_windows") for database in ("miiv", "mimic")]
_INDICATORS += [("vent_helper", "miiv")]


@pytest.mark.parametrize("kind,database", _INDICATORS)
def test_ri_03_indicators_use_half_open_windows_without_stretching_pre_admission_ends(kind, database):
    expected = {1: [0., 1.], 2: [0., 1., 2., 3.], 3: [-5.]}
    durations = _table("norepi_dur", [(1, 0., 2.), (2, .8, 2.4), (3, -5., 1.), (4, 3., 0.)])
    if kind == "vaso":
        out = _callback_vaso_ind({"norepi_dur": durations}, _context("vaso_ind", database)).data
    elif kind == "vaso60":
        rate = _table("norepi_rate", [(stay, hour, .1) for stay in range(1, 5) for hour in range(-5, 5)])
        out = _callback_vaso60({"norepi_dur": durations, "norepi_rate": rate}, _context("norepi60", database)).data
    else:
        frame = durations.data.rename(columns={"charttime": "starttime", "norepi_dur": "dur_var"})
        if kind == "vent_helper":
            out = _expand_wintbl_to_hourly(frame, ["stay_id"], "starttime", "dur_var", "charttime")
        elif kind == "vent_windows":
            frame["mech_vent"] = "invasive"
            table = WinTbl(frame, id_vars=["stay_id"], index_var="starttime", dur_var="dur_var")
            windows = _callback_vent_ind({"mech_vent": table}, _context("vent_ind", database))
            out = _expand_wintbl_to_hourly(windows.data, ["stay_id"], "starttime", "dur_var", "charttime")
        else:
            starts = _table("vent_start", [(1, 0., True), (2, .8, True), (3, -5., True), (4, 3., True)])
            ends = _table("vent_end", [(1, 2., True), (2, 3.2, True), (3, -4., True), (4, 3., True)])
            out = _callback_vent_ind({"vent_start": starts, "vent_end": ends}, _context("vent_ind", database)).data
    assert out.groupby("stay_id").charttime.apply(list).to_dict() == expected


@pytest.mark.parametrize("database", ["miiv", "mimic", "aumc"])
@pytest.mark.parametrize("stop,expected", [(2., [28., 40.]), (3.2, [24., 20., 20., 4.])])
def test_ri_04_continuous_diuretic_amount_is_conserved_by_icu_hour_overlap(tmp_path, database, stop, expected):
    name = "loop_diuretic_furosemide_eq"
    if database == "aumc":
        origin = 17.25
        _write(pd.DataFrame({"admissionid": [1], "admittedat": [origin * 60_000], "dischargedat": [(origin + 48 * 60) * 60_000]}), tmp_path / "admissions.parquet")
        _write(pd.DataFrame({
            "admissionid": [1] * 5, "itemid": [7244] * 5,
            "start": [(origin + h * 60) * 60_000 for h in (.8, .8, 20., 10., 10.)],
            "stop": [(origin + h * 60) * 60_000 for h in (stop, .8, 23., 10., 10.)],
            "administered": [48., 20., 3000., 1500., 1500.], "administeredunit": ["mg"] * 5,
            "ordercategory": ["Spuitpompen", "Injecties", "Spuitpompen", "Injecties", "Injecties"],
            "iscontinuous": [True, False, True, False, False],
        }), tmp_path / "drugitems.parquet")
        source = _source(tmp_path, database)
    else:
        source = _mv_source(tmp_path, database, {
            "starttime": [.8, .8, 20., 10., 10.], "endtime": [stop, 3.2, 23., 11., 11.],
            "itemid": [228340, 221794, 228340, 221794, 221794],
            "amount": [48., 20., 3000., 1500., 1500.], "rate": [20., np.nan, 1000., np.nan, np.nan],
        })
    out = _load_one(source, name)
    clock = next(c for c in ("start", "starttime", "charttime") if c in out)
    # The 2000 mg bound belongs to each original dose. Splitting must not
    # rescue a 3000 mg record, or reject two valid 1500 mg boluses at hour 10.
    assert out[clock].tolist() == list(map(float, range(len(expected)))) + [10.]
    assert out[name].tolist() == pytest.approx(expected + [3000.])
    assert out[name].sum() == pytest.approx(3068.)


def test_ri_04_amount_allocation_preserves_dose_records_across_chunks():
    from easyicu.utils.callback_utils import distribute_volume_hourly

    frame = pd.DataFrame({"stay_id": 1, "starttime": [.25] * 3, "endtime": [1.25] * 3, "amount": [600., 600., 800.]})
    out = distribute_volume_hourly(frame, val_col="amount", row_chunk_size=1, aggregate=False)
    assert out.groupby("starttime").amount.apply(sorted).to_dict() == {
        0.: [450., 450., 600.], 1.: [150., 150., 200.],
    }
    assert out.amount.sum() == pytest.approx(frame.amount.sum())


@pytest.mark.parametrize("window", [6, 24])
@pytest.mark.parametrize("datetime_clock", [False, True])
def test_ri_05_sofa2_component_and_availability_expire_with_the_score_window(window, datetime_clock):
    times = [0., window - 1., float(window), window + 1., window * 2.]
    clock = pd.Timestamp("2020-01-01") + pd.to_timedelta(times, unit="h") if datetime_clock else times
    tables = {}
    for name in SOFA2_COMPONENT_NAMES:
        liver = name == "sofa2_liver"
        frame = pd.DataFrame({
            "stay_id": 1, "charttime": clock,
            name: [3., 0., 0., 0., 0.] if liver else 0.,
            f"{name}_observed": [1, 0, 0, 0, 0] if liver else 1,
            f"{name}_available": [1, 0, 0, 0, 0] if liver else 1,
        })
        tables[name] = ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column=name)
    out = _callback_sofa2_score(tables, _context("sofa2", keep_components=True, win_length=pd.Timedelta(hours=window))).data.set_index("charttime")
    assert out.loc[clock[1], "sofa2_liver"] == 3
    assert pd.isna(out.loc[clock[2], "sofa2_liver"])
    assert out.loc[clock[2], "sofa2"] == 0
    assert out.loc[clock[2], "sofa2_n_available_components"] == 5
    assert out.loc[clock[-1], "sofa2_n_available_components"] == 5


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("timedelta_clock", [False, True])
def test_ri_06_locf_and_locb_limit_elapsed_age_from_real_observations(direction, grouped, timedelta_clock):
    times = pd.to_timedelta([0, 1, 2, 3, 4, 6, 7], unit="h")
    if not timedelta_clock:
        times = pd.Timestamp("2020-01-01") + times
    values = [10., np.nan, np.nan, np.nan, np.nan, 20., np.nan] if direction == "forward" else [np.nan, np.nan, np.nan, 10., np.nan, 20., np.nan]
    expected = [10., 10., 10., np.nan, np.nan, 20., 20.] if direction == "forward" else [np.nan, 10., 10., 10., 20., 20., np.nan]
    frame = pd.DataFrame({"stay_id": 1, "time": times, "value": values})
    if grouped:
        frame = pd.concat([frame, frame.assign(stay_id=2, value=np.nan)], ignore_index=True)
    frame = frame.sample(frac=1, random_state=17)
    before = frame.copy(deep=True)
    callback = locf if direction == "forward" else locb
    out = callback(max_gap=pd.Timedelta(hours=2))(frame, index_col="time", id_cols=["stay_id"] if grouped else None)
    result = out.loc[out.stay_id.eq(1)].sort_values("time").value.to_numpy()
    np.testing.assert_allclose(result, expected, equal_nan=True)
    assert out.loc[out.stay_id.eq(2), "value"].isna().all()
    pd.testing.assert_frame_equal(frame, before)


def test_ri_07_eicu_care_stop_does_not_create_an_extra_hour():
    intervals = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5],
        "start_h": [.5, 1.5, 3., 4., np.nan], "end_h": [2., 2.25, 3., 3., 5.],
    })
    points = pd.DataFrame({"patientunitstayid": [6], "hour": [4.]})
    out = bridge_hours(points, intervals)
    assert out.groupby("patientunitstayid").hour.apply(list).to_dict() == {1: [0, 1], 2: [1, 2], 6: [4]}


def test_r6_f011_eicu_zero_markers_close_but_never_start_support():
    from easyicu.utils.callback_utils import eicu_duration_callback

    rows = [(1, 0, 0), (1, 60, 0), (2, 30, 2), (2, 60, 0),
            (3, 0, 0), (3, 30, 2), (3, 90, 2), (3, 120, 0),
            (4, 0, None), (4, 60, None), (5, 30, 2)]
    frame = pd.DataFrame(rows, columns=["patientunitstayid", "infusionoffset", "drugrate"])
    duration = eicu_duration_callback(pd.Timedelta(hours=5))(
        frame, val_col="drugrate", index_var="infusionoffset", id_cols=["patientunitstayid"],
    )
    known = duration.dropna(subset=["drugrate"]).set_index("patientunitstayid")
    assert known.index.tolist() == [2, 3]
    assert known.infusionoffset.tolist() == [30, 30]
    assert known.drugrate.tolist() == [.5, 1.5]
    durations = _table("norepi_dur", [(2, .5, .5), (3, .5, 1.5)])
    other = _table("other_vaso", [(1, 0., True), (4, 0., False)])
    out = _callback_vaso_ind({"norepi_dur": durations, "other_vaso": other}, _context("vaso_ind", "eicu")).data
    assert out.groupby("stay_id").charttime.apply(list).to_dict() == {1: [0.], 2: [0.], 3: [0., 1.]}


@pytest.mark.parametrize("database", ["miiv", "eicu", "sic", "aumc"])
def test_r6_f012_adjunct_only_support_is_unioned_in_every_database(database):
    identifier = "admissionid" if database == "aumc" else "stay_id"
    durations = _table("norepi_dur", [(1, .999, 1.002), (2, 0., 0.)])
    other = _table("other_vaso", [(3, .5, True), (4, .5, False)])
    if identifier != "stay_id":
        durations = ICUTable(durations.data.rename(columns={"stay_id": identifier}), id_columns=[identifier], index_column="charttime", value_column="norepi_dur")
        other = ICUTable(other.data.rename(columns={"stay_id": identifier}), id_columns=[identifier], index_column="charttime", value_column="other_vaso")
    out = _callback_vaso_ind({"norepi_dur": durations, "other_vaso": other}, _context("vaso_ind", database)).data
    clock = next(c for c in ("charttime", "start") if c in out)
    assert out.groupby(identifier)[clock].apply(list).to_dict() == {1: [0., 1., 2.], 3: [0.]}


@pytest.mark.parametrize("database", ["miiv", "hirid"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f013_public_and_native_ventilation_keep_precise_mixed_windows(tmp_path, monkeypatch, database, layout):
    from easyicu.api.extraction import _canonicalise_native_export_frame
    from tests.support.hirid_source import hirid_source_root as _source_root

    if database == "hirid":
        root = _source_root(tmp_path, "observations", [
            {"patientid": 1, "datetime": .999, "variableid": 15001552, "value": 1.},
            {"patientid": 1, "datetime": 1.001, "variableid": 15001552, "value": 3.},
            {"patientid": 1, "datetime": 2., "variableid": 15001552, "value": 6.},
        ], layout)
        source = _source(root, database)
    else:
        source = _mv_source(tmp_path, database, {"starttime": [0.], "endtime": [0.], "itemid": [220950], "amount": [0.], "rate": [0.]})
        origin = pd.Timestamp("2180-01-01 00:17:15")
        frame = pd.DataFrame({"stay_id": [1]*4, "subject_id": [101]*4, "hadm_id": [201]*4,
                              "itemid": [225792, 225794, 225792, 225792],
                              "starttime": origin + pd.to_timedelta([.999, 1.001, 4., 5.], unit="h"),
                              "endtime": origin + pd.to_timedelta([1.001, 2., 4., 6.], unit="h"),
                              "value": [1.]*4, "statusdescription": ["FinishedRunning"]*4})
        path = tmp_path / ("procedureevents.parquet" if layout == "flat" else "procedureevents_bucket/bucket_id=0/1.parquet")
        _write(frame, path)
    dictionary = load_dictionary()
    names = ["mech_vent", "vent_ind"]
    for requested in (["mech_vent"], names, list(reversed(names))):
        resolver = ConceptResolver(dictionary)
        merged = resolver.load_concepts(requested, source, merge=True, interval=pd.Timedelta(hours=1), r_compatible=True, verbose=False, concept_workers=1)
        if hasattr(merged, "data"):
            merged = merged.data
        if len(requested) == 1:
            # The single window API intentionally returns WinTbl intervals.
            # Verify its exact endpoints before materializing the same owner.
            clock = next(c for c in ("datetime", "starttime", "charttime") if c in merged)
            first = merged.loc[merged.mech_vent.eq("invasive")].iloc[0]
            assert first[clock] == pytest.approx(.999)
            assert first.dur_var == pytest.approx(.002)
            continue
        merged = merged.rename(columns={c: "charttime" for c in ("datetime", "givenat", "starttime") if c in merged and "charttime" not in merged})
        native = _canonicalise_native_export_frame(merged, module="respiratory", requested_concepts=requested, dictionary=dictionary, database=database)
        mech = native.loc[native.stay_id.eq(1) & native.mech_vent.notna()].set_index("charttime").mech_vent
        assert set(mech.index) == ({0., 1.} if database == "hirid" else {0., 1., 5.})
        assert mech.loc[0.] == "invasive"
        assert "invasive" in mech.loc[1.] and "noninvasive" in mech.loc[1.]
        if "vent_ind" in requested:
            actual = set(native.loc[native.stay_id.eq(1) & native.vent_ind.eq(True), "charttime"])
            assert actual == ({0., 1.} if database == "hirid" else {0., 1., 5.})
    if layout == "bucket":
        published = _publish_synthetic_subset(source, names, tmp_path / "published", monkeypatch)
        actual = published.loc[published.stay_id.eq(1)].set_index("charttime")
        assert actual.loc[1., "mech_vent"] == "invasive|noninvasive"
        assert set(actual.index[actual.vent_ind.eq(True)]) == ({0., 1.} if database == "hirid" else {0., 1., 5.})


def test_r6_f014_care_to_bridge_retains_partial_hours_across_stop_barriers():
    from easyicu.concept.eicu_ventilation_windows import care_intervals, cut_hours_at_barriers

    care = pd.DataFrame({"patientunitstayid": [1, 2, 3], "ventstartoffset": [0., 30., -30.], "ventendoffset": [60., 60., 30.],
                         "airwaytype": "Oral ETT"})
    intervals = care_intervals(care)
    points = pd.DataFrame(columns=["patientunitstayid", "hour"])
    out = bridge_hours(points, intervals)
    assert out.groupby("patientunitstayid").hour.apply(list).to_dict() == {1: [0.], 2: [0.], 3: [0.]}
    # A stop halfway through the cell cannot erase its earlier IMV overlap.
    tables = {"respiratorycare": care}
    class Source:
        def load_table(self, name, **kwargs):
            return tables.get(name, pd.DataFrame(columns=kwargs.get("columns", [])))
    cut = cut_hours_at_barriers(out, points, intervals, Source(), [1, 2, 3], "patientunitstayid")
    assert cut.groupby("patientunitstayid").hour.apply(list).to_dict() == {1: [0.], 2: [0.], 3: [0.]}
    tables["respiratorycare"] = pd.DataFrame({"patientunitstayid": [1], "ventstartoffset": [0.], "ventendoffset": [300.], "airwaytype": ["Oral ETT"]})
    tables["respiratorycharting"] = pd.DataFrame({"patientunitstayid": [1]*3, "respchartoffset": [90., 210., 240.],
        "respchartvaluelabel": ["O2 Device"]*3, "respchartvalue": ["BiPAP", "Ventilator", "Extubated"]})
    intervals = care_intervals(tables["respiratorycare"])
    hours = bridge_hours(points, intervals)
    cut = cut_hours_at_barriers(hours, points, intervals, Source(), [1], "patientunitstayid")
    assert set(cut.hour) == {0., 1., 3.}


@pytest.mark.parametrize("prefix", ["norepi", "epi", "dobu"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f015_duration_is_identical_in_single_batch_and_native_paths(tmp_path, monkeypatch, prefix, layout):
    from easyicu.api.extraction import _canonicalise_native_export_frame
    from tests.support.hirid_source import hirid_source_root as _source_root

    dictionary = load_dictionary()
    ids = dictionary[f"{prefix}_dur"].sources["hirid"][0].ids
    item = ids[0] if isinstance(ids, list) else ids
    root = _source_root(tmp_path, "pharma", [
        {"patientid": stay, "givenat": t, "pharmaid": item, "givendose": dose, "doseunit": "µg", "route": "iv-inf", "infusionid": stay, "recordstatus": 0}
        for stay, start, end in [(1, .999, 2.001), (2, .75, 1.5)]
        for t, dose in [(start, 0.), (end, 6.)]
    ], layout)
    source = _source(root, "hirid")
    name = f"{prefix}_dur"
    for requested in ([name], [name, "vaso_ind", f"{prefix}60"], [f"{prefix}60", "vaso_ind", name]):
        resolver = ConceptResolver(dictionary)
        out = resolver.load_concepts(requested, source, merge=False, interval=pd.Timedelta(hours=1), r_compatible=False, verbose=False, concept_workers=1)
        duration = out[name].data.sort_values("patientid")
        assert duration.givenat.tolist() == pytest.approx([.999, .75])
        assert duration[name].tolist() == pytest.approx([1.002, .75])
        if f"{prefix}60" in requested:
            sixty = out[f"{prefix}60"].data
            clock = next(c for c in ("charttime", "givenat", "datetime") if c in sixty)
            assert set(sixty.loc[sixty.patientid.eq(1), clock]) == {0., 1., 2.}
            assert sixty.loc[sixty.patientid.eq(2)].empty
        merged = ConceptResolver(dictionary).load_concepts(requested, source, merge=True, interval=pd.Timedelta(hours=1), r_compatible=True, verbose=False, concept_workers=1)
        merged = merged.rename(columns={c: "charttime" for c in ("datetime", "givenat", "starttime") if c in merged and "charttime" not in merged})
        native = _canonicalise_native_export_frame(merged, module="medications", requested_concepts=requested, dictionary=dictionary, database="hirid")
        actual = native.loc[native[name].notna()].sort_values("stay_id")
        assert actual[name].tolist() == pytest.approx([1.002, .75])
    if layout == "bucket" and prefix == "norepi":
        published = _publish_synthetic_subset(source, [name, "vaso_ind", f"{prefix}60"], tmp_path / "published", monkeypatch)
        assert published.loc[published[name].notna()].sort_values("stay_id")[name].tolist() == pytest.approx([1.002, .75])
        assert set(published.loc[published.stay_id.eq(1) & published[f"{prefix}60"].gt(0), "charttime"]) == {0., 1., 2.}


def _publish_synthetic_subset(source, names, output, monkeypatch):
    import os
    import tempfile
    from easyicu.api import extraction

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "vitals", names)
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "outcome", ["los_icu"])
    result = extraction.extract_database(
        source.config.name, data_path=str(source.base_path), output_dir=str(output),
        modules=["vitals", "outcome"], native_export_v2=True, verbose=False,
        resource_budget_mb=1024,
    )
    assert not result["modules"]["vitals"].get("errors")
    return pd.read_parquet(output / "vitals.parquet")


@pytest.mark.parametrize("prefix,label", [("norepi", "Norepinephrine"), ("epi", "Epinephrine"), ("dopa", "Dopamine"), ("dobu", "dobutamine")])
@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("stop_value", [0, None])
def test_r6_f011_public_eicu_duration_requires_positive_rate_and_keeps_alternate_support(tmp_path, prefix, label, alternate, stop_value):
    from tests.support.eicu_source import _eicu_source

    rows = [(t, f"{label} (mcg/kg/min)", rate) for t, rate in [(0, 0), (60, 0), (90, .1), (150, .1), (180, stop_value)]]
    if alternate:
        rows += [(15, "Phenylephrine (mcg/kg/min)", .2), (45, "Phenylephrine (mcg/kg/min)", 0)]
    source = _eicu_source(tmp_path, "eicu", infusiondrug=pd.DataFrame(rows, columns=["infusionoffset", "drugname", "drugrate"]))
    names = [f"{prefix}_dur", f"{prefix}_rate", "vaso_ind"]
    out = ConceptResolver(load_dictionary()).load_concepts(names, source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1)
    durations = out[names[0]].data
    assert durations[names[0]].tolist() == [1.5]
    assert durations[out[names[0]].index_column].tolist() == [1.5]
    indicator = out["vaso_ind"]
    assert set(indicator.data[indicator.index_column]) == ({0., 1., 2.} if alternate else {1., 2.})
    rate = out[names[1]]
    assert set(rate.data.loc[rate.data[names[1]].gt(0), rate.index_column]) == {1., 2.}


@pytest.mark.parametrize("kind", ["kg", "mass"])
@pytest.mark.parametrize("stop_value", [0., None])
def test_r6_f012_eicu_rate_quantizes_after_exact_stop_and_keeps_partial_cell(kind, stop_value):
    from easyicu.utils.callback_utils import eicu_rate_kg_callback, eicu_rate_mass_callback

    frame = pd.DataFrame({"patientunitstayid": [1, 1, 2, 2], "infusionoffset": [30., 90., 30., 60.],
                          "drugname": ["drug (mcg/kg/min)" if kind == "kg" else "drug (mg/hr)"]*4,
                          "rate": [1., stop_value, 1., stop_value]})
    callback = eicu_rate_kg_callback(32) if kind == "kg" else eicu_rate_mass_callback("mg/hour")
    out = callback(frame, val_var="rate", sub_var="drugname", concept_name="rate", **({"weight_var": "patientweight"} if kind == "kg" else {}))
    assert out.loc[out.rate.gt(0)].groupby("patientunitstayid").infusionoffset.apply(list).to_dict() == {1: [0., 60.], 2: [0.]}


@pytest.mark.parametrize("database", ["miiv", "aumc", "sic"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f012_public_source_clocks_match_rate_duration_and_all_drug_support(tmp_path, database, layout):
    from tests.support.aumc_source import aumc_db as _db, aumc_drug as _drug
    from easyicu.synthetic_sources import generate_source

    tmp_path = tmp_path / "source"
    intervals = [(-.5, 1., "ne"), (2.5, 3.5, "phn"), (5., 6., "adh"), (7., 7., "ne"), (8.999, 9.001, "ne")]
    items = {"miiv": {"ne": 221906, "phn": 221749, "adh": 222315},
             "aumc": {"ne": 7229, "phn": 19929}, "sic": {"ne": 1562, "phn": 1593, "adh": 1550}}[database]
    intervals = [(a, b, drug) for a, b, drug in intervals if drug in items]
    if database == "aumc":
        source = _db(tmp_path, drugs=[_drug(a*60, b*60, items[drug], dose=.1) for a, b, drug in intervals], layout="bucket" if layout == "bucket" else "file")
    elif database == "miiv":
        source = _mv_source(tmp_path, database, {"starttime": [a for a, b, drug in intervals], "endtime": [b for a, b, drug in intervals],
            "itemid": [items[drug] for a, b, drug in intervals], "rate": [.1]*len(intervals), "amount": [1.]*len(intervals)})
        path = tmp_path / "inputevents.parquet"
        frame = pd.read_parquet(path)
        frame["rateuom"] = ["units/min" if drug == "adh" else "mcg/kg/min" for a, b, drug in intervals]
        frame["patientweight"] = 80.
        if layout == "bucket":
            path.unlink()
            path = tmp_path / "inputevents_bucket/bucket_id=0/1.parquet"
        _write(frame, path)
    else:
        generate_source(tmp_path, "sic", stays=3)
        origin = pd.read_parquet(tmp_path / "cases.parquet").set_index("CaseID").loc[1, "ICUOffset"]
        path = tmp_path / "medication.parquet"
        frame = pd.DataFrame({"CaseID": 1, "DrugID": [items[drug] for a, b, drug in intervals],
            "Offset": [origin + a*3600 for a, b, drug in intervals], "OffsetDrugEnd": [origin + b*3600 for a, b, drug in intervals],
            "Amount": 1., "AmountPerMinute": [.02 if drug == "adh" else .000008 for a, b, drug in intervals], "IsSingleDose": 0})
        if layout == "bucket":
            path.unlink()
            path = tmp_path / "medication_bucket/bucket_id=0/1.parquet"
        _write(frame, path)
        source = _source(tmp_path, database)
    names = ["norepi_dur", "norepi_rate", "norepi60", "phn_rate", "adh_rate", "vaso_ind"]
    expected = {-1., 0., 2., 3., 8., 9.} | ({5.} if "adh" in items else set())
    for requested in (["vaso_ind"], names, list(reversed(names))):
        out = ConceptResolver(load_dictionary()).load_concepts(requested, source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1)
        table = out["vaso_ind"]
        assert set(table.data.loc[table.data.vaso_ind.eq(True), table.index_column]) == expected
        if "norepi60" in out:
            table = out["norepi60"]
            assert set(table.data.loc[table.data.norepi60.gt(0), table.index_column]) == {-1., 0.}


def test_r6_f012_boolean_union_does_not_lose_positive_adjunct_under_zero_rates(tmp_path):
    source = _mv_source(tmp_path, "miiv", {"starttime": [2.5]*3, "endtime": [3.5]*3,
        "itemid": [221749]*3, "rate": [.1, 0., 0.], "amount": [1., 0., 0.]})
    path = tmp_path / "inputevents.parquet"
    frame = pd.read_parquet(path).assign(rateuom="mcg/kg/min", patientweight=80.)
    _write(frame, path)
    for names in (["vaso_ind"], ["phn_rate", "vaso_ind"], ["vaso_ind", "phn_rate"]):
        out = ConceptResolver(load_dictionary()).load_concepts(names, source, merge=False, r_compatible=False,
            interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1)
        indicator = out["vaso_ind"]
        assert set(indicator.data[indicator.index_column]) == {2., 3.}
        if "phn_rate" in out:
            # The boolean union's max must not change the public numeric rate.
            assert out["phn_rate"].data.phn_rate.tolist() == [0., 0.]


@pytest.mark.parametrize("concept", ["mech_vent", "rrt"])
def test_ri_01_merged_window_declared_duration_wins_over_absolute_source_end(concept):
    value = "invasive" if concept == "mech_vent" else True
    frame = pd.DataFrame({"stay_id": [1], "charttime": [.5], "dur_var": [.5],
                          "endtime": [pd.Timestamp("2180-01-01 01:00")], concept: [value]})
    table = WinTbl(frame, id_vars=["stay_id"], index_var="charttime", dur_var="dur_var", dur_unit="hours")
    out = ConceptResolver(load_dictionary())._to_r_format_merged_enhanced(
        {concept: table}, [concept], pd.Timedelta(hours=1), data_source=None)
    assert out.loc[out[concept].notna(), "charttime"].tolist() == [0.]
