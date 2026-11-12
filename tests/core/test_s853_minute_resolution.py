"""S853 sub-hour contract: the opt-in flag, SICdb rawdata minute rows and the gap summary."""

import struct

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.minute_resolution import (
    MINUTE_RESOLUTION_KWARG,
    SICDB_RECEIPT_ATTR,
    SUBHOUR_MODULE_INTERVALS,
    expand_sicdb_minute_rows,
    minute_resolution_requested,
    resolution_summary,
    sicdb_minute_rows_apply,
)


def _raw(values):
    """SICdb encoding: 0x + 60 little-endian float32; 0.0 marks an empty slot."""
    slots = list(values) + [0.0] * (60 - len(values))
    return "0x" + struct.pack("<60f", *slots).hex().upper()


def test_minute_mode_needs_the_explicit_flag_and_a_sub_hour_grid():
    minute = pd.Timedelta(minutes=1)
    assert not minute_resolution_requested({}, minute)
    assert not minute_resolution_requested(None, minute)
    # A concept-declared one-minute grid (vent_ind, norepi60) never opts in.
    assert not minute_resolution_requested({"interval": minute}, minute)
    assert not minute_resolution_requested({MINUTE_RESOLUTION_KWARG: True}, pd.Timedelta(hours=1))
    assert not minute_resolution_requested({MINUTE_RESOLUTION_KWARG: "yes"}, minute)
    assert minute_resolution_requested({MINUTE_RESOLUTION_KWARG: True}, minute)
    assert minute_resolution_requested({MINUTE_RESOLUTION_KWARG: True}, "30min")
    flag = {MINUTE_RESOLUTION_KWARG: True}
    assert sicdb_minute_rows_apply("sic", "data_float_h", flag, minute)
    assert not sicdb_minute_rows_apply("sic", "laboratory", flag, minute)
    assert not sicdb_minute_rows_apply("hirid", "data_float_h", flag, minute)
    assert SUBHOUR_MODULE_INTERVALS["subhour_signals"] == minute


def test_rawdata_rows_become_minute_rows_at_offset_plus_slot_minutes():
    frame = pd.DataFrame({
        "CaseID": [1, 1, 2],
        "DataID": [703, 703, 703],
        "Offset": [3600, 7200, 0],
        "Val": [80.0, 70.0, 65.0],
        "cnt": [3, 1, 2],
        # Slots 0, 1, 5 hold values; the second row is a single sample.
        "rawdata": [_raw([79.0, 81.0, 0, 0, 0, 80.0]), None, _raw([64.0, 66.0])],
    })
    out = expand_sicdb_minute_rows(frame)
    assert set(out.columns) == {"CaseID", "DataID", "Offset", "Val"}
    got = out.sort_values(["CaseID", "Offset"]).reset_index(drop=True)
    assert got["Offset"].tolist() == [3600, 3660, 3900, 7200, 0, 60]
    assert got["Val"].tolist() == [79.0, 81.0, 80.0, 70.0, 64.0, 66.0]
    assert got["Offset"].dtype == frame["Offset"].dtype
    receipt = out.attrs[SICDB_RECEIPT_ATTR]
    assert receipt["hour_rows"] == 3
    assert receipt["hour_rows_with_rawdata"] == 2
    assert receipt["minute_values"] == 5
    assert receipt["unplaceable_zero_values"] == 0


def test_hourly_value_is_the_mean_over_cnt_and_zeros_follow_cnt():
    complete = [10.0] * 30 + [0.0] * 30          # cnt 60: the zeros are measured
    partial = [6.0, 0.0, 6.0]                    # cnt 3: one measured zero, slot unknown
    frame = pd.DataFrame({
        "CaseID": [1, 2], "DataID": [2278, 2278], "Offset": [0, 0],
        "Val": [5.0, 4.0], "cnt": [60, 3],
        "rawdata": [_raw(complete), _raw(partial)],
    })
    out = expand_sicdb_minute_rows(frame)
    first = out.loc[out.CaseID.eq(1), "Val"]
    assert len(first) == 60 and first.mean() == pytest.approx(5.0)
    second = out.loc[out.CaseID.eq(2), "Val"]
    assert second.tolist() == [6.0, 6.0]
    receipt = out.attrs[SICDB_RECEIPT_ATTR]
    assert receipt["measured_zero_values"] == 30
    assert receipt["unplaceable_zero_values"] == 1


def test_malformed_and_non_finite_slots_are_counted_not_guessed():
    nan_slot = _raw([float("nan"), 7.0])
    frame = pd.DataFrame({
        "CaseID": [1, 2], "DataID": [2018, 2018], "Offset": [0, 0], "Val": [7.0, 9.0],
        "cnt": [1, 2], "rawdata": [nan_slot, "0xABC"],
    })
    out = expand_sicdb_minute_rows(frame)
    assert out.loc[out.CaseID.eq(1), ["Offset", "Val"]].values.tolist() == [[60, 7.0]]
    # A malformed stream keeps its hourly value at its own Offset.
    assert out.loc[out.CaseID.eq(2), ["Offset", "Val"]].values.tolist() == [[0, 9.0]]
    receipt = out.attrs[SICDB_RECEIPT_ATTR]
    assert receipt["non_finite_values"] == 1
    assert receipt["hour_rows_malformed_rawdata"] == 1


def test_missing_rawdata_columns_fail_closed():
    with pytest.raises(ValueError, match="sicdb_minute_rows_missing_columns"):
        expand_sicdb_minute_rows(pd.DataFrame({"Offset": [0], "Val": [1.0]}))


def test_numeric_rawdata_column_fails_closed():
    # A reader that cast rawdata to a number lost the minute stream.
    frame = pd.DataFrame({"Offset": [0, 0], "Val": [1.0, 2.0], "cnt": [2, 2], "rawdata": [np.nan, 3.5]})
    with pytest.raises(ValueError, match="sicdb_minute_rows_rawdata_not_text"):
        expand_sicdb_minute_rows(frame)
    # An all-null numeric column (no stream at all) keeps the hourly rows.
    out = expand_sicdb_minute_rows(frame.assign(rawdata=np.nan))
    assert out[["Offset", "Val"]].values.tolist() == [[0, 1.0], [0, 2.0]]


def test_resolution_summary_reports_gaps_without_identifiers(tmp_path):
    rows = []
    for stay in range(12):
        for minute in range(0, 40, 2):
            rows.append({"stay_id": 1000 + stay, "charttime": minute / 60.0,
                         "map_invasive": 70.0, "peep_set": 5.0 if minute % 30 == 0 else None})
    path = tmp_path / "subhour_signals.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    summary = resolution_summary(path, id_column="stay_id", concepts=["map_invasive", "peep_set", "cvp"])
    assert set(summary) == {"map_invasive", "peep_set"}
    m = summary["map_invasive"]
    assert m["rows"] == 240 and m["stays"] == 10 and m["gaps"] == 220
    assert m["gap_minutes_median"] == pytest.approx(2.0)
    assert m["share_gaps_le_5_min"] == 1.0
    p = summary["peep_set"]
    assert p["stays"] == 10 and p["gap_minutes_median"] == pytest.approx(30.0)
    assert p["share_gaps_le_15_min"] == 0.0
    text = repr(summary)
    assert "1000" not in text and "1011" not in text


@pytest.fixture
def hirid_minute_source(tmp_path):
    """Synthetic HiRID with three extra arterial MAP samples 2, 4 and 6 minutes after one,
    measured PEEP (2610) at minutes 3 and 5 and a PEEP setting (2600) at minute 3."""
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "hirid"
    generate_source(source, "hirid", stays=3)
    path = source / "observations" / "1.parquet"
    table = pd.read_parquet(path)
    anchor = table.loc[table["variableid"].eq(110)].sort_values(["patientid", "datetime"]).iloc[[0]]
    extra = pd.concat([anchor.assign(datetime=anchor["datetime"] + pd.Timedelta(minutes=m), value=v)
                       for m, v in ((2, 61.0), (4, 62.0), (6, 63.0))])
    extra = pd.concat([extra, *(anchor.assign(variableid=item, value=v,
                                              datetime=anchor["datetime"] + pd.Timedelta(minutes=m))
                                for item, m, v in ((2610, 3, 8.0), (2610, 5, 9.0), (2600, 3, 5.0)))])
    pd.concat([table, extra], ignore_index=True).astype(table.dtypes.to_dict()).to_parquet(path, index=False)
    yield source, int(anchor["patientid"].iloc[0])
    clear_global_loader()


def test_subhour_module_keeps_minute_samples_that_the_hourly_grid_merges(hirid_minute_source, tmp_path):
    from easyicu.api import extraction as api

    source, patient = hirid_minute_source
    frames = {}
    for module in ("subhour_signals", "vitals"):
        output = tmp_path / module
        output.mkdir()
        api._run_module_extraction(
            module_name=module, concepts=api.EXTRACT_MODULES[module], database="hirid",
            data_path=str(source), patient_ids_filter={"patientid": [1, 2, 3]}, batch_size=3,
            output_dir=str(output),
        )
        manifest = __import__("json").loads((output / "_manifest.json").read_text())
        assert manifest["errors"] == []
        frames[module] = pd.read_parquet(output / f"{module}.parquet")
    minute = frames["subhour_signals"]
    stay = minute.loc[minute.iloc[:, 0].eq(patient) & minute["map_invasive"].notna()]
    by_minute = dict(zip((pd.to_numeric(stay.iloc[:, 1]) * 60).round(6), stay["map_invasive"]))
    # Each extra sample is its own minute row (the synthetic anchor is out of
    # bounds and dropped by both grids).
    assert {k: by_minute.get(k) for k in (2.0, 4.0, 6.0)} == {2.0: 61.0, 4.0: 62.0, 6.0: 63.0}
    # The hourly module still merges them into one hourly median.
    hourly = frames["vitals"]
    hourly_stay = hourly.loc[hourly.iloc[:, 0].eq(patient) & hourly["map_invasive"].notna()]
    assert (pd.to_numeric(hourly_stay.iloc[:, 1]) % 1 == 0).all()
    assert hourly_stay.loc[pd.to_numeric(hourly_stay.iloc[:, 1]).eq(0.0), "map_invasive"].tolist() == [62.0]


def test_subhour_module_keeps_measured_peep_apart_from_the_setting(hirid_minute_source, tmp_path):
    from easyicu.api import extraction as api
    from easyicu.concept.catalog import SUBHOUR_SIGNAL_CONCEPTS

    assert SUBHOUR_SIGNAL_CONCEPTS[:4] == ("map_invasive", "cvp", "peep_set", "peep_total")
    source, patient = hirid_minute_source
    api._run_module_extraction(
        module_name="subhour_signals", concepts=api.EXTRACT_MODULES["subhour_signals"],
        database="hirid", data_path=str(source), patient_ids_filter={"patientid": [1, 2, 3]},
        batch_size=3, output_dir=str(tmp_path),
    )
    frame = pd.read_parquet(tmp_path / "subhour_signals.parquet")
    stay = frame.loc[frame.iloc[:, 0].eq(patient)]
    minutes = (pd.to_numeric(stay.iloc[:, 1]) * 60).round(6)
    total = dict(zip(minutes[stay["peep_total"].notna()], stay["peep_total"].dropna()))
    setting = dict(zip(minutes[stay["peep_set"].notna()], stay["peep_set"].dropna()))
    # HiRID 2610 (measured) feeds peep_total only, 2600 (setting) peep_set only.
    assert {k: total.get(k) for k in (3.0, 5.0)} == {3.0: 8.0, 5.0: 9.0}
    assert setting.get(3.0) == 5.0 and 5.0 not in setting
    # The native-v2 receipt summarises every requested signal with values.
    native = tmp_path / "native"
    result = api.extract_database(
        "hirid", data_path=str(source), output_dir=str(native), modules=["outcome", "subhour_signals"],
        native_export_v2=True, resource_budget_mb=2048, verbose=False,
    )
    assert not {k: v["errors"] for k, v in result["modules"].items() if v.get("errors")}
    entry = next(row for row in __import__("json").loads((native / "_manifest.json").read_text())["files"]
                 if row["module"] == "subhour_signals")
    assert entry["physical_concept_ids"] == list(SUBHOUR_SIGNAL_CONCEPTS)
    assert {"peep_set", "peep_total"} <= set(entry["time_resolution"]["signals"])
