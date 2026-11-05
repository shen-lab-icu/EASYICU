"""Round-7 interval/evidence regressions; all clinical rows are invented."""

from __future__ import annotations

import json
from math import ceil, floor
from pathlib import Path

import pandas as pd
import pytest

import easyicu
from easyicu.concept import ConceptResolver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource
from easyicu.resources import load_dictionary
from tests.support.hirid_source import hirid_load, hirid_source_root
from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source


DATA = Path(easyicu.__file__).parent / "data"
ORIGIN = pd.Timestamp("2150-01-01")


def _definition(name, filename="concept-dict.json"):
    return json.loads((DATA / filename).read_text())[name]


def _hours(table, name, *, stay=1, value=True):
    frame = table.data if hasattr(table, "data") else table
    clock = table.index_column if hasattr(table, "index_column") else next(c for c in ("datetime", "charttime", "starttime") if c in frame)
    identifier = next(c for c in ("stay_id", "icustay_id", "patientid") if c in frame)
    selected = frame.loc[frame[identifier].eq(stay) & frame[name].eq(value)]
    # merge=False returns exact WinTbl windows by contract. Independently
    # enumerate their covered cells; native checks below inspect published rows.
    if "dur_var" in selected:
        return {float(hour) for start, duration in selected[[clock, "dur_var"]].itertuples(index=False, name=None)
                if pd.notna(duration) and duration > 0
                for hour in range(floor(start), ceil(start + duration))}
    return set(selected[clock])


def _native(source, names, output, monkeypatch, module="respiratory"):
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    monkeypatch.setitem(extraction.EXTRACT_MODULES, module, names)
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "outcome", ["los_icu"])
    identifier = {"miiv": "stay_id", "mimic": "icustay_id", "hirid": "patientid"}[source.config.name]
    try:
        result = extraction.extract_database(
            source.config.name, data_path=str(source.base_path), output_dir=str(output),
            modules=[module, "outcome"], patient_ids={identifier: [1, 2, 3]},
            native_export_v2=True, verbose=False, resource_budget_mb=1024,
        )
        assert not {key: state["errors"] for key, state in result["modules"].items() if state.get("errors")}
        return pd.read_parquet(output / f"{module}.parquet")
    finally:
        clear_global_loader()


@pytest.mark.parametrize("filename", ["concept-dict.json", "sofa2-dict.json"])
@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
def test_r7_f008_rrt_stops_only_close_existing_half_open_courses(tmp_path, filename, database):
    mapping = next(s for s in _definition("rrt", filename)["sources"][database] if s.get("ids") == [582])
    frame = pd.DataFrame({
        "icustay_id": [1, 1, 2, 2, 3, 1, 1, 1, 1], "itemid": 582,
        "charttime": [0., 1., .5, 1.5, 2., 4., 4.5, 6., 5.],
        "value": ["CVVHD Start", "CVVHD D/C", "Hemodialysis st", "Hemodialysis end",
                  "CAVH D/C", "CAVH Start", "CAVH Start", "CAVH D/C", "CVVHD D/C"],
    })
    for projected in (False, True):
        direct = frame.rename(columns={"value": "rrt"}) if projected else frame
        out = _apply_callback(direct, ConceptSource.from_mapping(mapping), "rrt")
        assert _hours(out, "rrt") == {0., 4., 5.}
        assert set(out.loc[out.icustay_id.eq(2), "charttime"]) == {0., 1.}
        assert not out.icustay_id.eq(3).any()
        # A last start has no inferred duration beyond its observed hour.
        open_start = pd.DataFrame({"icustay_id": [1], "itemid": [582], "charttime": [8.5], "value": ["CVVHD Start"]})
        assert _apply_callback(open_start, ConceptSource.from_mapping(mapping), "rrt").charttime.tolist() == [8.5]
    if filename == "concept-dict.json":
        frame["charttime"] = ORIGIN + pd.to_timedelta(frame.charttime, unit="h")
        frame = frame.assign(subject_id=frame.icustay_id + 10, hadm_id=frame.icustay_id + 20,
                             error=0, valuenum=float("nan"), valueuom="")
        source = mimic_synthetic_source(tmp_path, database, chartevents=frame)
        for batch in (False, True):
            loaded = mimic_synthetic_load(source, ["rrt", "acute_rrt_input"], batch=batch)
            for name in ("rrt", "acute_rrt_input"):
                assert _hours(loaded[name], name) == {0., 4., 5.}
                assert _hours(loaded[name], name, stay=2) == {0., 1.}
                assert _hours(loaded[name], name, stay=3) == set()


def test_r7_f010_hirid_exact_starts_reach_every_overlapped_hour(tmp_path, monkeypatch):
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from easyicu.utils.callback_utils import hirid_vent

    for start, stop, expected in ((.5, 1.5, {0., 1.}), (0., 1., {0.}), (.5, 1., {0.}), (1., 1., set())):
        frame = pd.DataFrame({"patientid": 1, "datetime": [start, stop], "value": [1., 6.]})
        direct = hirid_vent(frame, concept_name="mech_vent", val_col="value", padding_hours=0.)
        assert set(direct.loc[direct.mech_vent.eq(1), "datetime"]) == expected
    transition = pd.DataFrame({"patientid": 1, "datetime": [.25, .5, 1.5], "value": [6., 1., 6.]})
    direct = hirid_vent(transition, concept_name="mech_vent", val_col="value", padding_hours=0.)
    assert set(direct.loc[direct.mech_vent.eq(1), "datetime"]) == {0., 1.}
    root = hirid_source_root(tmp_path, "observations", [
        {"patientid": 1, "datetime": .25, "variableid": 15001552, "value": 6.},
        {"patientid": 1, "datetime": .5, "variableid": 15001552, "value": 1.},
        {"patientid": 1, "datetime": 1.5, "variableid": 15001552, "value": 6.},
    ])
    for names in (["mech_vent"], ["mech_vent", "vent_ind"]):
        loaded = hirid_load(root, names, interval=pd.Timedelta(hours=1))
        assert _hours(loaded["mech_vent"], "mech_vent", value="invasive") == {0., 1.}
        if "vent_ind" in loaded:
            assert _hours(loaded["vent_ind"], "vent_ind") == {0., 1.}
    source = ICUDataSource(load_src_cfg("hirid"), base_path=root, enable_cache=False)
    published = _native(source, ["mech_vent", "vent_ind"], tmp_path / "native", monkeypatch)
    assert set(published.loc[published.mech_vent.eq("invasive"), "charttime"]) == {0., 1.}
    assert set(published.loc[published.vent_ind.eq(True), "charttime"]) == {0., 1.}  # noqa: E712


@pytest.mark.parametrize("numeric", [False, True])
def test_r7_f011_miiv_intervals_are_selected_by_overlap_then_clipped(tmp_path, monkeypatch, numeric):
    frame = pd.DataFrame({
        "stay_id": 1, "subject_id": 11, "hadm_id": 21, "itemid": 225792,
        "starttime": [-48., -48., -50., 1., 96., -48., 6.],
        "endtime": [2.5, -24., -49., 3.5, 97., 8., 6.],
        "statusdescription": ["FinishedRunning"] * 5 + ["Rewritten", "FinishedRunning"],
        "cancelreason": 0, "value": 1., "valueuom": "hour",
    })
    durations = frame.endtime - frame.starttime
    if not numeric:
        for column in ("starttime", "endtime"):
            frame[column] = ORIGIN + pd.to_timedelta(frame[column], unit="h")
    source = mimic_synthetic_source(tmp_path, "miiv", procedureevents=frame)
    from easyicu.table.duration import set_dur_var_unit

    controls = [0, 1, 2, 3, 4, 6]
    windows = frame.iloc[controls].assign(dur_var=durations.iloc[controls])
    set_dur_var_unit(windows, "hours")
    aligned = ConceptResolver(load_dictionary())._align_time_to_admission(
        windows, source, ["stay_id"], "starttime",
    )
    assert aligned.starttime.tolist() == [-24., 1.]
    assert aligned.endtime.tolist() == [2.5, 3.5]
    # Full source rows use the real absolute-time schema, including its void gate.
    if numeric:
        for column in ("starttime", "endtime"):
            frame[column] = ORIGIN + pd.to_timedelta(frame[column], unit="h")
    # Other start/end point sources have their own six-hour inferred courses.
    # Isolate the long pre-ICU window and its disjoint/void controls here.
    frame.iloc[[0, 1, 2, 5]].to_parquet(tmp_path / "procedureevents.parquet", index=False)
    for batch in (False, True):
        loaded = mimic_synthetic_load(source, ["mech_vent", "vent_ind"], batch=batch)
        vent = loaded["mech_vent"]
        assert _hours(vent, "mech_vent", value="invasive") == set(range(-24, 3))
        for name in ("vent_ind",):
            assert _hours(loaded[name], name) == set(range(-24, 3))
    published = _native(source, ["mech_vent", "vent_ind"], tmp_path / "native", monkeypatch)
    assert set(published.loc[published.mech_vent.eq("invasive"), "charttime"]) == set(range(-24, 3))
    for name in ("vent_ind",):
        assert set(published.loc[published[name].eq(True), "charttime"]) == set(range(-24, 3))  # noqa: E712


@pytest.mark.parametrize(("name", "item", "unit"), [("adh_rate", 30051, "units/min"), ("phn_rate", 30128, "mcg/kg/min")])
def test_r7_f012_carevue_adjunct_support_uses_the_rate_clock(tmp_path, name, item, unit):
    frame = pd.DataFrame({
        "icustay_id": 1, "subject_id": 11, "hadm_id": 21, "itemid": item,
        "charttime": ORIGIN + pd.to_timedelta([0., 2., 10., 12., 20.], unit="h"),
        "rate": [.01, .02, .01, .01, .01], "rateuom": unit,
        "amount": float("nan"), "amountuom": "", "linkorderid": 1,
        "stopped": [None, None, None, "Stopped", None],
    })
    source = mimic_synthetic_source(tmp_path, "mimic", inputevents_cv=frame)
    expected = set(range(8)) | {10., 11., 20.}
    for batch in (False, True):
        loaded = mimic_synthetic_load(source, [name, "other_vaso", "vaso_ind"], batch=batch)
        rate = loaded[name]
        positive = set(rate.data.loc[rate.data[name].gt(0), rate.index_column])
        assert _hours(loaded["other_vaso"], "other_vaso") == positive == expected
        assert _hours(loaded["vaso_ind"], "vaso_ind") == expected


@pytest.mark.parametrize("name", ["abx", "delirium_tx_proxy", "aspirin", "dexamethasone"])
def test_r7_f030_miiv_prescriptions_require_positive_evidence_and_valid_intervals(tmp_path, name):
    filename = "sofa2-dict.json" if name == "delirium_tx_proxy" else "concept-dict.json"
    mapping = next(s for s in _definition(name, filename)["sources"]["miiv"] if s["table"] == "prescriptions")
    source_spec = ConceptSource.from_mapping(mapping)
    drug = {"abx": "Meropenem", "delirium_tx_proxy": "Haloperidol", "aspirin": "Aspirin", "dexamethasone": "Dexamethasone"}[name]
    doses = ["0", "-1", None, "2", "2", "2", "2", "2", None, "2", ".5", "1e309", "0-2", "1-2", "2", "0", "-1"]
    starts = pd.Series(range(len(doses)), dtype=float) * 4
    stops = starts + 1.5
    stops.iloc[4:7] = [starts.iloc[4], starts.iloc[5] - 1, float("nan")]
    stops.iloc[14] = starts.iloc[14] + 1
    frame = pd.DataFrame({
        "stay_id": 1, "subject_id": 11, "hadm_id": 21, "drug": drug,
        "dose_val_rx": doses, "dose_unit_rx": "mg", "route": "PO",
        "starttime": starts, "stoptime": stops,
        "statusdescription": [None] * 7 + ["Not given", "Given", "Held"] + [None] * 5 + ["Given", "Given"],
    })
    for projected in (False, True):
        direct = frame.rename(columns={source_spec.value_var: name}) if projected else frame
        out = _apply_callback(direct, source_spec, name)
        active = out.loc[out[name].eq(True), "starttime"]  # noqa: E712
        assert set(active) == {12., 13., 32., 33., 40., 41., 52., 53., 56.}
    assert {"dose_val_rx", "stoptime"} <= set(mapping["extra_vars"])
    # Execution status is optional in direct evidence; prescriptions themselves
    # have no execution-status column. Their dose and both clocks must survive.
    frame = frame.drop(columns="statusdescription")
    for column in ("starttime", "stoptime"):
        frame[column] = ORIGIN + pd.to_timedelta(frame[column], unit="h")
    source = mimic_synthetic_source(tmp_path, "miiv", prescriptions=frame)
    expected = {12., 13., 28., 29., 36., 37., 40., 41., 52., 53., 56.}
    for batch in (False, True):
        loaded = mimic_synthetic_load(source, [name, "los_icu"], batch=batch)
        assert _hours(loaded[name], name) == expected
