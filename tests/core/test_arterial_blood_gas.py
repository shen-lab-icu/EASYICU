"""Invented temporary MIIV events only; no raw database or export access."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.io.arterial_blood_gas import extract_miiv_arterial_blood_gas
from easyicu.io.source_events import SourceEventContractError

ORIGIN = pd.Timestamp("2180-01-01")


def lab(
    item, value, *, specimen=900, time=25, stored=26, subject=1, hadm=100, native=1
):
    return dict(
        subject_id=subject,
        hadm_id=hadm,
        itemid=item,
        specimen_id=specimen,
        labevent_id=native,
        charttime=ORIGIN + pd.Timedelta(hours=time),
        storetime=pd.NaT if stored is None else ORIGIN + pd.Timedelta(hours=stored),
        valuenum=float(value) if item != 52033 and value is not None else np.nan,
        value=None if value is None else str(value),
        valueuom="mmHg" if item == 50821 else "%" if item == 50816 else None,
    )


def chart(value=0.4, *, time=24, stored=25, subject=1, hadm=100, stay=20):
    return dict(
        subject_id=subject,
        hadm_id=hadm,
        stay_id=stay,
        itemid=223835,
        charttime=ORIGIN + pd.Timedelta(hours=time),
        storetime=pd.NaT if stored is None else ORIGIN + pd.Timedelta(hours=stored),
        valuenum=value,
        value=str(value),
        valueuom="%",
    )


def source(tmp_path, *, labs=None, charts=None, clocks=None):
    if clocks is None:
        clocks = [(1, 100, 10, 0, 10), (1, 100, 20, 20, 40), (2, 200, 30, 0, 50)]
    frame = pd.DataFrame(
        clocks, columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
    )
    for name in ["intime", "outtime"]:
        frame[name] = frame[name].map(
            lambda x: pd.NaT if pd.isna(x) else ORIGIN + pd.Timedelta(hours=x)
        )
    frame.to_parquet(tmp_path / "icustays.parquet", index=False)
    labs = (
        labs
        if labs is not None
        else [lab(50821, 80), lab(52033, "ART.", native=2), lab(50816, 0.5, native=3)]
    )
    charts = charts if charts is not None else [chart()]
    for table, rows, extra in [
        ("labevents", labs, ["specimen_id", "labevent_id"]),
        ("chartevents", charts, ["stay_id"]),
    ]:
        (tmp_path / table).mkdir()
        frame = pd.DataFrame(rows)
        for name in ["subject_id", "hadm_id", "itemid", *extra]:
            frame[name] = frame[name].astype("Int64")
        frame.to_parquet(tmp_path / table / "part.parquet", index=False)
    return ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)


def extract(tmp_path, **kwargs):
    return extract_miiv_arterial_blood_gas(
        source(tmp_path, **kwargs), allowed_stay_ids=[20]
    )


def test_two_routes_preserve_sources_and_storetime():
    # Explicit temporary source is supplied by the caller in every real test.
    with pytest.raises(SourceEventContractError):
        extract_miiv_arterial_blood_gas(
            ICUDataSource(
                load_src_cfg("miiv"),
                base_path=Path("/nonexistent-synthetic"),
                enable_cache=False,
            ),
            allowed_stay_ids=[],
        )


def test_direct_arterial_two_routes(tmp_path):
    result = extract(tmp_path)
    assert result.po2.specimen_state.tolist() == ["DIRECT_ARTERIAL"]
    assert len(result.events) == 4 and len(result.pairs) == 2
    pairs = result.pairs.set_index("pairing_route")
    assert pairs.pafi_mmhg.to_dict() == {
        "SAME_SPECIMEN_LAB": 160.0,
        "PRIOR_CHART_4H": 200.0,
    }
    assert pairs.available_at.eq(ORIGIN + pd.Timedelta(hours=26)).all()
    assert pairs.availability_status.eq("KNOWN").all()
    assert set(result.clock_context.stay_id) == {20}


@pytest.mark.parametrize(
    "types,state",
    [
        ([], "TYPE_MISSING"),
        ([""], "TYPE_EMPTY"),
        ([None], "TYPE_EMPTY"),
        (["ART"], "OTHER_RECORDED_TYPE"),
        (["VEN."], "OTHER_RECORDED_TYPE"),
        (["ART.", "VEN."], "CONFLICTING_TYPES"),
        ([" art. ", "ART."], "DIRECT_ARTERIAL"),
        (["ART.", ""], "DIRECT_ARTERIAL"),
    ],
)
def test_type_states_never_infer_from_po2(tmp_path, types, state):
    labs = [lab(50821, 80), lab(50816, 0.5, native=2)] + [
        lab(52033, t, native=100 + i) for i, t in enumerate(types)
    ]
    result = extract(tmp_path, labs=labs)
    assert result.po2.specimen_state.tolist() == [state]
    assert result.po2.arterial_certified.item() == (state == "DIRECT_ARTERIAL")
    if state != "DIRECT_ARTERIAL":
        assert result.pairs.pafi_mmhg.isna().all()


def test_null_specimen_ids_never_link_by_same_time(tmp_path):
    result = extract(
        tmp_path,
        labs=[
            lab(50821, 80, specimen=None),
            lab(52033, "ART.", specimen=None, native=2),
            lab(50816, 0.5, specimen=None, native=3),
        ],
    )
    assert result.po2.specimen_state.tolist() == ["NO_SPECIMEN_ID"]
    assert result.pairs.pafi_mmhg.isna().all()
    assert (
        result.pairs.loc[
            result.pairs.pairing_route.eq("SAME_SPECIMEN_LAB"), "fio2_event_key"
        ]
        .isna()
        .all()
    )


def test_multiple_physical_po2_and_fio2_not_max_pooled(tmp_path):
    src = source(
        tmp_path,
        labs=[
            lab(50821, 80),
            lab(50821, 100, native=2),
            lab(52033, "ART.", native=3),
            lab(50816, 0.4, native=4),
            lab(50816, 0.8, native=5),
        ],
    )
    p = tmp_path / "labevents/part.parquet"
    pd.read_parquet(p).iloc[[0]].to_parquet(
        tmp_path / "labevents/duplicate.parquet", index=False
    )
    result = extract_miiv_arterial_blood_gas(src, allowed_stay_ids=[20])
    assert len(result.po2) == 3 and result.po2.po2_distinct_valid_values.eq(2).all()
    pairs = result.pairs.loc[result.pairs.pairing_route.eq("SAME_SPECIMEN_LAB")]
    assert len(pairs) == 6
    assert sorted(pairs.pafi_mmhg) == [100, 100, 125, 200, 200, 250]
    assert result.events.event_key.nunique() == len(result.events)


def test_chart_four_hour_boundary_latest_ties_and_no_future(tmp_path):
    result = extract(
        tmp_path,
        charts=[
            chart(0.4, time=21),
            chart(0.8, time=21),
            chart(0.9, time=20.999),
            chart(0.7, time=26),
        ],
    )
    pairs = result.pairs.loc[result.pairs.pairing_route.eq("PRIOR_CHART_4H")]
    assert len(pairs) == 2 and pairs.fio2_candidate_count.eq(2).all()
    assert pairs.fio2_distinct_valid_values.eq(2).all()
    assert sorted(pairs.pafi_mmhg) == [100, 200]
    assert pairs.fio2_minus_po2_hours.eq(-4).all()


def test_chart_different_stay_admission_patient_never_matches(tmp_path):
    src = source(
        tmp_path,
        charts=[
            chart(0.4, stay=10),
            chart(0.8, stay=30, subject=2, hadm=200),
            chart(0.9, stay=20, hadm=999),
        ],
    )
    result = extract_miiv_arterial_blood_gas(src, allowed_stay_ids=[10, 20, 30])
    pairs = result.pairs.loc[result.pairs.pairing_route.eq("PRIOR_CHART_4H")]
    assert pairs.pair_status.tolist() == ["NO_MATCH"]


def test_unreturned_specimen_type_cannot_make_false_arterial(tmp_path):
    result = extract(
        tmp_path,
        labs=[
            lab(50821, 80),
            lab(52033, "ART.", native=2),
            lab(52033, "VEN.", time=5, native=3),
            lab(50816, 0.5, native=4),
        ],
    )
    assert result.po2.specimen_state.tolist() == ["INCOMPLETE_ALLOWED_TYPE_EVIDENCE"]
    assert result.po2.unreturned_type_rows.tolist() == [1]
    assert "VEN." not in result.events.raw_value.tolist()
    assert result.pairs.pafi_mmhg.isna().all()


def test_cross_admission_specimen_collision_is_flag_not_value_leak(tmp_path):
    result = extract(
        tmp_path,
        labs=[
            lab(50821, 80),
            lab(52033, "ART.", native=2),
            lab(52033, "VEN.", subject=2, hadm=200, native=3),
            lab(50816, 0.5, native=4),
        ],
    )
    assert result.po2.specimen_state.tolist() == ["IDENTITY_CONFLICT"]
    assert set(result.events.subject_id) == {1}
    assert "VEN." not in result.events.raw_value.tolist()
    assert result.pairs.pafi_mmhg.isna().all()


@pytest.mark.parametrize("which", ["po2", "type", "lab", "chart"])
def test_any_required_missing_storetime_keeps_availability_unknown(tmp_path, which):
    items = [
        lab(50821, 80, stored=None if which == "po2" else 26),
        lab(52033, "ART.", native=2, stored=None if which == "type" else 26),
        lab(50816, 0.5, native=3, stored=None if which == "lab" else 26),
    ]
    result = extract(
        tmp_path, labs=items, charts=[chart(stored=None if which == "chart" else 25)]
    )
    affected = (
        result.pairs
        if which in ["po2", "type"]
        else result.pairs.loc[
            result.pairs.pairing_route.eq(
                "SAME_SPECIMEN_LAB" if which == "lab" else "PRIOR_CHART_4H"
            )
        ]
    )
    assert affected.available_at.isna().all()
    assert affected.availability_status.eq("MISSING").all()
    assert (
        affected.pafi_mmhg.notna().all()
    )  # retrospective arithmetic, not available prediction


def test_store_before_chart_not_repaired(tmp_path):
    result = extract(
        tmp_path,
        labs=[
            lab(50821, 80, stored=24),
            lab(52033, "ART.", native=2),
            lab(50816, 0.5, native=3),
        ],
    )
    assert result.pairs.available_at.isna().all()
    assert result.pairs.availability_status.eq("STORE_BEFORE_CHART").all()
    assert result.events.loc[result.events.itemid.eq(50821), "raw_storetime"].iloc[
        0
    ] == ORIGIN + pd.Timedelta(hours=24)


def test_unknown_exit_reuses_v2_identifiable_assignment(tmp_path):
    result = extract(tmp_path, clocks=[(1, 100, 10, 0, 10), (1, 100, 20, 20, None)])
    assert result.po2.arterial_certified.tolist() == [True]
    assert (
        result.events.loc[
            result.events.source_table.eq("labevents"), "lab_assignment_status"
        ]
        .eq("identified_under_unknown_outtime")
        .all()
    )
    assert result.events.temporal_position.eq("outtime_unknown").all()


def test_dictionary_bounds_preserve_raw_invalid_values(tmp_path):
    result = extract(
        tmp_path,
        labs=[lab(50821, 10), lab(52033, "ART.", native=2), lab(50816, 0.1, native=3)],
    )
    assert result.po2.arterial_certified.tolist() == [True]
    assert result.pairs.pafi_mmhg.isna().all()
    assert 10 in result.events.raw_valuenum.tolist()


def test_empty_allowed_po2_keeps_schema(tmp_path):
    result = extract(
        tmp_path,
        labs=[
            lab(50821, 80, time=5),
            lab(52033, "ART.", time=5, native=2),
            lab(50816, 0.5, time=5, native=3),
        ],
    )
    assert result.po2.empty and result.pairs.empty
    assert "specimen_state" in result.po2 and "pafi_mmhg" in result.pairs


def test_many_specimens_and_event_permutation_preserve_all_pairs(tmp_path):
    from easyicu.io.arterial_blood_gas import _pair_tables

    labs = []
    for i in range(30):
        spec = 900 + i
        time = 25 + i % 8
        labs.extend(
            [
                lab(
                    50821,
                    80 + i,
                    specimen=spec,
                    time=time,
                    stored=time + 1,
                    native=i * 3 + 1,
                ),
                lab(
                    52033,
                    "ART.",
                    specimen=spec,
                    time=time,
                    stored=time + 1,
                    native=i * 3 + 2,
                ),
                lab(
                    50816,
                    0.5,
                    specimen=spec,
                    time=time,
                    stored=time + 1,
                    native=i * 3 + 3,
                ),
            ]
        )
    charts = [chart(v, time=t, stored=t + 1) for t in range(20, 34) for v in [0.4, 0.6]]
    result = extract(tmp_path, labs=labs, charts=charts)
    audit = pd.DataFrame(
        [
            dict(
                specimen_id=900 + i,
                identity_conflict=False,
                total_type_rows=1,
                total_fio2_rows=1,
                unreturned_type_rows=0,
                unreturned_fio2_rows=0,
            )
            for i in range(30)
        ]
    )
    po2, pairs = _pair_tables(
        result.events.sample(frac=1, random_state=739).reset_index(drop=True), audit
    )

    def ordered(frame, keys):
        return frame.sort_values(keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        ordered(po2, ["po2_event_key"]), ordered(result.po2, ["po2_event_key"])
    )
    pd.testing.assert_frame_equal(
        ordered(pairs, ["po2_event_key", "pairing_route", "fio2_event_key"]),
        ordered(result.pairs, ["po2_event_key", "pairing_route", "fio2_event_key"]),
    )
    assert len(pairs) == 90


def test_source_python_boundary_excludes_other_stay_values(tmp_path, monkeypatch):
    from easyicu.io import source_events

    real = source_events.duckdb.connect
    calls = []

    class Spy:
        def __init__(self, conn):
            self.conn = conn
            self.sql = ""

        def execute(self, sql, *a, **kw):
            self.sql = sql
            self.conn.execute(sql, *a, **kw)
            return self

        def fetchdf(self, *a, **kw):
            result = self.conn.fetchdf(*a, **kw)
            if "raw_value" in result:
                assert "identity_status='identity_allowed'" in self.sql
                assert set(result.stay_id) <= {20}
                assert "VEN." not in result.raw_value.tolist()
                calls.append("allowed_values")
            else:
                assert "raw_valuenum" not in result
                calls.append("metadata")
            return result

        def __getattr__(self, key):
            return getattr(self.conn, key)

    monkeypatch.setattr(
        source_events.duckdb, "connect", lambda *a, **kw: Spy(real(*a, **kw))
    )
    result = extract(
        tmp_path,
        labs=[
            lab(50821, 80),
            lab(52033, "ART.", native=2),
            lab(52033, "VEN.", time=5, native=3),
            lab(50816, 0.5, native=4),
        ],
    )
    assert calls == ["metadata", "allowed_values", "metadata", "allowed_values"]
    assert result.po2.specimen_state.tolist() == ["INCOMPLETE_ALLOWED_TYPE_EVIDENCE"]


def test_no_cross_identity_physical_pair_even_when_both_stays_allowed(tmp_path):
    src = source(
        tmp_path,
        labs=[
            lab(50821, 80),
            lab(52033, "ART.", native=2),
            lab(50816, 0.9, subject=2, hadm=200, native=3),
        ],
    )
    result = extract_miiv_arterial_blood_gas(src, allowed_stay_ids=[20, 30])
    assert result.po2.specimen_state.tolist() == ["IDENTITY_CONFLICT"]
    lab_pair = result.pairs.loc[result.pairs.pairing_route.eq("SAME_SPECIMEN_LAB")]
    assert lab_pair.fio2_event_key.isna().all()


def test_source_mutation_is_detected_before_result_returns(tmp_path, monkeypatch):
    from easyicu.io import arterial_blood_gas as module

    src = source(tmp_path)
    original = module._pair_tables

    def altered(*a):
        result = original(*a)
        p = tmp_path / "chartevents/part.parquet"
        frame = pd.read_parquet(p)
        frame.loc[0, "valuenum"] = 0.9
        frame.to_parquet(p, index=False)
        return result

    monkeypatch.setattr(module, "_pair_tables", altered)
    with pytest.raises(SourceEventContractError, match="source_changed"):
        extract_miiv_arterial_blood_gas(src, allowed_stay_ids=[20])
