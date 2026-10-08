"""Only invented temporary Parquet: no real events, routes or exports."""

import numpy as np
import pandas as pd
import pytest

from easyicu.io import scoped_measurements as api
from easyicu.io.source_events import SourceEventContractError
import importlib.util
from pathlib import Path

_fixture_spec = importlib.util.spec_from_file_location(
    "measurement_synthetic_fixture",
    Path(__file__).with_name("test_arterial_blood_gas.py"),
)
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
source, lab, chart = _fixture.source, _fixture.lab, _fixture.chart


def lab_event(item=50912, value=1.5, **kwargs):
    x = lab(item, value, **kwargs)
    x["valueuom"] = "mg/dL"
    return x


def chart_event(item=220052, value=65, **kwargs):
    x = chart(value, **kwargs)
    x["itemid"] = item
    x["valueuom"] = "mg/dL" if item in [220615, 229761] else "mmHg"
    return x


def extract(tmp_path, allowed=(20,), concepts=api.CONCEPTS, **kwargs):
    return api.extract_miiv_measurement_events(
        source(tmp_path, **kwargs), allowed_stay_ids=allowed, concepts=concepts
    )


def test_full_dictionary_block_channels_and_duplicate_events(tmp_path):
    labs = [lab_event(i, native=j) for j, i in enumerate([50912, 52546, 52024])]
    labs.append(labs[0].copy())
    charts = [
        chart_event(i, 1 if i in [220615, 229761] else 65 if i != 220074 else 8)
        for i in [220615, 229761, 220052, 220181, 225312, 220074]
    ]
    r = extract(tmp_path, labs=labs, charts=charts)
    assert len(r.events) == 10
    assert r.events.event_key.nunique() == 10
    assert r.events.retained_for_analysis.all()
    assert set(r.events.source_item_id) == {i for _, i in api.CHANNELS}
    assert set(r.events.source_channel) == {v[1] for v in api.CHANNELS.values()}
    assert r.events.loc[r.events.source_item_id.eq(50912), "labevent_id"].tolist() == [
        0,
        0,
    ]
    assert r.receipt["aggregation"].startswith("none")


@pytest.mark.parametrize(
    "stored,status,order",
    [
        (None, "missing", "unknown"),
        (24, "known", "store_before_chart"),
        (25, "known", "store_at_or_after_chart"),
        (30, "known", "store_at_or_after_chart"),
    ],
)
def test_clocks_do_not_remove_values(tmp_path, stored, status, order):
    r = extract(
        tmp_path,
        concepts=["crea"],
        labs=[lab_event(stored=stored)],
        charts=[chart_event(220615, 1.5)],
    )
    x = r.events.loc[r.events.source_table.eq("labevents")].iloc[0]
    assert x.storetime_status == status and x.clock_order_status == order
    assert x.retained_for_analysis and x.converted_value == 1.5
    if stored is not None:
        assert x.store_delay_hours == stored - 25
        assert x.store_hours == stored - 20
    assert "available_at" not in r.events


@pytest.mark.parametrize(
    "stored", ["2180-01-02 02:00:00 UTC", "2180-01-02 02:00:00+00:00", "nonsense"]
)
def test_invalid_store_strings_preserved_unknown(tmp_path, stored):
    row = lab_event()
    row["storetime"] = stored
    r = extract(tmp_path, concepts=["crea"], labs=[row])
    x = r.events.iloc[0]
    assert x.raw_storetime == stored and x.storetime_status == "invalid"
    assert pd.isna(x.storetime) and x.retained_for_analysis


@pytest.mark.parametrize(
    "unit,status,retained",
    [
        ("mg/dL", "declared_match", True),
        (None, "missing_assumed_dictionary", True),
        ("", "missing_assumed_dictionary", True),
        ("umol/L", "mismatch", False),
    ],
)
def test_unit_contract_without_silent_conversion(tmp_path, unit, status, retained):
    row = lab_event()
    row["valueuom"] = unit
    x = extract(tmp_path, concepts=["crea"], labs=[row]).events.iloc[0]
    assert x.unit_status == status and bool(x.retained_for_analysis) == retained
    assert x.converted_value == 1.5


@pytest.mark.parametrize(
    "item,value,within",
    [
        (50912, 0, True),
        (50912, 25, True),
        (50912, 25.1, False),
        (220052, 250, True),
        (220052, 251, False),
        (220074, -5, True),
        (220074, 50, True),
        (220074, -5.1, False),
    ],
)
def test_exact_dictionary_bounds(tmp_path, item, value, within):
    r = extract(
        tmp_path,
        labs=[lab_event(item, value)] if item == 50912 else [lab_event()],
        charts=[chart_event(item, value)] if item != 50912 else [chart_event()],
    )
    x = r.events.loc[r.events.source_item_id.eq(item)].iloc[0]
    assert bool(x.retained_for_analysis) == within
    assert x.converted_value == value


def test_lab_explicit_valuenum_and_chart_fallback(tmp_path):
    a = lab_event()
    a["valuenum"] = np.nan
    a["value"] = "1.7"
    b = chart_event(220615, 1.5)
    b["valuenum"] = np.nan
    b["value"] = "1.7"
    r = extract(tmp_path, concepts=["crea"], labs=[a], charts=[b])
    l = r.events.loc[r.events.source_table.eq("labevents")].iloc[0]
    c = r.events.loc[r.events.source_table.eq("chartevents")].iloc[0]
    assert (
        l.numeric_source == "valuenum"
        and not l.retained_for_analysis
        and pd.isna(l.converted_value)
    )
    assert c.numeric_source == "value" and c.converted_value == 1.7


def test_full_context_before_allowlist_and_zero_events(tmp_path):
    r = extract(
        tmp_path,
        allowed=[20, 30],
        labs=[lab_event(time=5, value=999), lab_event(time=25, value=1)],
        charts=[chart_event(stay=10, time=5, value=888)],
    )
    assert set(r.events.stay_id) == {20} and 999 not in set(r.events.converted_value)
    assert list(r.clock_context.stay_id) == [20, 30]
    assert any(
        x["reason"] == "assigned_stay_not_allowed" and x["count"] == 1
        for x in r.receipt["source_filter_counts"]
    )


def test_unknown_exit_reuses_unique_assignment(tmp_path):
    r = extract(
        tmp_path,
        clocks=[(1, 100, 10, 0, 10), (1, 100, 20, 20, None)],
        labs=[lab_event(time=25)],
        charts=[chart_event()],
    )
    assert r.events.lab_assignment_status.isin(
        ["native_stay_id", "identified_under_unknown_outtime"]
    ).all()
    assert r.events.temporal_position.eq("outtime_unknown").all()
    assert r.clock_context.outtime_status.eq("unknown").all()


def test_multiple_unknown_lab_ambiguous_native_chart_retained(tmp_path):
    r = extract(
        tmp_path,
        clocks=[(1, 100, 10, 0, None), (1, 100, 20, 20, None)],
        labs=[lab_event(time=25)],
        charts=[chart_event()],
    )
    assert r.events.source_table.eq("chartevents").all()
    assert any(
        x["reason"] == "unknown_outtime_assignment"
        for x in r.receipt["source_filter_counts"]
    )


def test_empty_events_schema_and_zero_clock_ledger(tmp_path):
    r = extract(tmp_path, labs=[lab(50821, 80)], charts=[chart()])
    assert r.events.empty and len(r.clock_context) == 1
    assert set(api.EXTRA_COLUMNS) <= set(r.events)


@pytest.mark.parametrize("allowed", [[], [20, 20], [True], [20.0]])
def test_bad_roster_rejected(tmp_path, allowed):
    with pytest.raises(SourceEventContractError):
        extract(tmp_path, allowed=allowed)


def test_subset_does_not_require_unused_lab_table(tmp_path):
    ds = source(tmp_path, charts=[chart_event(220074, 8)])
    for p in (tmp_path / "labevents").glob("*"):
        p.unlink()
    (tmp_path / "labevents").rmdir()
    r = api.extract_miiv_measurement_events(ds, allowed_stay_ids=[20], concepts=["cvp"])
    assert len(r.events) == 1 and r.events.iloc[0].concept == "cvp"


def test_all_clinical_fetches_sql_restricted(tmp_path, monkeypatch):
    original = api.duckdb.connect
    queries = []

    class Wrapped:
        def __init__(self, c):
            self.c = c
            self.sql = ""

        def execute(self, sql, *a, **kw):
            self.sql = sql
            self.c.execute(sql, *a, **kw)
            return self

        def fetchdf(self):
            queries.append(self.sql)
            if "SELECT * EXCLUDE(identity_status)" in self.sql:
                assert "WHERE identity_status='identity_allowed'" in self.sql
            return self.c.fetchdf()

        def __getattr__(self, n):
            return getattr(self.c, n)

    monkeypatch.setattr(
        api.duckdb, "connect", lambda *a, **kw: Wrapped(original(*a, **kw))
    )
    r = extract(
        tmp_path,
        labs=[lab_event(), lab_event(subject=2, hadm=200, value=999)],
        charts=[chart_event()],
    )
    assert len(queries) == 3 and set(r.events.stay_id) == {20}


def test_source_mutation_fails_closed(tmp_path, monkeypatch):
    old = api._decorate

    def mutate(events, specs, hashes, sha):
        p = tmp_path / "labevents/part.parquet"
        p.write_bytes(p.read_bytes() + b"altered")
        return old(events, specs, hashes, sha)

    monkeypatch.setattr(api, "_decorate", mutate)
    with pytest.raises(SourceEventContractError, match="source_changed"):
        extract(tmp_path, labs=[lab_event()], charts=[chart_event()])
