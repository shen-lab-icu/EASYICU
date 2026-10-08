"""Only self-created temporary MIIV parquet; no clinical data/environment reads."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.io.source_events import (
    CONCEPTS, LAB_ASSIGNMENT, SourceEventContractError,
    extract_miiv_respiratory_events,
)

ORIGIN = pd.Timestamp("2180-01-01")


def write_source(tmp_path, *, clocks=None, chart=None, lab=None):
    if clocks is None:
        clocks = [(101, 501, 10, 0, 10), (101, 501, 20, 20, 30),
                  (101, 502, 30, 0, 40), (102, 503, 40, 0, 40)]
    clock_frame = pd.DataFrame(clocks, columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    for column in ["intime", "outtime"]:
        clock_frame[column] = clock_frame[column].map(
            lambda t: None if pd.isna(t) else str(ORIGIN + pd.Timedelta(hours=t)))
    clock_frame.to_parquet(tmp_path / "icustays.parquet", index=False)
    chart = chart if chart is not None else [(101, 501, 20, 21, 220277, 97., "97", "%")]
    lab = lab if lab is not None else [(101, 501, 21, 50816, .5, "0.5", "%", 1, 11)]
    for table, rows, columns in [
        ("chartevents", chart, ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "value", "valueuom"]),
        ("labevents", lab, ["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "value", "valueuom", "labevent_id", "specimen_id"]),
    ]:
        frame = pd.DataFrame(rows, columns=columns)
        for name in ["subject_id", "hadm_id", "itemid"] + (["stay_id"] if table == "chartevents" else []):
            frame[name] = frame[name].astype("Int64")
        frame["charttime"] = frame.charttime.map(lambda t: pd.NaT if pd.isna(t) else ORIGIN + pd.Timedelta(hours=t))
        frame["storetime"] = frame.charttime + pd.Timedelta(hours=2)
        folder = tmp_path / table
        folder.mkdir()
        frame.to_parquet(folder / "part.parquet", index=False)
    return ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)


def test_full_hospital_assignment_precedes_exact_allowed_stay(tmp_path):
    lab = [(101, 501, t, 50816, .5, ".5", "%", i, 100+i)
           for i, t in enumerate([-1, 8, 10, 15, 20, 30, 31])]
    # Other admissions/patients may not enter even with matching clock times.
    lab += [(101, 502, 21, 50816, .99, ".99", "%", 91, 191),
            (102, 503, 21, 50816, .99, ".99", "%", 92, 192),
            (101, None, 21, 50816, .99, ".99", "%", 93, 193),
            (101, 501, None, 50816, .99, ".99", "%", 94, 194),
            (999, 501, 21, 50816, .99, ".99", "%", 95, 195)]
    source = write_source(tmp_path, lab=lab)
    result = extract_miiv_respiratory_events(source, allowed_stay_ids=[20], concepts=["fio2_lab"])
    trace = result.trace
    assert trace.labevent_id.tolist() == [3, 4, 5, 6]
    assert trace.stay_id.tolist() == [20] * 4
    assert trace.temporal_position.tolist() == ["before_intime", "within_icu", "at_outtime", "after_outtime"]
    assert trace.hour_bucket.tolist() == [-5., 0., 10., 11.]
    assert trace.assignment_rule.eq(LAB_ASSIGNMENT).all()
    counts = {r["reason"]: r["count"] for r in result.receipt["source_filter_counts"]}
    assert counts == {"assigned_stay_not_allowed": 3, "identity_allowed": 4,
                      "identity_mismatch": 1, "missing_or_invalid_event_time": 1}
    assert "not quantified" in result.receipt["missing_hadm_contribution"]
    assert (trace.raw_storetime - trace.raw_charttime).dt.total_seconds().eq(7200).all()


def test_first_stay_only_does_not_capture_later_events(tmp_path):
    source = write_source(tmp_path, lab=[(101, 501, t, 50816, .5, ".5", "%", i, i)
                                         for i, t in enumerate([-1, 10, 15, 31])])
    result = extract_miiv_respiratory_events(source, allowed_stay_ids=[10], concepts=["fio2_lab"])
    assert result.trace.labevent_id.tolist() == [0, 1]
    assert result.trace.temporal_position.tolist() == ["before_intime", "at_outtime"]


def test_event_first_conversion_and_trace_preserve_all_null_hours(tmp_path):
    values = [(0, .4, "wrong"), (0, .6, "wrong"), (0, 50., "wrong"),
              (1, None, "50%"), (1, None, "0.4"), (2, 0., "80%"),
              (3, 1., "wrong"), (4, 21., "wrong"), (5, 100., "wrong"),
              (6, np.inf, "40%"), (7, None, None)]
    chart = [(101, 501, 20, 20 + h, 223835, val, text, "%") for h, val, text in values]
    source = write_source(tmp_path, chart=chart)
    result = extract_miiv_respiratory_events(source, allowed_stay_ids=[20], concepts=["fio2_chart"])
    hourly = result.hourly.set_index("charttime").fio2_chart
    assert hourly.loc[0] == 50.  # median(.4,.6,50) then percent callback would be 60.
    assert hourly.loc[1] == 45.
    assert hourly.loc[[3, 4, 5]].tolist() == [100., 21., 100.]
    assert hourly.loc[[2, 6, 7]].isna().all()
    trace = result.trace
    text = trace.loc[trace.callback_input.eq("50%")].iloc[0]
    assert pd.isna(text.numeric_value) and text.converted_value == 50.
    infinite = trace.loc[trace.hour_bucket.eq(6)].iloc[0]
    assert np.isinf(infinite.numeric_value) and np.isinf(infinite.converted_value)
    assert not infinite.retained_for_aggregation and infinite.numeric_source == "valuenum"
    assert trace.loc[trace.hour_bucket.eq(2), "bounds_status"].tolist() == ["outside"]
    assert result.receipt["concept_counts"][0]["all_null_hours"] == 3


def test_distinct_physical_duplicate_events_keep_median_weight(tmp_path):
    lab = [(101, 501, 21, 50816, .4, ".4", "%", 1, 10),
           (101, 501, 21, 50816, .8, ".8", "%", 2, 20)]
    source = write_source(tmp_path, lab=lab)
    path = tmp_path / "labevents/part.parquet"
    pd.read_parquet(path).iloc[[0]].to_parquet(tmp_path / "labevents/duplicate.parquet", index=False)
    result = extract_miiv_respiratory_events(source, allowed_stay_ids=[20], concepts=["fio2_lab"])
    assert len(result.trace) == 3 and result.trace.event_key.nunique() == 3
    assert result.trace.labevent_id.tolist().count(1) == 2
    assert result.hourly.fio2_lab.tolist() == [40.]
    assert result.trace.aggregate_n.tolist() == [3] * 3


def test_all_channels_and_native_chart_boundary(tmp_path):
    chart = [(101, 501, 20, 21, item, value, str(value), unit) for item, value, unit in
             [(220277, 97., "%"), (226253, 70., "%"), (220339, 0., "cmH2O"),
              (224700, 16., "cmH2O"), (223835, .4, "%")]]
    chart += [(101, 501, 10, 21, 220277, 55., "55", "%"),
              (102, 501, 20, 21, 220277, 56., "56", "%")]
    source = write_source(tmp_path, chart=chart)
    result = extract_miiv_respiratory_events(source, allowed_stay_ids=[20])
    assert list(result.hourly) == ["stay_id", "charttime", *CONCEPTS]
    row = result.hourly.iloc[0]
    assert [row.spo2, row.o2sat, row.peep_set, row.peep_total, row.fio2_chart, row.fio2_lab] == [97., 97., 0., 16., 40., 50.]
    assert result.trace.source_item_id.ne(226253).all()
    assert result.trace.loc[result.trace.source_table.eq("chartevents"), "assignment_rule"].eq("native_stay_id").all()
    assert all(len(r["sha256"]) == 64 for r in result.receipt["source_files"])
    assert set(result.trace.source_file_sha256) <= {r["sha256"] for r in result.receipt["source_files"]}


@pytest.mark.parametrize("clocks,reason", [
    ([(101, 501, 10, 0, 30), (101, 501, 20, 20, 30)], "ambiguous_outtime"),
    ([(101, 501, 10, 0, None), (101, 501, 20, 20, 30)], "invalid_identity_or_clock"),
    ([(999, 501, 10, 0, 10), (101, 501, 20, 20, 30)], "hospital_subject_conflict"),
    ([(101, 501, 20, 20, 30), (101, 501, 20, 20, 30)], "duplicate_stay_id"),
    ([(101, 501, 10, 0, 10)], "missing_requested_stay"),
])
def test_bad_complete_identity_context_fails_before_clinical_read(tmp_path, monkeypatch, clocks, reason):
    source = write_source(tmp_path, clocks=clocks)
    import easyicu.io.source_events as module
    monkeypatch.setattr(module, "_read_scoped_events", lambda *_: pytest.fail("clinical source read before clock rejection"))
    with pytest.raises(SourceEventContractError) as error:
        extract_miiv_respiratory_events(source, allowed_stay_ids=[20])
    assert error.value.counts[reason] > 0


@pytest.mark.parametrize("ids", [None, [], [True], [20.], [20, 20], ["20"]])
def test_invalid_allow_list_fails_closed(tmp_path, ids):
    source = ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)
    with pytest.raises(SourceEventContractError):
        extract_miiv_respiratory_events(source, allowed_stay_ids=ids)


@pytest.mark.parametrize("table,column", [("chartevents", "stay_id"), ("labevents", "hadm_id"), ("chartevents", "valuenum")])
def test_missing_exact_key_or_value_contract_is_not_silently_returned(tmp_path, table, column):
    source = write_source(tmp_path)
    path = tmp_path / table / "part.parquet"
    pd.read_parquet(path).drop(columns=column).to_parquet(path, index=False)
    with pytest.raises(SourceEventContractError, match="missing_required_columns"):
        extract_miiv_respiratory_events(source, allowed_stay_ids=[20])


def test_naive_clock_contract_rejects_timezone_aware_timestamp(tmp_path):
    source = write_source(tmp_path)
    path = tmp_path / "chartevents/part.parquet"
    frame = pd.read_parquet(path)
    frame["charttime"] = frame.charttime.dt.tz_localize("UTC")
    frame.to_parquet(path, index=False)
    with pytest.raises(SourceEventContractError, match="timezone_aware_source_clock"):
        extract_miiv_respiratory_events(source, allowed_stay_ids=[20])


def test_empty_allowed_measurements_keep_typed_schema(tmp_path):
    source = write_source(tmp_path, chart=[(101, 501, 10, 1, 220277, 95., "95", "%")])
    result = extract_miiv_respiratory_events(source, allowed_stay_ids=[20], concepts=["spo2"])
    assert result.hourly.empty and result.trace.empty
    assert str(result.hourly.stay_id.dtype) == "int64"
    assert str(result.hourly.charttime.dtype) == "float64"


def test_parse_failure_missing_infinite_and_bounds_are_distinct(tmp_path):
    inputs = [(None, None), (None, "bad"), (-np.inf, "40%"), (None, "50%"), (0., "40%")]
    chart = [(101, 501, 20, 21, 223835, val, text, "%") for val, text in inputs]
    result = extract_miiv_respiratory_events(write_source(tmp_path, chart=chart),
                                            allowed_stay_ids=[20], concepts=["fio2_chart"])
    assert result.trace.conversion_status.tolist() == ["missing_input", "unparseable", "nonfinite", "finite", "finite"]
    assert result.trace.exclusion_reason.tolist() == ["missing_input", "unparseable", "nonfinite", "", "outside_dictionary_bounds"]
    assert result.hourly.fio2_chart.tolist() == [50.]


@pytest.mark.parametrize("suffix", ["+05:00", "Z", " UTC"])
def test_string_timezone_is_not_silently_stripped(tmp_path, suffix):
    source = write_source(tmp_path)
    path = tmp_path / "icustays.parquet"
    clock = pd.read_parquet(path)
    clock.loc[clock.stay_id.eq(20), "outtime"] += suffix
    clock.to_parquet(path, index=False)
    with pytest.raises(SourceEventContractError, match="invalid_complete_hospital_clock_context"):
        extract_miiv_respiratory_events(source, allowed_stay_ids=[20])


def test_noninteger_source_key_schema_cannot_round_into_allowed_stay(tmp_path):
    source = write_source(tmp_path)
    path = tmp_path / "chartevents/part.parquet"
    chart = pd.read_parquet(path)
    chart["stay_id"] = 20.1
    chart.to_parquet(path, index=False)
    with pytest.raises(SourceEventContractError, match="noninteger_source_identity_schema:stay_id"):
        extract_miiv_respiratory_events(source, allowed_stay_ids=[20])


def test_every_clinical_fetch_has_sql_scope_predicate(tmp_path, monkeypatch):
    source = write_source(tmp_path)
    import easyicu.io.source_events as module
    original = module.duckdb.connect
    statements, fetches = [], []

    class Connection:
        def __init__(self, *args, **kwargs):
            self.connection = original(*args, **kwargs)

        def register(self, *args):
            return self.connection.register(*args)

        def execute(self, sql, *args):
            self.sql = sql
            statements.append(sql)
            self.connection.execute(sql, *args)
            return self

        def fetchone(self):
            return self.connection.fetchone()

        def fetchall(self):
            return self.connection.fetchall()

        def fetchdf(self):
            fetches.append(self.sql)
            assert "WHERE identity_status='identity_allowed'" in self.sql
            frame = self.connection.fetchdf()
            assert set(frame.stay_id) <= {20}
            return frame

        def close(self):
            self.connection.close()

    monkeypatch.setattr(module.duckdb, "connect", Connection)
    extract_miiv_respiratory_events(source, allowed_stay_ids=[20])
    assert len(fetches) == 2
    domains = [s for s in statements if "CREATE OR REPLACE VIEW source_domain" in s]
    assert len(domains) == 2
    assert any("o.stay_id IN (SELECT stay_id FROM requested)" in s for s in domains)
    assert any("o.hadm_id IN (SELECT hadm_id FROM targets)" in s for s in domains)
    assert all("o.itemid IN" in s for s in domains)


def test_source_change_during_extraction_is_detected(tmp_path, monkeypatch):
    source = write_source(tmp_path)
    import easyicu.io.source_events as module
    original = module._read_scoped_events

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = tmp_path / "chartevents/part.parquet"
        frame = pd.read_parquet(path)
        frame["value"] = "modified_after_engine_read"
        frame.to_parquet(path, index=False)
        return result

    monkeypatch.setattr(module, "_read_scoped_events", mutate)
    with pytest.raises(SourceEventContractError, match="source_changed_during_extraction"):
        extract_miiv_respiratory_events(source, allowed_stay_ids=[20])
