"""Typed aggregation failures, exact batch sums and a total same-time order.

Lead-r13 round 2: a failed hourly aggregation or a failed DuckDB wide query
used to fall back to a different definition (keep-first rows, raw loading);
the wide and long batch loaders summed in thread order; HiRID same-time
duplicates kept whichever row the input order put last.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.errors import ConceptTableReadError
from easyicu.io.ts_utils import IntervalAggregationError, change_interval
from tests.support.eicu_source import _eicu_source, _hourly
from tests.support.hirid_source import hirid_source_root

pytestmark = pytest.mark.filterwarnings("ignore")

# Left-to-right double addition loses both 1.0 terms; the exact total is 2.0.
CANCELLING = [1e16, 1.0, 1.0, -1e16]


def test_failed_hourly_aggregation_is_a_typed_failure(monkeypatch):
    import easyicu.io.ts_utils as ts_utils
    import easyicu.utils.exact_sum as exact_sum

    def _fail(*args, **kwargs):
        raise MemoryError("injected aggregation failure")

    monkeypatch.setattr(ts_utils, "_fast_groupby_agg", _fail)
    monkeypatch.setattr(exact_sum, "exact_groupby_agg", _fail)
    frame = pd.DataFrame({"stay": [1, 1], "time": [0.0, 0.5], "value": [1.0, 2.0]})
    # Both the frame and the ICUTable entry points refuse keep-first rows.
    with pytest.raises(IntervalAggregationError) as caught:
        change_interval(frame, pd.Timedelta(hours=1), time_col="time", id_cols=["stay"],
                        aggregation="sum", time_unit="h")
    assert caught.value.reason == "interval_aggregation_failed"
    assert isinstance(caught.value.__cause__, MemoryError)
    from easyicu.table import ICUTable

    table = ICUTable(frame, id_columns=["stay"], index_column="time", value_column="value")
    with pytest.raises(IntervalAggregationError):
        change_interval(table, pd.Timedelta(hours=1), aggregation="sum", time_unit="h")


def test_frame_hourly_sum_is_exact():
    frame = pd.DataFrame({"stay": 1, "time": [0.1, 0.2, 0.3, 0.4], "value": CANCELLING})
    out = change_interval(frame, pd.Timedelta(hours=1), time_col="time", id_cols=["stay"],
                          aggregation="sum", time_unit="h")
    assert out["value"].tolist() == [2.0]


def _vitals(tmp_path, values):
    offsets = [5.0 + i for i in range(len(values))]
    return _eicu_source(tmp_path, "eicu", vitalperiodic=pd.DataFrame({
        "vitalperiodicid": range(len(values)), "observationoffset": offsets,
        "heartrate": values,
    }))


def test_failed_wide_duckdb_query_is_not_reread_raw(tmp_path, monkeypatch):
    source = _vitals(tmp_path, [80.0, 82.0])
    import easyicu.datasource as datasource

    real = datasource._get_duckdb_connection

    class _FailingWideQuery:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            if "AS observationoffset" in sql:
                raise RuntimeError("injected wide query failure")
            return self._conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(datasource, "_get_duckdb_connection", lambda: _FailingWideQuery(real()))
    with pytest.raises(ConceptTableReadError) as caught:
        _hourly(source, "hr")
    assert caught.value.stage == "wide_table_duckdb_aggregation"


def test_wide_batch_sum_is_exact(tmp_path):
    from easyicu.datasource import load_wide_table_aggregated

    source = _vitals(tmp_path, CANCELLING)
    frame = load_wide_table_aggregated(
        source, "vitalperiodic", ["heartrate"], interval_hours=1.0,
        patient_ids=[1], agg_func="sum",
    )
    assert frame["heartrate"].tolist() == [2.0]


def test_long_batch_sum_is_exact(tmp_path):
    from easyicu.datasource import load_bucketed_table_multi_aggregated

    values = CANCELLING + [7.0]
    source = _eicu_source(tmp_path, "eicu", lab=pd.DataFrame({
        "labid": range(len(values)), "labresultoffset": [5.0] * len(values),
        "labtypeid": 1, "labname": ["creatinine"] * 4 + ["lactate"],
        "labresult": values, "labresulttext": [str(v) for v in values],
        "labmeasurenamesystem": "mg/dL", "labmeasurenameinterface": "mg/dL",
        "labresultrevisedoffset": [5.0] * len(values),
    }))
    frame = load_bucketed_table_multi_aggregated(
        source, "lab", {"crea": ["creatinine"], "lact": ["lactate"]},
        value_column="labresult", patient_ids=[1], agg_func="sum",
        itemid_col="labname", time_col="labresultoffset",
    )
    assert frame["crea"].tolist() == [2.0]
    assert frame["lact"].tolist() == [7.0]


def _neut(tmp_path, name, rows):
    from tests.support.hirid_source import hirid_load

    root = hirid_source_root(tmp_path / name, "observations", rows)
    frame = hirid_load(root, ["neut"], patient_ids={"patientid": [1]})["neut"]
    time = next(c for c in frame.columns if c not in {"neut", "patientid"})
    return dict(zip(frame[time].astype(float), frame["neut"]))


def test_same_time_duplicates_resolve_by_a_total_order(tmp_path):
    def row(variable, value, entered):
        return {"patientid": 1, "variableid": variable, "value": value, "datetime": 5.0,
                "entertime": pd.Timestamp("2100-01-01") + pd.Timedelta(minutes=entered)}

    # Two segmented results at one time; the latest entry is kept, then the
    # band forms are added.  Row order on disk must not matter.
    rows = [row(24000556, 60.0, 1), row(24000556, 64.0, 2), row(24000557, 20.0, 1)]
    forward = _neut(tmp_path, "forward", rows)
    backward = _neut(tmp_path, "backward", rows[::-1])
    assert forward == backward == {5.0: 84.0}
