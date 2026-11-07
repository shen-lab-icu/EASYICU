"""A read that started racily clean is never cached or memoized, however long it ran (delta30d).

B-D30-RACY-AGE: whether a source was racily clean (changed within 2 s of
its read) was decided again when the payload was cached, and for the CSV
validation memo only from the stat taken after the scan. A file racy at its
first read, rewritten within the same coarse timestamp tick, aged out of
the window during a long read. The stale payload or memo was then accepted.
The flag is now frozen at consume time (Parquet capture) and from the stat
taken before the scan (CSV), and only the frozen flag admits a cache or memo
entry. A coarse-timestamp filesystem is modelled by a frozen stat and a
controlled clock. Rows are synthetic.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from easyicu import datasource as datasource_module
from easyicu import storage_binding
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_binding import binding_unit
from easyicu.storage_layout import TableScanError

LONG_READ_NS = 3_000_000_000


def _coarse_filesystem(monkeypatch, path):
    """Every stat of ``path`` is the one taken now; the clock starts at its timestamp."""
    real = storage_binding._version
    frozen = real("t", str(path))
    clock = [max(frozen[1], frozen[2])]
    monkeypatch.setattr(
        storage_binding, "_version",
        lambda table, p: frozen if str(p) == str(path) else real(table, p),
    )
    monkeypatch.setattr(storage_binding, "time", types.SimpleNamespace(time_ns=lambda: clock[0]))
    return clock


def _parquet(path, value):
    pd.DataFrame({"value": np.repeat(value, 20)}).to_parquet(
        path, index=False, compression=None, use_dictionary=False
    )


def test_a_long_read_that_started_racy_is_not_cached(tmp_path, monkeypatch):
    root = tmp_path / "miiv"
    (root / "chartevents").mkdir(parents=True)
    path = root / "chartevents" / "part.parquet"
    _parquet(path, 60.0)
    clock = _coarse_filesystem(monkeypatch, path)
    source = ICUDataSource(load_src_cfg("miiv"), base_path=root, enable_cache=True)

    def cold(self, table, columns, pf, items, wide, enable, key):
        frame = pd.read_parquet(self.resolve_target_files(table)[0])
        _parquet(path, 61.0)  # rewritten within the same coarse tick
        clock[0] += LONG_READ_NS  # the read ran past the racy window
        if enable:
            self._cache_raw_frame(key, frame)
        return frame

    source._load_raw_frame_uncached = types.MethodType(cold, source)
    with binding_unit():
        source._load_raw_frame("chartevents", None)
    assert not source._table_cache


def _csv(root):
    header = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,STORETIME,CGID,VALUE,"
        "VALUENUM,VALUEUOM,WARNING,ERROR,RESULTSTATUS,STOPPED"
    )
    rows = [
        f"{i},1,11,101,220045,2140-01-02 01:00:00,2140-01-02 01:05:00,17,60,60,bpm,0,0,,"
        for i in range(1, 6)
    ]
    path = root / "CHARTEVENTS.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_a_validation_scan_that_started_racy_is_not_memoized(tmp_path, monkeypatch):
    monkeypatch.setattr(ICUDataSource, "_CSV_VALIDATED", set(), raising=False)
    path = _csv(tmp_path)
    clock = _coarse_filesystem(monkeypatch, path)
    real_connection = datasource_module._get_duckdb_connection()
    mutated = []

    class SlowScan:
        """The full-schema scan returns after a same-tick rewrite and a long run."""

        def execute(self, query, *args, **kwargs):
            result = real_connection.execute(query, *args, **kwargs)
            if "SELECT count(*)," in query and not mutated:
                text = path.read_text()
                path.write_text(text.replace(",60,bpm,", ",xx,bpm,", 1))
                clock[0] += LONG_READ_NS
                mutated.append(True)
            return result

    monkeypatch.setattr(datasource_module, "_get_duckdb_connection", lambda: SlowScan())
    source = ICUDataSource(load_src_cfg("mimic"), base_path=tmp_path, enable_cache=False)
    with binding_unit():
        source._read_mimic3_csv_fallback(path, ["itemid", "value"], ("itemid", {220045}))
    assert mutated and not ICUDataSource._CSV_VALIDATED
    with binding_unit(), pytest.raises(TableScanError) as error:
        source._read_mimic3_csv_fallback(path, ["itemid", "value"], ("itemid", {220045}))
    assert error.value.reason == "csv_rows_rejected"
