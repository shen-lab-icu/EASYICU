"""A MIMIC-III CSV value is cast and checked whatever the query selects (delta25d).

B-D25-CSV-PROJECTION: DuckDB applies a declared column type only to the
columns a query reads. A chartevents CSV with a bad VALUENUM token was
therefore accepted when the fallback read only itemid/value, and refused
when it read every column. Every read now requires a full-schema scan of the
file that casts all declared columns. The scan runs once per file version
(realpath + stat identity) per process. Rows are synthetic.
"""

from __future__ import annotations

import os

import pytest

from easyicu import storage_binding
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_layout import TableScanError

HEADER = (
    "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,STORETIME,CGID,VALUE,"
    "VALUENUM,VALUEUOM,WARNING,ERROR,RESULTSTATUS,STOPPED"
)


def _row(i, *, valuenum=None, charttime="2140-01-02 01:00:00", cgid="17"):
    valuenum = str(60 + i % 5) if valuenum is None else valuenum
    return f"{i},1,11,101,220045,{charttime},2140-01-02 01:05:00,{cgid},{60 + i % 5},{valuenum},bpm,0,0,,"


def _csv(root, bad: str | None = None):
    lines = [HEADER] + [_row(i) for i in range(1, 6)]
    if bad is not None:
        lines.insert(3, bad)
    path = root / "CHARTEVENTS.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def _read(root, path, columns=("itemid", "value")):
    source = ICUDataSource(load_src_cfg("mimic"), base_path=root, enable_cache=False)
    return source._read_mimic3_csv_fallback(path, list(columns), ("itemid", {220045}))


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    monkeypatch.setattr(ICUDataSource, "_CSV_VALIDATED", set(), raising=False)


@pytest.mark.parametrize("bad", [
    _row(9, valuenum="invalid"),  # a numeric measurement
    _row(9, charttime="not a time"),  # a time field
    _row(9, cgid="caregiver?"),  # an identifier field
])
def test_a_bad_token_in_an_unselected_column_refuses_the_read(tmp_path, bad):
    with pytest.raises(TableScanError) as error:
        _read(tmp_path, _csv(tmp_path, bad))
    assert error.value.reason == "csv_rows_rejected"
    assert "full-schema validation scan" in str(error.value) and "(CAST)" in str(error.value)


def test_a_clean_file_is_validated_once_per_version(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_binding, "RACY_WINDOW_NS", 0)  # a settled file
    scans = []
    real = ICUDataSource._validate_mimic3_csv.__func__

    def counting(cls, con, csv_path, table, header, options):
        before = len(cls._CSV_VALIDATED)
        real(cls, con, csv_path, table, header, options)
        scans.append(len(cls._CSV_VALIDATED) > before)

    monkeypatch.setattr(ICUDataSource, "_validate_mimic3_csv", classmethod(counting))
    path = _csv(tmp_path)
    assert len(_read(tmp_path, path)) == 5
    assert len(_read(tmp_path, path)) == 5
    assert scans == [True, False]  # the second read hit the memo
    # The same file rewritten in place (same size, mtime restored) is a new
    # version: it is scanned again, and its bad token now refuses the read.
    stat = path.stat()
    path.write_text(path.read_text().replace(",62,bpm,", ",6x,bpm,", 1))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    with pytest.raises(TableScanError) as error:
        _read(tmp_path, path)
    assert error.value.reason == "csv_rows_rejected"


def test_a_racily_clean_file_is_not_memoized(tmp_path):
    path = _csv(tmp_path)
    _read(tmp_path, path)
    assert not ICUDataSource._CSV_VALIDATED
