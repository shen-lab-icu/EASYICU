"""Rows and stay tables are never dropped or substituted silently (delta17c, commit 4).

(a) The MIMIC-III chartevents CSV fallback read with
``read_csv_auto(ignore_errors=true)``, so a row the parser rejected vanished.
It now parses strictly with DuckDB's ``store_rejects``. Any reject refuses
the read with ``csv_rows_rejected``, giving the count and reason classes.
(b) Stay-universe discovery read a present but unreadable Parquet stay table
as "try the CSV sibling". Corruption is not absence, so it now refuses with
``table_scan_file_unreadable``. The CSV is used only when the Parquet stay
table is absent, and the receipt says ``source_format: csv_fallback``.
Rows are synthetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_layout import TableScanError

ROWS_REJECTED = "csv_rows_rejected"  # the stable reason codes
FILE_UNREADABLE = "table_scan_file_unreadable"
HEADER = (
    "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,STORETIME,CGID,VALUE,"
    "VALUENUM,VALUEUOM,WARNING,ERROR,RESULTSTATUS,STOPPED"
)


def _row(i, *, value=None, item="220045"):
    value = str(60 + i % 5) if value is None else value
    return (
        f"{i},1,11,101,{item},2140-01-02 01:00:00,2140-01-02 01:05:00,17,"
        f"{value},{60 + i % 5},bpm,0,0,,"
    )


def _chartevents_csv(root, *, rows: int = 5, insert: str | None = None, late: str | None = None):
    """``insert`` lands mid-file; ``late`` after DuckDB's 20,480-row sniffing sample."""
    lines = [HEADER] + [_row(i) for i in range(1, rows + 1)]
    if insert is not None:
        lines.insert(3, insert)
    if late is not None:
        lines.append(late)
    path = root / "CHARTEVENTS.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def _mimic(root):
    return ICUDataSource(load_src_cfg("mimic"), base_path=root, enable_cache=False)


def _read(root, path):
    return _mimic(root)._read_mimic3_csv_fallback(path, ["itemid", "value", "valuenum"], ("itemid", {220045}))


def test_a_clean_chartevents_csv_reads_every_row(tmp_path):
    frame = _read(tmp_path, _chartevents_csv(tmp_path))
    assert sorted(frame["valuenum"].tolist()) == [60, 61, 62, 63, 64]


@pytest.mark.parametrize("value,expected", [
    ('"4, Spontaneously"', "4, Spontaneously"),  # a quoted comma
    ('"line one\nline two"', "line one\nline two"),  # a quoted newline
])
def test_legitimate_quoting_still_reads(tmp_path, value, expected):
    frame = _read(tmp_path, _chartevents_csv(tmp_path, insert=_row(9, value=value)))
    assert len(frame) == 6 and expected in frame["value"].tolist()


@pytest.mark.parametrize("bad,where,klass", [
    (_row(9) + ",EXTRA", "insert", "TOO MANY COLUMNS"),
    ("9,1,11,101,220045,2140-01-02 01:00:00", "insert", "MISSING COLUMNS"),
    (_row(9, value='"70'), "insert", "UNQUOTED VALUE"),
    (_row(25_001, item="NOT_AN_ITEM"), "late", "CAST"),  # ignore_errors dropped it
])
def test_a_row_the_strict_parser_rejects_refuses_the_read(tmp_path, bad, where, klass):
    rows = 25_000 if where == "late" else 5
    path = _chartevents_csv(tmp_path, rows=rows, **{where: bad})
    with pytest.raises(TableScanError) as error:
        _read(tmp_path, path)
    assert error.value.reason == ROWS_REJECTED
    assert f"rejected 1 rows ({klass}=1)" in str(error.value)


def test_a_header_that_is_not_the_configured_schema_is_refused(tmp_path):
    path = tmp_path / "CHARTEVENTS.csv"
    path.write_text("ICUSTAY_ID,ITEMID,VALUE\n101,220045,60\n")
    with pytest.raises(TableScanError) as error:
        _read(tmp_path, path)
    assert error.value.reason == "csv_header_mismatch"


def _stays(root, name="icustays.parquet"):
    frame = pd.DataFrame({"subject_id": [1, 2, 3], "hadm_id": [11, 12, 13], "icustay_id": [101, 102, 103]})
    frame.to_parquet(root / name, index=False)
    return frame


def _corrupt_pages(path):
    """Keep the footer (row counts still read) and break every column chunk's pages."""
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    data = bytearray(path.read_bytes())
    for group in range(metadata.num_row_groups):
        for column in range(metadata.num_columns):
            chunk = metadata.row_group(group).column(column)
            start = chunk.dictionary_page_offset or chunk.data_page_offset
            for offset in range(start, start + chunk.total_compressed_size):
                data[offset] = 0xFF
    path.write_bytes(bytes(data))
    assert pq.ParquetFile(path).metadata.num_rows == metadata.num_rows


def test_a_present_but_unreadable_parquet_stay_table_is_refused_not_replaced(tmp_path):
    from easyicu.api import get_all_patient_ids

    _stays(tmp_path)
    _corrupt_pages(tmp_path / "icustays.parquet")
    pd.DataFrame({"icustay_id": [900]}).to_csv(tmp_path / "icustays.csv", index=False)
    with pytest.raises(TableScanError) as error:
        get_all_patient_ids(tmp_path, database="mimic")
    assert error.value.reason == FILE_UNREADABLE


def test_an_absent_parquet_stay_table_uses_the_csv_and_says_so(tmp_path):
    from easyicu.api import get_all_patient_ids

    pd.DataFrame({"icustay_id": [101, 102]}).to_csv(tmp_path / "icustays.csv", index=False)
    listing: dict = {}
    ids, column = get_all_patient_ids(tmp_path, database="mimic", listing=listing)
    assert (sorted(ids), column) == ([101, 102], "icustay_id")
    assert listing["source_format"] == "csv_fallback"


def test_a_parquet_stay_table_is_named_as_parquet(tmp_path):
    from easyicu.api import get_all_patient_ids

    _stays(tmp_path)
    listing: dict = {}
    ids, _column = get_all_patient_ids(tmp_path, database="mimic", listing=listing)
    assert sorted(ids) == [101, 102, 103] and listing["source_format"] == "parquet"
