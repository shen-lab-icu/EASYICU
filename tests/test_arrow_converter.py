"""Unit tests for the end-to-end Arrow conversion path in data_converter.

All tests build tiny synthetic CSVs in ``tmp_path`` — they never touch
the real ICU databases on the macfuse mount, so they're safe to run in
parallel with on-mount work.

What's covered
--------------
- ID-partition assignment math matches ``np.searchsorted(side='right') + 1``.
- ``ConvertOptions.column_types`` keeps a numeric-looking text column as
  string (the bug pattern that crashed medication / corrupted
  labresulttext in the legacy pandas converter).
- ``_threaded_batch_iter`` does not deadlock when the consumer raises
  while the producer queue is full.
- ``write_conversion_manifest`` skips SHA-256 unless ``evidence_root``
  is provided (records size + mtime instead).
- The Arrow path emits zstd parquet by default and honours the
  ``EASYICU_PARQUET_COMPRESSION`` env override.
- The dispatcher falls back to the pandas path when the Arrow path
  raises.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.unit


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Tiny CSV writer that uses only stdlib + the values given."""
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join("" if v is None else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_converter(tmp_path: Path, **overrides):
    """Build a DataConverter rooted at *tmp_path* with no env coupling."""
    from easyicu.io.data_converter import DataConverter

    # Make sure no parent env steers compression unless the test does.
    os.environ.pop("EASYICU_PARQUET_COMPRESSION", None)
    return DataConverter(
        tmp_path,
        database=overrides.get("database", "eicu"),
        parallel_workers=1,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# ID-partition assignment: Arrow's vectorised math must agree with the
# pandas-era np.searchsorted(side='right') + 1 expression.
# ---------------------------------------------------------------------------


def test_id_partition_math_matches_numpy_searchsorted(tmp_path):
    """Synthetic table → Arrow id-partitioning → row distribution matches
    the searchsorted reference exactly across all partitions."""
    breaks = [10, 25, 50, 100]
    n_partitions = len(breaks) + 1
    rng = np.random.default_rng(0)
    ids = rng.integers(low=1, high=200, size=5000)

    csv = tmp_path / "tiny.csv"
    _write_csv(
        csv,
        ["pid", "v"],
        [[int(i), float(rng.random())] for i in ids],
    )

    shard_dir = tmp_path / "tiny"
    shard_dir.mkdir()

    c = _make_converter(tmp_path)
    c._convert_with_id_partitioning_arrow(
        csv,
        shard_dir,
        {"col": "pid", "breaks": breaks},
        {"file": csv.name, "status": "PENDING", "row_count": 0, "shards": 0},
    )

    # Reference: per-partition expected counts.
    expected = np.bincount(
        np.searchsorted(breaks, ids, side="right") + 1,
        minlength=n_partitions + 2,
    )[1:n_partitions + 1]

    actual = []
    for p in range(1, n_partitions + 1):
        fp = shard_dir / f"{p}.parquet"
        actual.append(pq.read_metadata(fp).num_rows if fp.exists() else 0)

    assert list(actual) == list(expected), (
        f"partition row counts diverged: arrow={actual} expected={list(expected)}"
    )

    # Cross-check total rows preserved.
    assert sum(actual) == len(ids)


# ---------------------------------------------------------------------------
# Mixed-type column pinning: the bug that crashed medication / corrupted
# labresulttext in the pandas path. With ``column_types`` set, values like
# '0 mg' are preserved verbatim as strings.
# ---------------------------------------------------------------------------


def test_mixed_type_column_preserved_as_string(tmp_path, monkeypatch):
    """A column that looks numeric in the first chunk and has a string
    later must come out of the Arrow path as a string column with the
    string value preserved (no float round-trip corruption)."""
    csv = tmp_path / "medication.csv"
    rows = [[i, "5" if i < 100 else "0 mg"] for i in range(1, 121)]
    _write_csv(csv, ["pid", "dosage"], rows)

    c = _make_converter(tmp_path)
    # Pretend this table has 'dosage' in MIXED_TYPE_COLUMNS — that's what
    # production already does (DataConverter.MIXED_TYPE_COLUMNS).
    assert "dosage" in c.MIXED_TYPE_COLUMNS.get("medication", []), (
        "production MIXED_TYPE_COLUMNS must keep dosage pinned to string"
    )

    shard_dir = tmp_path / "medication"
    shard_dir.mkdir()
    c._convert_with_row_partitioning_arrow(
        csv,
        shard_dir,
        {"file": csv.name, "status": "PENDING", "row_count": 0, "shards": 0},
    )

    files = sorted(shard_dir.glob("*.parquet"))
    assert files, "no shard files written"
    tbl = pa.concat_tables([pq.read_table(p) for p in files])
    assert pa.types.is_string(tbl.column("dosage").type), (
        f"dosage should be string, got {tbl.column('dosage').type}"
    )
    values = tbl.column("dosage").to_pylist()
    assert values[0] == "5"
    assert values[-1] == "0 mg", "string value lost or corrupted"
    assert tbl.num_rows == 120


def test_aumc_latin1_arrow_path_keeps_numericitems_value_numeric(tmp_path):
    """Latin-1 is not a reason to send AUMC's 80GB table through pandas."""

    csv = tmp_path / "numericitems.csv"
    csv.write_bytes(
        (
            "admissionid,itemid,item,value,unit,registeredby\n"
            "1,6640,Température,38.5,°C,Système\n"
        ).encode("latin1")
    )
    converter = _make_converter(tmp_path, database="aumc")
    shard_dir = tmp_path / "numericitems"
    shard_dir.mkdir()

    assert converter._arrow_csv_enabled(csv)
    result = converter._convert_with_row_partitioning_arrow(
        csv,
        shard_dir,
        {"file": csv.name, "status": "PENDING", "row_count": 0, "shards": 0},
    )

    assert result["status"] == "completed"
    table = pq.read_table(shard_dir / "1.parquet")
    assert table.column("item").to_pylist() == ["Température"]
    assert table.column("value").to_pylist() == [38.5]
    assert pa.types.is_floating(table.column("value").type)


# ---------------------------------------------------------------------------
# Threaded batch iterator: must not deadlock when the consumer raises
# while the producer queue is full.
# ---------------------------------------------------------------------------


def test_threaded_batch_iter_no_deadlock_on_early_exit(tmp_path):
    """Simulate a many-batch reader and a consumer that bails after the
    first batch. The generator's finally block must release the
    background thread (no hang) within a reasonable timeout."""
    import threading

    class FakeReader:
        def __init__(self, n):
            self.left = n

        def read_next_batch(self):
            if self.left <= 0:
                raise StopIteration
            self.left -= 1
            # tiny batch so we don't allocate much
            return pa.record_batch([pa.array([1, 2, 3])], names=["x"])

    c = _make_converter(tmp_path)
    reader = FakeReader(n=200)

    done = threading.Event()

    def consume():
        try:
            for i, _ in enumerate(c._threaded_batch_iter(reader, queue_size=2)):
                if i == 0:
                    raise RuntimeError("consumer bails on purpose")
        except RuntimeError:
            pass
        finally:
            done.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    # Generous timeout; well under the helper's 10s join timeout but
    # well above any reasonable shutdown latency.
    assert done.wait(timeout=5.0), (
        "_threaded_batch_iter deadlocked on consumer early-exit"
    )
    t.join(timeout=2.0)
    assert not t.is_alive(), "consumer thread did not exit"


# ---------------------------------------------------------------------------
# Manifest SHA-256 lazy by default. The legacy converter hashed every
# input + output parquet to populate the manifest; on a slow mount that
# was a full extra read pass. SHA-256 now runs only when evidence_root
# is provided.
# ---------------------------------------------------------------------------


def test_manifest_record_no_hash_by_default(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"not actually parquet")
    c = _make_converter(tmp_path)
    record = c._file_manifest_record(f, hash_files=False)
    assert record["size_bytes"] == f.stat().st_size
    assert "sha256" not in record
    assert "mtime_ns" in record


def test_manifest_record_with_hash(tmp_path):
    f = tmp_path / "x.parquet"
    payload = b"hello world"
    f.write_bytes(payload)
    c = _make_converter(tmp_path)
    record = c._file_manifest_record(f, hash_files=True)
    assert "sha256" in record
    # 64 hex chars
    assert len(record["sha256"]) == 64


# ---------------------------------------------------------------------------
# Compression default and env override.
# ---------------------------------------------------------------------------


def test_parquet_compression_defaults_to_zstd(tmp_path, monkeypatch):
    monkeypatch.delenv("EASYICU_PARQUET_COMPRESSION", raising=False)
    from easyicu.io.data_converter import DataConverter

    c = DataConverter(tmp_path, database="eicu", parallel_workers=1, verbose=False)
    assert c.parquet_compression == "zstd"


def test_parquet_compression_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_PARQUET_COMPRESSION", "snappy")
    from easyicu.io.data_converter import DataConverter

    c = DataConverter(tmp_path, database="eicu", parallel_workers=1, verbose=False)
    assert c.parquet_compression == "snappy"


def test_zstd_output_actually_zstd(tmp_path):
    """End-to-end: convert a tiny CSV via the row-partitioning Arrow path
    and verify pyarrow reports zstd as the on-disk compression."""
    csv = tmp_path / "tiny.csv"
    _write_csv(
        csv,
        ["pid", "v"],
        [[i, float(i)] for i in range(1, 51)],
    )
    shard_dir = tmp_path / "tiny"
    shard_dir.mkdir()

    c = _make_converter(tmp_path)
    c._convert_with_row_partitioning_arrow(
        csv, shard_dir,
        {"file": csv.name, "status": "PENDING", "row_count": 0, "shards": 0},
    )
    p = next(shard_dir.glob("*.parquet"))
    meta = pq.ParquetFile(p).metadata
    # any column will do; all should share the converter's compression.
    codec = meta.row_group(0).column(0).compression
    assert codec.lower() == "zstd", f"expected zstd, got {codec}"


# ---------------------------------------------------------------------------
# Dispatcher fallback: if the Arrow path raises, the pandas path takes
# over and still produces output. We force a failure by passing a
# partition column that does not exist in the CSV.
# ---------------------------------------------------------------------------


def test_id_partitioning_falls_back_to_pandas_on_arrow_failure(tmp_path):
    csv = tmp_path / "fb.csv"
    # 'pid' is the partition col below; deliberately omit it from the
    # CSV so the Arrow path raises ValueError("partition column ... not
    # in CSV header") and the dispatcher falls back to pandas.
    _write_csv(
        csv,
        ["alt_id", "v"],
        [[i, i * 2] for i in range(1, 41)],
    )
    shard_dir = tmp_path / "fb"
    shard_dir.mkdir()
    c = _make_converter(tmp_path)
    res = c._convert_with_id_partitioning(
        csv,
        shard_dir,
        {"col": "pid", "breaks": [10, 20, 30]},
        {"file": csv.name, "status": "PENDING", "row_count": 0, "shards": 0},
    )
    # Pandas fallback also has no 'pid' column, so it logs a warning
    # and re-routes to row-partitioning. The point of this test is
    # that the dispatcher catches the Arrow-path exception without
    # crashing the whole conversion.
    assert res is not None
    assert "status" in res
