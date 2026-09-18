"""Error contract for ``easyicu.io.parquet_reader.read_parquet_parallel``."""

from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from easyicu.io.parquet_reader import read_parquet_parallel


def test_parallel_parquet_failure_with_string_path_has_stable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"

    with pytest.raises(RuntimeError, match="missing.parquet"):
        read_parquet_parallel([str(missing)])


def test_parallel_reader_retains_declared_partition_order(monkeypatch):
    from easyicu.io import parquet_reader
    monkeypatch.setattr(parquet_reader, "read_parquet", lambda path, **kw: pd.DataFrame({"file": [path]}))
    # Force reverse completion order without timing assumptions.
    monkeypatch.setattr(parquet_reader, "as_completed", lambda futures: reversed(list(futures)))
    assert read_parquet_parallel(["a", "b", "c"]).file.tolist() == ["a", "b", "c"]
