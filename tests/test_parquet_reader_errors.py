"""Error contract for ``easyicu.io.parquet_reader.read_parquet_parallel``."""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.io.parquet_reader import read_parquet_parallel


def test_parallel_parquet_failure_with_string_path_has_stable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"

    with pytest.raises(RuntimeError, match="missing.parquet"):
        read_parquet_parallel([str(missing)])
