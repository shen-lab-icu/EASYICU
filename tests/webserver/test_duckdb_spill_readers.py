"""Direct fallback-reader checks without ASGI or a running server."""

from pathlib import Path

import pandas as pd
import pytest

from easyicu.io import parquet_reader
from easyicu.runtime import duckdb_spill
from easyicu.webserver import cohort_parquet, crossdb_parquet, dataio


@pytest.mark.parametrize("reader", ["parquet", "dataio"])
def test_fallback_readers_use_private_spill(monkeypatch, tmp_path, reader):
    path = tmp_path / "synthetic.parquet"
    pd.DataFrame({"stay_id": [1, 2], "value": [11, 22]}).to_parquet(path)
    paths = []
    original = duckdb_spill.connect

    def record(*args, **kwargs):
        connection = original(*args, **kwargs)
        paths.append(
            Path(
                connection.execute(
                    "SELECT current_setting('temp_directory')"
                ).fetchone()[0]
            )
        )
        return connection

    monkeypatch.setattr(duckdb_spill, "connect", record)
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("EASYICU_DUCKDB_TEMP_DIR", raising=False)
    if reader == "parquet":
        frame = parquet_reader._read_parquet_duckdb(
            path, ["stay_id", "value"], [("stay_id", "in", [2])]
        )
    else:
        frame = dataio._read_export_frame_duckdb(
            path, ["stay_id", "value"], stay_ids={"2"}, source="parquet"
        )
    assert frame.to_dict("list") == {"stay_id": [2], "value": [22]}
    assert len(paths) == 1
    assert paths[0].is_relative_to(tmp_path / "home" / ".easyicu")
    assert not paths[0].exists()


@pytest.mark.parametrize("owner", [cohort_parquet, crossdb_parquet])
def test_aggregate_owners_allow_private_spill(monkeypatch, tmp_path, owner):
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_DUCKDB_MAX_TEMP_SIZE", "32MiB")
    scope = (
        owner._connection() if owner is cohort_parquet else owner._connection(None, 10)
    )
    with scope as value:
        connection = value if owner is cohort_parquet else value[0]
        directory, maximum = connection.execute(
            "SELECT current_setting('temp_directory'), current_setting('max_temp_directory_size')"
        ).fetchone()
        path = Path(directory)
        assert path.parent == tmp_path / "duckdb-spill"
        assert path.is_dir()
        assert maximum == "32.0 MiB"
    assert not path.exists()
