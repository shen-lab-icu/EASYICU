"""Regression tests for the second-round IO/converter review."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.config import DataSourceConfig
from easyicu.datasource import FilterOp, ICUDataSource
from easyicu.io.data_converter import DataConverter
from easyicu.io.data_load import load_src
from easyicu.io.parquet_reader import read_parquet_parallel
from easyicu.io.ts_utils import fill_gaps, round_to_interval


pytestmark = pytest.mark.unit


def test_load_src_accepts_mapping_filters_before_generic_iterables(monkeypatch) -> None:
    source = ICUDataSource(DataSourceConfig(name="unit"))
    captured: dict[str, object] = {}

    def fake_load_table(name, *, columns=None, filters=None, **_kwargs):
        captured.update(name=name, columns=columns, filters=filters)
        return SimpleNamespace(data=pd.DataFrame({"stay_id": [1, 2]}))

    monkeypatch.setattr(source, "load_table", fake_load_table)

    result = load_src(
        "events",
        src=source,
        rows={"stay_id": [1, 2], "site": "icu"},
    )

    filters = captured["filters"]
    assert result["stay_id"].tolist() == [1, 2]
    assert [(item.column, item.op, item.value) for item in filters] == [
        ("stay_id", FilterOp.IN, [1, 2]),
        ("site", FilterOp.EQ, "icu"),
    ]


def test_parallel_parquet_failure_with_string_path_has_stable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"

    with pytest.raises(RuntimeError, match="missing.parquet"):
        read_parquet_parallel([str(missing)])


def test_fill_gaps_fast_path_preserves_off_grid_observations() -> None:
    frame = pd.DataFrame(
        {"stay_id": [1, 1], "time": [0.5, 1.0], "value": [5.0, 10.0]}
    )
    limits = pd.DataFrame({"stay_id": [1], "start": [0.0], "end": [2.0]})

    result = fill_gaps(
        frame,
        ["stay_id"],
        "time",
        pd.Timedelta(hours=1),
        limits=limits,
        method="none",
    ).sort_values("time")

    assert result["time"].tolist() == [0.0, 0.5, 1.0, 2.0]
    assert result.loc[result["time"] == 0.5, "value"].item() == 5.0


@pytest.mark.parametrize("times", [1.5, pd.Index([0.0, 1.5])])
def test_round_to_interval_rejects_all_bare_numeric_axes(times) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        round_to_interval(times, pd.Timedelta(hours=1))


def test_hirid_archive_completeness_checks_exact_member_size(tmp_path: Path) -> None:
    data_dir = tmp_path / "hirid"
    data_dir.mkdir()
    archive = data_dir / "reference_data.tar.gz"
    payload = b"complete payload"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("reference/general_table.csv")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    target = data_dir / "reference" / "general_table.csv"
    target.parent.mkdir()
    target.write_bytes(b"x")
    converter = DataConverter(data_dir, database="hirid", verbose=False)

    assert not converter._archive_extraction_complete(archive)

    target.write_bytes(payload)
    assert converter._archive_extraction_complete(archive)


def test_hirid_archive_completeness_accepts_real_zero_byte_member(tmp_path: Path) -> None:
    data_dir = tmp_path / "hirid"
    data_dir.mkdir()
    archive = data_dir / "reference_data.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("reference/empty.csv")
        member.size = 0
        handle.addfile(member, io.BytesIO())

    target = data_dir / "reference" / "empty.csv"
    target.parent.mkdir()
    target.touch()
    converter = DataConverter(data_dir, database="hirid", verbose=False)

    assert converter._archive_extraction_complete(archive)


def test_aumc_extra_stale_parquet_shard_does_not_hide_sources(tmp_path: Path) -> None:
    source_dir = tmp_path / "numericitems_split"
    source_dir.mkdir()
    sources = [source_dir / "num_00.csv", source_dir / "num_01.csv"]
    for index, source in enumerate(sources):
        source.write_text(f"admissionid,value\n{index + 1},{index}.0\n", encoding="utf-8")

    target_dir = tmp_path / "numericitems"
    target_dir.mkdir()
    for shard in range(1, 4):
        pd.DataFrame({"admissionid": [shard], "value": [float(shard)]}).to_parquet(
            target_dir / f"{shard}.parquet", index=False
        )

    converter = DataConverter(tmp_path, database="aumc", verbose=False)

    assert set(converter._get_csv_files()) == set(sources)


def test_pandas_streaming_schema_scan_preserves_late_fraction(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed_numeric.csv"
    csv_path.write_text("id,value\n1,1\n2,2\n3,1.5\n", encoding="utf-8")
    shard_dir = tmp_path / "mixed_numeric"
    shard_dir.mkdir()
    converter = DataConverter(
        tmp_path,
        database="eicu",
        chunk_size=2,
        parallel_workers=1,
        verbose=False,
    )

    converter._convert_with_row_partitioning_pandas(
        csv_path,
        shard_dir,
        {"file": csv_path.name, "status": "pending", "row_count": 0, "shards": 0},
    )

    result = pd.read_parquet(shard_dir / "1.parquet")
    assert result["value"].tolist() == [1.0, 2.0, 1.5]


def test_noteevents_timestamp_columns_are_not_forced_to_string() -> None:
    pinned = set(DataConverter.MIXED_TYPE_COLUMNS["noteevents"])

    assert {"category", "description", "text", "iserror"} <= pinned
    assert {"charttime", "storetime", "chartdate"}.isdisjoint(pinned)


def test_noteevents_timestamp_columns_stay_temporal_across_converter_paths(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "NOTEEVENTS.csv"
    csv_path.write_text(
        "ROW_ID,CHARTDATE,CHARTTIME,STORETIME,CATEGORY,DESCRIPTION,TEXT,ISERROR\n"
        "1,2196-04-09,2196-04-09 12:34:56,,Nursing,Report,example,\n",
        encoding="utf-8",
    )
    converter = DataConverter(tmp_path, database="mimic", verbose=False)

    reader = converter._open_arrow_csv(csv_path, "noteevents")
    try:
        arrow_table = reader.read_all()
    finally:
        reader.close()
    for column in ("CHARTDATE", "CHARTTIME", "STORETIME"):
        assert "timestamp" in str(arrow_table.schema.field(column).type)

    pandas_frame = converter._fix_mixed_type_columns(
        pd.read_csv(csv_path), csv_path
    )
    for column in ("CHARTDATE", "CHARTTIME", "STORETIME"):
        assert pd.api.types.is_datetime64_any_dtype(pandas_frame[column])
