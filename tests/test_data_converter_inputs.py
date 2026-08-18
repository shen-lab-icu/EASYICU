"""DataConverter source discovery and column typing.

Owner: which raw files the converter reads and how it types their
columns. Added by the 2026-08-16 second-round IO review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.io.data_converter import DataConverter


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
