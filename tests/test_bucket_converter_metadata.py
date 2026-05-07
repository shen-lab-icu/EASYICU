from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import easyicu.bucket_converter as bucket_converter
from easyicu.bucket_converter import BucketConfig, convert_parquet_directory_to_buckets, convert_to_buckets


def test_parquet_directory_bucket_conversion_writes_complete_marker(tmp_path: Path) -> None:
    source_dir = tmp_path / "events"
    source_dir.mkdir()
    pd.DataFrame({"stay_id": [1, 2], "itemid": [220045, 220181]}).to_parquet(
        source_dir / "part_1.parquet"
    )
    pd.DataFrame({"stay_id": [3], "itemid": [220045]}).to_parquet(
        source_dir / "part_2.parquet"
    )

    output_dir = tmp_path / "events_bucket"
    result = convert_parquet_directory_to_buckets(
        source_dir,
        output_dir,
        partition_col="itemid",
        num_buckets=8,
        progress_callback=lambda _: None,
    )

    assert result.success, result.error
    assert list(output_dir.glob("bucket_id=*/*.parquet"))

    marker = output_dir / "_COMPLETE"
    assert marker.exists()

    row_count, actual_buckets, total_size = marker.read_text(encoding="utf-8").split(",")
    assert int(row_count) == 3
    assert int(actual_buckets) == result.num_buckets
    assert int(total_size) == result.total_size_bytes


def test_single_file_bucket_conversion_writes_copy_row_count_to_complete_marker(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "itemid": [220045, 220181, 220045, 220210],
            "valuenum": [80.0, 120.0, 81.0, 15.0],
        }
    ).to_parquet(source)

    output_dir = tmp_path / "events_bucket"
    result = convert_to_buckets(
        source,
        output_dir,
        BucketConfig(num_buckets=8, partition_col="itemid", compression="snappy"),
        progress_callback=lambda _: None,
    )

    assert result.success, result.error

    marker = output_dir / "_COMPLETE"
    row_count, actual_buckets, total_size = marker.read_text(encoding="utf-8").split(",")
    assert int(row_count) == 4
    assert int(row_count) == result.total_rows
    assert int(actual_buckets) == result.num_buckets
    assert int(total_size) == result.total_size_bytes


def test_bucket_conversion_can_sort_rows_within_bucket_for_extraction_profile(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "itemid": [300, 100, 200],
            "valuenum": [3.0, 1.0, 2.0],
        }
    ).to_parquet(source)

    output_dir = tmp_path / "events_bucket"
    result = convert_to_buckets(
        source,
        output_dir,
        BucketConfig(
            num_buckets=1,
            partition_col="itemid",
            compression="snappy",
            skip_sorting=False,
        ),
        progress_callback=lambda _: None,
    )

    assert result.success, result.error
    bucket_files = list((output_dir / "bucket_id=0").glob("*.parquet"))
    assert bucket_files
    converted = pd.read_parquet(bucket_files[0])
    assert converted["itemid"].tolist() == [100, 200, 300]


def test_completed_single_file_bucket_conversion_is_reused(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    pd.DataFrame({"stay_id": [1, 2], "itemid": [220045, 220181]}).to_parquet(source)

    output_dir = tmp_path / "events_bucket"
    first = convert_to_buckets(
        source,
        output_dir,
        BucketConfig(num_buckets=8, partition_col="itemid", compression="snappy"),
        progress_callback=lambda _: None,
    )
    assert first.success, first.error

    second = convert_to_buckets(
        source,
        output_dir,
        BucketConfig(num_buckets=8, partition_col="itemid", compression="snappy"),
        progress_callback=lambda _: None,
    )

    assert second.success, second.error
    assert second.total_rows == first.total_rows
    assert second.num_buckets == first.num_buckets
    assert second.total_size_bytes == first.total_size_bytes


def test_completed_single_file_bucket_conversion_is_not_reused_when_source_is_newer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "events.parquet"
    pd.DataFrame({"stay_id": [1, 2], "itemid": [220045, 220181]}).to_parquet(source)

    output_dir = tmp_path / "events_bucket"
    first = convert_to_buckets(
        source,
        output_dir,
        BucketConfig(num_buckets=8, partition_col="itemid", compression="snappy"),
        progress_callback=lambda _: None,
    )
    assert first.success, first.error

    marker_mtime = (output_dir / "_COMPLETE").stat().st_mtime
    os.utime(source, (marker_mtime + 10, marker_mtime + 10))

    second = convert_to_buckets(
        source,
        output_dir,
        BucketConfig(num_buckets=8, partition_col="itemid", compression="snappy"),
        progress_callback=lambda _: None,
    )

    assert not second.success
    assert "no reusable completion marker" in (second.error or "")


def test_large_csv_low_memory_two_stage_conversion_uses_copy_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "itemid": [220045, 220181, 220045, 220210, 220181, 220045],
            "valuenum": [80.0, 120.0, 81.0, 15.0, 121.0, 82.0],
        }
    ).to_csv(source, index=False)

    real_stat = Path.stat

    def fake_stat(self: Path, *args, **kwargs):
        stat = real_stat(self, *args, **kwargs)
        if self == source:
            values = list(stat)
            values[6] = 2 * 1024 * 1024 * 1024
            return os.stat_result(values)
        return stat

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(bucket_converter, "_get_total_ram_gb", lambda: 3.0)

    output_dir = tmp_path / "events_bucket"
    result = convert_to_buckets(
        source,
        output_dir,
        BucketConfig(num_buckets=12, partition_col="itemid", compression="snappy"),
        progress_callback=lambda _: None,
    )

    assert result.success, result.error

    row_count, actual_buckets, total_size = (output_dir / "_COMPLETE").read_text(
        encoding="utf-8"
    ).split(",")
    assert int(row_count) == 6
    assert int(row_count) == result.total_rows
    assert int(actual_buckets) == result.num_buckets
    assert int(total_size) == result.total_size_bytes


def test_completed_parquet_directory_bucket_conversion_is_reused(tmp_path: Path) -> None:
    source_dir = tmp_path / "events"
    source_dir.mkdir()
    pd.DataFrame({"stay_id": [1, 2, 3], "itemid": [220045, 220181, 220045]}).to_parquet(
        source_dir / "part_1.parquet"
    )

    output_dir = tmp_path / "events_bucket"
    first = convert_parquet_directory_to_buckets(
        source_dir,
        output_dir,
        partition_col="itemid",
        num_buckets=8,
        progress_callback=lambda _: None,
    )
    assert first.success, first.error

    second = convert_parquet_directory_to_buckets(
        source_dir,
        output_dir,
        partition_col="itemid",
        num_buckets=8,
        progress_callback=lambda _: None,
    )

    assert second.success, second.error
    assert second.total_rows == first.total_rows
    assert second.num_buckets == first.num_buckets
    assert second.total_size_bytes == first.total_size_bytes
