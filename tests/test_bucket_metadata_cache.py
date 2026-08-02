from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from easyicu.config import DataSourceConfig
from easyicu.datasource import (
    ICUDataSource,
    load_bucketed_table_aggregated,
    load_bucketed_table_multi_aggregated,
)


def _make_data_source(tmp_path: Path) -> ICUDataSource:
    return ICUDataSource(DataSourceConfig(name="miiv"), base_path=tmp_path)


def _write_aumc_time_fixture(
    tmp_path: Path,
    *,
    admissions_format: str = "parquet",
) -> tuple[ICUDataSource, float]:
    """Create a tiny AUMC source with stay 16075's 23-minute clock phase."""
    admittedat = 13_141_380_000.0
    admissions = pd.DataFrame(
        {
            "admissionid": [16075, 99999],
            "admittedat": [admittedat, admittedat + 60_000.0],
        }
    )
    if admissions_format == "parquet":
        admissions.to_parquet(tmp_path / "admissions.parquet", index=False)
    else:
        admissions.to_csv(tmp_path / "admissions.csv", index=False)

    numericitems = tmp_path / "numericitems"
    numericitems.mkdir()
    pd.DataFrame(
        {
            "admissionid": [
                16075,
                16075,
                16075,
                16075,
                16075,
                16075,
                16075,
                16075,
                99999,
            ],
            "measuredat": [
                admittedat - 1.0,
                admittedat + 2 * 60_000.0,
                admittedat + 14 * 60_000.0,
                admittedat + 74 * 60_000.0,
                admittedat + 4 * 60_000.0,
                admittedat + 4 * 60_000.0,
                admittedat + 4 * 60_000.0,
                admittedat + 74 * 60_000.0,
                admittedat + 2 * 60_000.0,
            ],
            "itemid": [1, 1, 1, 1, 2, 3, 4, 5, 1],
            "value": [70.0, 90.0, 88.0, 95.0, 93.5, 122.5, 74.0, 97.0, 999.0],
        }
    ).to_parquet(numericitems / "part.parquet", index=False)

    source = ICUDataSource(DataSourceConfig(name="aumc"), base_path=tmp_path)
    return source, admittedat


def test_bucket_file_resolution_uses_duckdb_hash_and_skips_unrelated_buckets(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)
    bucket_dir = tmp_path / "chartevents_bucket"
    for bucket_id in range(10):
        (bucket_dir / f"bucket_id={bucket_id}").mkdir(parents=True)

    itemids = {220045, 220181}
    target_buckets = data_source._compute_target_buckets(itemids, 10, duckdb)
    expected_files = []
    for bucket_id in target_buckets:
        file_path = bucket_dir / f"bucket_id={bucket_id}" / f"data_{bucket_id}.parquet"
        file_path.touch()
        expected_files.append(file_path)

    unrelated_bucket = next(bucket_id for bucket_id in range(10) if bucket_id not in target_buckets)
    (bucket_dir / f"bucket_id={unrelated_bucket}" / "unrelated.parquet").touch()
    (bucket_dir / "_COMPLETE").write_text("ready", encoding="utf-8")

    resolved_buckets, num_buckets, resolved_files = data_source._get_bucket_files_for_ids(
        bucket_dir,
        itemids,
        duckdb,
    )

    assert num_buckets == 10
    assert resolved_buckets == target_buckets
    assert set(resolved_files) == set(expected_files)


def test_multi_aggregated_skips_null_value_rows_before_grouping(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)
    bucket_dir = tmp_path / "chartevents_bucket"
    for bucket_id in range(8):
        (bucket_dir / f"bucket_id={bucket_id}").mkdir(parents=True)

    target_bucket = next(iter(data_source._compute_target_buckets({220045}, 8, duckdb)))
    pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": pd.to_datetime(["2020-01-01 00:15:00", "2020-01-01 01:15:00"]),
            "itemid": [220045, 220045],
            "valuenum": [80.0, None],
        }
    ).to_parquet(bucket_dir / f"bucket_id={target_bucket}" / "part.parquet")
    pd.DataFrame(
        {
            "stay_id": [1],
            "hadm_id": [10],
            "intime": pd.to_datetime(["2020-01-01 00:00:00"]),
            "outtime": pd.to_datetime(["2020-01-02 00:00:00"]),
        }
    ).to_parquet(tmp_path / "icustays.parquet")

    result = load_bucketed_table_multi_aggregated(
        data_source,
        "chartevents",
        {"hr": [220045]},
        value_column="valuenum",
        interval_minutes=60.0,
        patient_ids=[1],
    )

    assert result[["stay_id", "charttime", "hr"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 0.0, "hr": 80.0}
    ]


def test_single_aggregated_skips_null_value_rows_before_grouping(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)
    bucket_dir = tmp_path / "chartevents_bucket"
    for bucket_id in range(8):
        (bucket_dir / f"bucket_id={bucket_id}").mkdir(parents=True)

    target_bucket = next(iter(data_source._compute_target_buckets({220045}, 8, duckdb)))
    pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": pd.to_datetime(["2020-01-01 00:15:00", "2020-01-01 01:15:00"]),
            "itemid": [220045, 220045],
            "valuenum": [80.0, None],
        }
    ).to_parquet(bucket_dir / f"bucket_id={target_bucket}" / "part.parquet")
    pd.DataFrame(
        {
            "stay_id": [1],
            "hadm_id": [10],
            "intime": pd.to_datetime(["2020-01-01 00:00:00"]),
            "outtime": pd.to_datetime(["2020-01-02 00:00:00"]),
        }
    ).to_parquet(tmp_path / "icustays.parquet")

    result = load_bucketed_table_aggregated(
        data_source,
        "chartevents",
        "valuenum",
        [220045],
        interval_minutes=60.0,
        patient_ids=[1],
    )

    assert result[["stay_id", "charttime", "valuenum"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 0.0, "valuenum": 80.0}
    ]


@pytest.mark.parametrize("admissions_format", ["parquet", "csv"])
def test_aumc_single_aggregation_floors_icu_relative_time_and_filters_patients(
    tmp_path: Path,
    admissions_format: str,
) -> None:
    source, admittedat = _write_aumc_time_fixture(
        tmp_path,
        admissions_format=admissions_format,
    )

    result = load_bucketed_table_aggregated(
        source,
        "numericitems",
        "value",
        [1],
        interval_minutes=60.0,
        patient_ids=[16075],
    )
    result = result.assign(
        relative_hour=(result["measuredat_minutes"] - admittedat / 60_000.0) / 60.0
    )

    assert result[["admissionid", "relative_hour", "value"]].to_dict(
        "records"
    ) == [
        {"admissionid": 16075, "relative_hour": -1.0, "value": 70.0},
        {"admissionid": 16075, "relative_hour": 0.0, "value": 89.0},
        {"admissionid": 16075, "relative_hour": 1.0, "value": 95.0},
    ]


def test_aumc_stay_16075_single_and_multi_aggregation_have_identical_time_keys(
    tmp_path: Path,
) -> None:
    source, admittedat = _write_aumc_time_fixture(tmp_path)

    single = load_bucketed_table_aggregated(
        source,
        "numericitems",
        "value",
        [1],
        interval_minutes=60.0,
        patient_ids=[16075],
    )
    multi = load_bucketed_table_multi_aggregated(
        source,
        "numericitems",
        {
            "hr": [1],
            "map": [2],
            "sbp": [3],
            "dbp": [4],
            "pulse": [5],
        },
        value_column="value",
        interval_minutes=60.0,
        patient_ids=[16075],
    )

    assert multi["measuredat_minutes"].tolist() == single[
        "measuredat_minutes"
    ].tolist()
    assert multi["hr"].tolist() == pytest.approx(single["value"].tolist())

    relative_hour = (
        multi["measuredat_minutes"] - admittedat / 60_000.0
    ) / 60.0
    multi = multi.assign(relative_hour=relative_hour).set_index("relative_hour")
    assert multi.loc[0.0, ["hr", "map", "sbp", "dbp"]].tolist() == pytest.approx(
        [89.0, 93.5, 122.5, 74.0]
    )
    assert multi.loc[0.0, "sbp"] - multi.loc[0.0, "dbp"] == pytest.approx(48.5)
    assert multi.loc[1.0, ["hr", "pulse"]].tolist() == pytest.approx([95.0, 97.0])


def test_bucket_layout_cache_invalidates_when_complete_marker_changes(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)
    bucket_dir = tmp_path / "labevents_bucket"
    for bucket_id in range(4):
        (bucket_dir / f"bucket_id={bucket_id}").mkdir(parents=True)

    target_buckets = data_source._compute_target_buckets({50813}, 4, duckdb)
    target_bucket = next(iter(target_buckets))
    first_file = bucket_dir / f"bucket_id={target_bucket}" / "first.parquet"
    second_file = bucket_dir / f"bucket_id={target_bucket}" / "second.parquet"
    first_file.touch()
    marker = bucket_dir / "_COMPLETE"
    marker.write_text("v1", encoding="utf-8")

    _, _, first_files = data_source._get_bucket_files_for_ids(bucket_dir, {50813}, duckdb)
    assert set(first_files) == {first_file}

    second_file.touch()
    marker.write_text("version-two", encoding="utf-8")

    _, _, second_files = data_source._get_bucket_files_for_ids(bucket_dir, {50813}, duckdb)
    assert set(second_files) == {first_file, second_file}


def test_bucket_hash_cache_reuses_individual_itemids_for_overlapping_sets(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)

    first = data_source._compute_target_buckets({220045, 220181}, 16, duckdb)
    assert first
    assert len(data_source._bucket_hash_item_cache) == 2

    data_source._bucket_hash_cache.clear()
    second = data_source._compute_target_buckets({220045, 220210}, 16, duckdb)
    expected_second = data_source._compute_target_buckets({220045, 220210}, 16, duckdb)

    assert second == expected_second
    assert len(data_source._bucket_hash_item_cache) == 3


def test_bucket_file_resolution_supports_string_bucket_keys(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)
    bucket_dir = tmp_path / "lab_bucket"
    for bucket_id in range(8):
        (bucket_dir / f"bucket_id={bucket_id}").mkdir(parents=True)

    itemids = {"heart rate", "lactate"}
    target_buckets = data_source._compute_target_buckets(itemids, 8, duckdb)
    expected_files = []
    for bucket_id in target_buckets:
        file_path = bucket_dir / f"bucket_id={bucket_id}" / f"data_{bucket_id}.parquet"
        file_path.touch()
        expected_files.append(file_path)
    (bucket_dir / "_COMPLETE").write_text("ready", encoding="utf-8")

    resolved_buckets, num_buckets, resolved_files = data_source._get_bucket_files_for_ids(
        bucket_dir,
        itemids,
        duckdb,
    )

    assert num_buckets == 8
    assert resolved_buckets == target_buckets
    assert set(resolved_files) == set(expected_files)


def test_bucket_directory_cache_only_stores_positive_results(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)

    assert data_source._resolve_bucket_directory("chartevents") is None
    assert data_source._bucket_dir_cache == {}

    bucket_dir = tmp_path / "chartevents_bucket" / "bucket_id=0"
    bucket_dir.mkdir(parents=True)

    resolved = data_source._resolve_bucket_directory("chartevents")
    assert resolved == tmp_path / "chartevents_bucket"
    assert data_source._bucket_dir_cache["chartevents"] == resolved

    assert data_source._resolve_loader_from_disk("chartevents") == resolved


def test_flat_parquet_directory_cache_only_stores_positive_results(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)

    assert data_source._resolve_flat_parquet_directory("observations") is None
    assert data_source._flat_parquet_dir_cache == {}

    flat_dir = tmp_path / "observations"
    flat_dir.mkdir()
    assert data_source._resolve_flat_parquet_directory("observations") is None
    assert data_source._flat_parquet_dir_cache == {}

    pd.DataFrame({"stay_id": [1], "itemid": [220045]}).to_parquet(flat_dir / "part.parquet")

    resolved = data_source._resolve_flat_parquet_directory("observations")
    assert resolved == flat_dir
    assert data_source._flat_parquet_dir_cache["observations"] == flat_dir

    data_source.clear_cache()
    assert data_source._flat_parquet_dir_cache == {}


def test_parquet_schema_cache_unions_columns_and_invalidates_on_file_change(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    pd.DataFrame({"stay_id": [1], "itemid": [220045]}).to_parquet(first)
    pd.DataFrame({"stay_id": [1], "valuenum": [80.0]}).to_parquet(second)

    columns = data_source._get_parquet_columns_for_files([first, second])
    assert columns == {"stay_id", "itemid", "valuenum"}

    pd.DataFrame({"stay_id": [1], "itemid": [220045], "charttime": [0]}).to_parquet(first)

    updated_columns = data_source._get_parquet_columns_for_files([first, second])
    assert updated_columns == {"stay_id", "itemid", "charttime", "valuenum"}


def test_parquet_schema_cache_reuses_file_level_schema_for_overlapping_sets(tmp_path: Path) -> None:
    data_source = _make_data_source(tmp_path)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    third = tmp_path / "third.parquet"

    pd.DataFrame({"stay_id": [1]}).to_parquet(first)
    pd.DataFrame({"itemid": [220045]}).to_parquet(second)
    pd.DataFrame({"valuenum": [80.0]}).to_parquet(third)

    first_columns = data_source._get_parquet_columns_for_files([second, first])
    assert first_columns == {"stay_id", "itemid"}
    assert len(data_source._parquet_file_columns_cache) == 2

    # Same files, different order should hit the group cache because the key is canonicalized.
    reordered_columns = data_source._get_parquet_columns_for_files([first, second])
    assert reordered_columns == first_columns
    assert len(data_source._parquet_columns_cache) == 1

    overlapping_columns = data_source._get_parquet_columns_for_files([second, third])
    assert overlapping_columns == {"itemid", "valuenum"}
    assert len(data_source._parquet_file_columns_cache) == 3
