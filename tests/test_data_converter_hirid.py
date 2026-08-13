"""HiRID-specific DataConverter regressions."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pandas as pd

from easyicu.content_identity import file_content_receipt
from easyicu.io.data_converter import DataConverter


def test_hirid_table_name_mapping_uses_ricu_targets(tmp_path: Path) -> None:
    converter = DataConverter(tmp_path, database="hirid", verbose=False)

    assert converter._get_table_name(Path("general_table.csv")) == "general"
    assert converter._get_parquet_path(Path("general_table.csv")) == (
        tmp_path / "general.parquet"
    )
    assert converter._get_shard_dir(Path("raw_stage/observation_tables.csv")) == (
        tmp_path / "observations"
    )
    assert converter._should_shard(Path("raw_stage/observation_tables.csv"))


def test_hirid_tar_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    data_dir = tmp_path / "hirid"
    data_dir.mkdir()
    archive_path = data_dir / "reference_data.tar.gz"

    payload = b"escaped"
    with tarfile.open(archive_path, "w:gz") as tar:
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    converter = DataConverter(data_dir, database="hirid", verbose=False)

    assert converter._extract_hirid_archives() == []
    assert not (tmp_path / "outside.txt").exists()
    assert not (data_dir / "general_table.csv").exists()


def test_hirid_parquet_conversion_replaces_truncated_target_shard(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "hirid"
    source_dir = data_dir / "observation_tables" / "parquet"
    source_dir.mkdir(parents=True)
    pd.DataFrame({"patientid": [1], "variableid": [110]}).to_parquet(
        source_dir / "part-0.parquet",
        index=False,
    )

    target_dir = data_dir / "observations"
    target_dir.mkdir()
    target_file = target_dir / "1.parquet"
    target_file.write_bytes(b"truncated")

    converter = DataConverter(data_dir, database="hirid", verbose=False)
    converter._convert_hirid_parquet("observation_tables", "observations")

    assert converter._has_parquet_footer(target_file)
    out = pd.read_parquet(target_file)
    assert out["patientid"].tolist() == [1]


def test_hirid_archive_extracts_when_existing_target_shards_are_partial(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "hirid"
    raw_stage = data_dir / "raw_stage"
    raw_stage.mkdir(parents=True)
    archive_path = raw_stage / "observation_tables_parquet.tar.gz"

    source_fixture = tmp_path / "fixture"
    source_fixture.mkdir()
    part0 = source_fixture / "part-0.parquet"
    part1 = source_fixture / "part-1.parquet"
    pd.DataFrame({"patientid": [1], "variableid": [110]}).to_parquet(
        part0,
        index=False,
    )
    pd.DataFrame({"patientid": [2], "variableid": [120]}).to_parquet(
        part1,
        index=False,
    )
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(part0, arcname="observation_tables/parquet/part-0.parquet")
        tar.add(part1, arcname="observation_tables/parquet/part-1.parquet")

    target_dir = data_dir / "observations"
    target_dir.mkdir()
    pd.DataFrame({"patientid": [1], "variableid": [110]}).to_parquet(
        target_dir / "1.parquet",
        index=False,
    )

    converter = DataConverter(data_dir, database="hirid", verbose=False)
    extracted = converter._extract_hirid_archives()

    assert extracted == ["observation_tables_parquet.tar.gz"]
    assert converter._has_parquet_footer(target_dir / "2.parquet")
    out = pd.read_parquet(target_dir / "2.parquet")
    assert out["patientid"].tolist() == [2]


def test_hirid_csv_shards_not_skipped_when_existing_parquet_is_corrupt(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "hirid"
    csv_dir = data_dir / "observation_tables" / "csv"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "part-0.csv"
    csv_path.write_text("patientid,variableid\n1,110\n", encoding="utf-8")

    target_dir = data_dir / "observations"
    target_dir.mkdir()
    (target_dir / "1.parquet").write_bytes(b"truncated")

    converter = DataConverter(data_dir, database="hirid", verbose=False)

    assert csv_path in converter._get_csv_files()


def test_hirid_same_named_csv_shards_keep_distinct_table_identity(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "hirid"
    observation_csv = (
        data_dir / "observation_tables" / "csv" / "part-0.csv"
    )
    pharma_csv = data_dir / "pharma_records" / "csv" / "part-0.csv"
    observation_csv.parent.mkdir(parents=True)
    pharma_csv.parent.mkdir(parents=True)
    observation_csv.write_text(
        "patientid,variableid,value\n1,110,<1.0\n",
        encoding="utf-8",
    )
    pharma_csv.write_text(
        "patientid,pharmaid,givendose\n1,20,2.5\n",
        encoding="utf-8",
    )

    converter = DataConverter(data_dir, database="hirid", verbose=False)
    discovered = converter._get_csv_files()

    assert set(discovered) == {observation_csv, pharma_csv}
    assert converter._get_table_name(observation_csv) == "observations"
    assert converter._get_table_name(pharma_csv) == "pharma"
    assert converter._get_parquet_path(observation_csv) == (
        data_dir / "observations" / "1.parquet"
    )
    assert converter._get_parquet_path(pharma_csv) == (
        data_dir / "pharma" / "1.parquet"
    )
    assert not converter._should_shard(observation_csv)
    assert not converter._should_shard(pharma_csv)
    assert converter._status_key(observation_csv) != converter._status_key(
        pharma_csv
    )


def test_hirid_csv_shards_convert_to_distinct_ricu_outputs(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "hirid"
    observation_csv = (
        data_dir / "observation_tables" / "csv" / "part-0.csv"
    )
    pharma_csv = data_dir / "pharma_records" / "csv" / "part-0.csv"
    observation_csv.parent.mkdir(parents=True)
    pharma_csv.parent.mkdir(parents=True)
    observation_csv.write_text(
        "patientid,variableid,stringvalue,value\n"
        "1,110,13.30,13.3\n"
        "2,120,<1.0,0.9\n",
        encoding="utf-8",
    )
    pharma_csv.write_text(
        "patientid,pharmaid,givendose\n1,20,2.5\n",
        encoding="utf-8",
    )

    converter = DataConverter(
        data_dir,
        database="hirid",
        parallel_workers=2,
        verbose=False,
    )
    results = converter.convert_all(write_manifest=False)

    assert set(results) == {
        "observation_tables/csv/part-0.csv",
        "pharma_records/csv/part-0.csv",
    }
    assert all(result["status"] == "completed" for result in results.values())
    observation_out = pd.read_parquet(
        data_dir / "observations" / "1.parquet"
    )
    pharma_out = pd.read_parquet(data_dir / "pharma" / "1.parquet")
    assert observation_out["stringvalue"].tolist() == ["13.30", "<1.0"]
    assert observation_out["value"].tolist() == [13.3, 0.9]
    assert pharma_out["givendose"].tolist() == [2.5]

    reloaded = DataConverter(data_dir, database="hirid", verbose=False)
    assert reloaded.is_ready() == (True, [])


def test_hirid_csv_shard_completion_is_checked_per_target(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "hirid"
    csv_dir = data_dir / "observation_tables" / "csv"
    csv_dir.mkdir(parents=True)
    first = csv_dir / "part-0.csv"
    second = csv_dir / "part-1.csv"
    first.write_text("patientid,variableid,value\n1,110,1\n", encoding="utf-8")
    second.write_text("patientid,variableid,value\n2,120,2\n", encoding="utf-8")

    converter = DataConverter(data_dir, database="hirid", verbose=False)
    first_result = converter._convert_file(first)
    assert first_result["status"] == "completed"

    discovered = converter._get_csv_files()
    assert set(discovered) == {first, second}
    assert converter._is_conversion_needed(first) == (
        False,
        "already converted and verified",
    )
    assert converter._is_conversion_needed(second) == (
        True,
        "parquet file does not exist",
    )


def test_corrupt_single_parquet_output_requires_reconversion(tmp_path: Path) -> None:
    csv_path = tmp_path / "general_table.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    (tmp_path / "general.parquet").write_bytes(b"truncated")

    converter = DataConverter(tmp_path, database="hirid", verbose=False)

    needs_conversion, reason = converter._is_conversion_needed(csv_path)
    assert needs_conversion
    assert reason == "parquet file corrupted"


def test_hirid_readiness_rejects_untracked_source_named_general_table_parquet(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "general_table.csv"
    csv_path.write_text("patientid,age\n1,70\n", encoding="utf-8")
    pd.DataFrame({"patientid": [1], "age": [70]}).to_parquet(
        tmp_path / "general_table.parquet",
        index=False,
    )

    converter = DataConverter(tmp_path, database="hirid", verbose=False)

    needs_conversion, reason = converter._is_conversion_needed(csv_path)
    assert needs_conversion
    assert reason == "parquet exists without completed conversion status"
    ready, missing = converter.is_ready()
    assert not ready
    assert missing == [
        "general_table.csv: parquet exists without completed conversion status"
    ]


def test_hirid_status_verification_uses_selected_parquet_candidate(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "general_table.csv"
    csv_path.write_text("patientid,age\n1,70\n", encoding="utf-8")
    pd.DataFrame({"patientid": [1], "age": [70]}).to_parquet(
        tmp_path / "general_table.parquet",
        index=False,
    )

    converter = DataConverter(tmp_path, database="hirid", verbose=False)
    converter._status[csv_path.name] = {
        "status": "completed",
        "row_count": 1,
        "source_content_receipt": file_content_receipt(csv_path),
    }

    needs_conversion, reason = converter._is_conversion_needed(csv_path)
    assert not needs_conversion
    assert reason == "already converted and verified"
