"""HiRID-specific DataConverter regressions."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pandas as pd

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


def test_corrupt_single_parquet_output_requires_reconversion(tmp_path: Path) -> None:
    csv_path = tmp_path / "general_table.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    (tmp_path / "general.parquet").write_bytes(b"truncated")

    converter = DataConverter(tmp_path, database="hirid", verbose=False)

    needs_conversion, reason = converter._is_conversion_needed(csv_path)
    assert needs_conversion
    assert reason == "parquet file corrupted"
