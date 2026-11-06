"""DataConverter conversion manifests are evidence-bindable."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.io.data_converter import DataConverter


import duckdb
import pyarrow
import pytest

from easyicu.io import conversion_provenance
from tests.support.bucket_storage import write_contract


def test_data_converter_writes_conversion_manifest_and_evidence(tmp_path: Path) -> None:
    data_dir = tmp_path / "miiv"
    data_dir.mkdir()
    csv_path = data_dir / "sample.csv"
    pd.DataFrame({"stay_id": [1, 2], "value": [3.0, 4.0]}).to_csv(
        csv_path,
        index=False,
    )

    evidence_root = tmp_path / "run"
    converter = DataConverter(
        data_dir,
        database="miiv",
        parallel_workers=1,
        verbose=False,
    )
    results = converter.convert_all(force=True, evidence_root=evidence_root)

    assert results["sample.csv"]["status"] == "completed"
    manifest_path = data_dir / "conversion_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "easyicu.conversion_manifest/1"
    assert manifest["database"] == "miiv"
    table = manifest["tables"][0]
    assert table["input"]["sha256"]
    assert table["outputs"][0]["sha256"]
    assert table["outputs"][0]["relative_path"] == "sample.parquet"

    aliases = json.loads(
        (evidence_root / "evidence" / "evidence_aliases.json").read_text(
            encoding="utf-8"
        )
    )
    assert "conversion_manifest" in aliases


@pytest.mark.parametrize(
    "id_partitioning,expected", [("1", "range"), ("0", "row-slice")]
)
def test_manifest_records_executed_branch_and_conversion_time_switches(
    tmp_path, monkeypatch, id_partitioning, expected
):
    monkeypatch.setenv("EASYICU_USE_ID_PARTITIONING", id_partitioning)
    pd.DataFrame(
        {"stay_id": [1, 2], "itemid": [220045, 220045], "valuenum": [71.0, 72.0]}
    ).to_csv(tmp_path / "chartevents.csv", index=False)
    converter = DataConverter(
        tmp_path, database="miiv", parallel_workers=1, verbose=False
    )
    result = converter._convert_file(tmp_path / "chartevents.csv")
    assert result["status"] == "completed", result
    monkeypatch.setenv(
        "EASYICU_USE_ID_PARTITIONING", "0" if id_partitioning == "1" else "1"
    )
    path = converter.write_conversion_manifest({"chartevents.csv": result})
    manifest = json.loads(path.read_text())
    provenance = manifest["tables"][0]["storage_provenance"]
    assert provenance["branch"] == expected
    assert provenance["effective_environment"]["EASYICU_USE_ID_PARTITIONING"] == (
        id_partitioning == "1"
    )
    assert provenance["git_sha"] and len(provenance["git_sha"]) == 40
    assert provenance["pyarrow_version"] == pyarrow.__version__
    assert provenance["duckdb_version"] == duckdb.__version__


def test_source_shard_branch_and_unavailable_git_are_explicit(tmp_path, monkeypatch):
    path = tmp_path / "observation_tables/csv/part-0.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "patientid": [1],
            "variableid": [200],
            "value": [71.0],
            "datetime": ["2020-01-01"],
        }
    ).to_csv(path, index=False)
    converter = DataConverter(
        tmp_path, database="hirid", parallel_workers=1, verbose=False
    )

    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(conversion_provenance.subprocess, "run", unavailable)
    result = converter._convert_file(path)
    assert result["status"] == "completed", result
    assert result["storage_provenance"]["branch"] == "source-shard"
    assert result["storage_provenance"]["git_sha"] is None


def test_manifest_records_the_selected_bucket_output_without_inventing_historical_switches(
    tmp_path,
):
    path = tmp_path / "chartevents.csv"
    pd.DataFrame({"itemid": [220045], "valuenum": [71.0]}).to_csv(path, index=False)
    cache = tmp_path / "chartevents_bucket/bucket_id=0"
    cache.mkdir(parents=True)
    pd.DataFrame({"itemid": [220045], "valuenum": [71.0]}).to_parquet(
        cache / "1.parquet", index=False
    )
    write_contract(cache.parent, key="itemid")
    converter = DataConverter(
        tmp_path, database="miiv", parallel_workers=1, verbose=False
    )
    manifest = json.loads(
        converter.write_conversion_manifest(
            {"chartevents.csv": {"table": "chartevents", "status": "skipped"}}
        ).read_text()
    )
    entry = manifest["tables"][0]
    assert entry["storage_provenance"]["branch"] == "bucket"
    assert entry["storage_provenance"]["effective_environment"] is None
    assert (
        entry["outputs"][0]["relative_path"]
        == "chartevents_bucket/bucket_id=0/1.parquet"
    )


def test_manifest_effective_cluster_switch_includes_explicit_api_override(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("EASYICU_CLUSTER_BY_PATIENT", raising=False)
    # One source table: an empty folder is refused (conversion_no_source_files).
    (tmp_path / "admissions.csv").write_text("subject_id,hadm_id\n1,11\n", encoding="utf-8")
    converter = DataConverter(
        tmp_path, database="miiv", parallel_workers=1, verbose=False
    )
    converter.convert_all(cluster_by_patient=True)
    manifest = json.loads((tmp_path / "conversion_manifest.json").read_text())
    assert (
        manifest["converter"]["effective_environment"]["EASYICU_CLUSTER_BY_PATIENT"]
        is True
    )
