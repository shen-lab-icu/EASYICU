from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SEALER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "releases"
    / "EX-A01_seal_full6_release.py"
)
SPEC = importlib.util.spec_from_file_location("easyicu_full6_release_sealer", SEALER_PATH)
assert SPEC is not None and SPEC.loader is not None
sealer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sealer
SPEC.loader.exec_module(sealer)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_synthetic_release(run_root: Path) -> None:
    export_root = run_root / "exports"
    timing_rows = []
    for database in sealer.DATABASES:
        database_root = export_root / database
        database_root.mkdir(parents=True)
        sidecar = database_root / "column_metadata.json"
        sidecar.write_text("{}\n", encoding="utf-8")
        entries = []
        for module in sealer.MODULES:
            if module == "demographics":
                table = pa.table(
                    {
                        "stay_id": pa.array([1, 2], type=pa.int64()),
                        "value": pa.array([1.0, 2.0], type=pa.float64()),
                    }
                )
                primary_key = ["stay_id"]
                row_grain = "one_row_per_icu_stay"
                null_key_equality = "not_applicable"
            else:
                table = pa.table(
                    {
                        "stay_id": pa.array([1, 2], type=pa.int64()),
                        "charttime": pa.array([None, 0.0], type=pa.float64()),
                        "value": pa.array([1.0, 2.0], type=pa.float64()),
                    }
                )
                primary_key = ["stay_id", "charttime"]
                row_grain = "one_row_per_icu_stay_relative_hour"
                null_key_equality = "nulls_equal"
            parquet = database_root / f"{module}.parquet"
            pq.write_table(table, parquet)
            physical_schema = {
                field.name: str(field.type) for field in pq.read_schema(parquet)
            }
            audit = {
                "primary_key": primary_key,
                "row_grain": row_grain,
                "null_key_equality": null_key_equality,
                "source_rows": 2,
                "published_rows": 2,
                "duplicate_excess_rows_after": 0,
            }
            if module != "demographics":
                audit["null_charttime_rows_after"] = 1
            entries.append(
                {
                    "file": parquet.name,
                    "module": module,
                    "rows": 2,
                    "physical_schema": physical_schema,
                    "parquet_sha256": _sha256(parquet),
                    "parquet_bytes": parquet.stat().st_size,
                    "primary_key": primary_key,
                    "row_grain": row_grain,
                    "row_grain_audit": audit,
                    "physical_concept_ids": ["value"],
                }
            )
        manifest = {
            "schema_version": sealer.NATIVE_SCHEMA_VERSION,
            "contract_revision": sealer.CONTRACT_REVISION,
            "database": database,
            "runtime_provenance": {
                "easyicu_git_commit": "a" * 40,
                "easyicu_git_dirty": False,
            },
            "column_metadata": {
                "file": sidecar.name,
                "sha256": _sha256(sidecar),
            },
            "files": entries,
        }
        (database_root / "_manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        timing_rows.append(
            {
                "database": database,
                "status": "complete",
                "elapsed_seconds": "60.0",
                "module_count": str(len(sealer.MODULES)),
                "valid_parquet_count": str(len(sealer.MODULES)),
                "total_rows": str(sum(entry["rows"] for entry in entries)),
                "total_parquet_bytes": str(
                    sum(entry["parquet_bytes"] for entry in entries)
                ),
                "batch_strategy": "one_shot",
                "error": "",
                "process_exit_code": "0",
                "peak_process_tree_rss_mb": "1000.0",
                "peak_process_tree_pss_mb": "900.0",
            }
        )
    timing_path = run_root / "database_extraction_timing.csv"
    with timing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)


def test_sealer_validates_6_by_19_and_atomically_writes_metadata(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "full6_test"
    _build_synthetic_release(run_root)

    destination = sealer.seal_release(
        run_root=run_root, execution_profile="server-adaptive"
    )

    metadata = json.loads(destination.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "full6_test"
    assert metadata["status"] == "verified"
    assert metadata["database_count"] == 6
    assert metadata["module_count"] == 19
    assert metadata["expected_parquet_count"] == 114
    assert metadata["contract_revision"] == sealer.CONTRACT_REVISION
    assert (
        "partial_wide_exact_floating_charttime_key_alignment"
        in metadata["corrections"]
    )
    assert (
        "aumc_relative_time_bucketed_before_admission_alignment"
        in metadata["corrections"]
    )
    assert (
        "eicu_susp_inf_uses_antibiotic_event_time"
        in metadata["corrections"]
    )
    assert metadata["extraction_execution"]["profile"] == "server-adaptive"
    assert not metadata["extraction_execution"]["portable_16gb_validated"]
    assert set(metadata["extraction_execution"]["timing"]["databases"]) == set(
        sealer.DATABASES
    )
    assert metadata["easyicu_commit"] == "a" * 40
    assert set(metadata["source_manifest_sha256"]) == set(sealer.DATABASES)
    assert metadata["validation"] == {
        "audited_parquet_count": 114,
        "valid_footer_count": 114,
        "primary_key_contract_verified_count": 114,
        "primary_key_uniqueness_verified_count": 114,
        "row_grain_contract_verified_count": 114,
        "parquet_sha256_verified_count": 114,
        "parquet_bytes_verified_count": 114,
        "exact_schema_module_count": 19,
        "total_rows": 228,
        "total_parquet_bytes": sum(
            path.stat().st_size
            for path in (run_root / "exports").glob("*/*.parquet")
        ),
        "failures": [],
    }
    assert not list(run_root.glob(".run_metadata.json.*.tmp"))


def test_failed_validation_preserves_existing_run_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "invalid_release"
    _build_synthetic_release(run_root)
    destination = run_root / "run_metadata.json"
    original = b'{"status":"do-not-replace"}\n'
    destination.write_bytes(original)
    manifest_path = run_root / "exports" / "aumc" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["parquet_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(sealer.ReleaseValidationError, match="size mismatch"):
        sealer.seal_release(
            run_root=run_root, execution_profile="server-adaptive"
        )

    assert destination.read_bytes() == original
    assert not list(run_root.glob(".run_metadata.json.*.tmp"))


def test_null_charttime_keys_are_compared_with_nulls_equal(tmp_path: Path) -> None:
    parquet = tmp_path / "duplicate_null_key.parquet"
    pq.write_table(
        pa.table(
            {
                "stay_id": pa.array([1, 1], type=pa.int64()),
                "charttime": pa.array([None, None], type=pa.float64()),
            }
        ),
        parquet,
    )
    connection = duckdb.connect(":memory:")
    try:
        audit = sealer._actual_key_audit(
            connection, parquet, ["stay_id", "charttime"]
        )
    finally:
        connection.close()

    assert audit == {
        "null_stay_id_rows": 0,
        "null_charttime_rows": 2,
        "duplicate_key_groups": 1,
    }


def test_stable_module_contract_matches_public_extractor() -> None:
    from easyicu.api import EXTRACT_MODULE_ORDER

    assert tuple(EXTRACT_MODULE_ORDER) == sealer.MODULES
