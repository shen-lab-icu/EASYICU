from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
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
SPEC = importlib.util.spec_from_file_location(
    "easyicu_full6_release_sealer", SEALER_PATH
)
assert SPEC is not None and SPEC.loader is not None
sealer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sealer
SPEC.loader.exec_module(sealer)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _finalized_foundation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give synthetic releases an internally consistent finalized foundation."""

    foundation = tmp_path / "foundation"
    foundation.mkdir()
    resource_paths = {
        "concept_dictionary_sha256": foundation / "concept-dict.json",
        "sofa2_dictionary_sha256": foundation / "sofa2-dict.json",
        "clinical_contracts_sha256": foundation / "clinical_contracts.py",
        "data_sources_sha256": foundation / "data-sources.json",
    }
    for index, path in enumerate(resource_paths.values(), start=1):
        path.write_text(f"foundation resource {index}\n", encoding="utf-8")
    lock_path = foundation / "concept-dict.LOCK.json"
    lock_path.write_text(
        json.dumps(
            {
                "finalized": True,
                "locked_for_extraction_run": "synthetic_finalized_release",
                "concept_dict_sha256": _sha256(
                    resource_paths["concept_dictionary_sha256"]
                ),
                "sofa2_dict_sha256": _sha256(
                    resource_paths["sofa2_dictionary_sha256"]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sealer,
        "FOUNDATION_RESOURCE_PATHS",
        resource_paths,
        raising=False,
    )
    monkeypatch.setattr(
        sealer,
        "FOUNDATION_LOCK_PATH",
        lock_path,
        raising=False,
    )


def _foundation_runtime_provenance() -> dict[str, str]:
    return {
        field: _sha256(path)
        for field, path in sealer.FOUNDATION_RESOURCE_PATHS.items()
    }


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
                        "charttime": pa.array([0.0, 1.0], type=pa.float64()),
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
                audit["null_charttime_rows_after"] = 0
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
                "easyicu_git_commit": sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT,
                "easyicu_git_dirty": False,
                **_foundation_runtime_provenance(),
            },
            "module_timings_seconds": {
                module: float(index + 1) for index, module in enumerate(sealer.MODULES)
            },
            "module_peak_rss_mb": {
                module: 100.0 + index for index, module in enumerate(sealer.MODULES)
            },
            "module_peak_working_set_mb": {
                module: 90.0 + index for index, module in enumerate(sealer.MODULES)
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
        "partial_wide_exact_floating_charttime_key_alignment" in metadata["corrections"]
    )
    assert (
        "aumc_relative_time_bucketed_before_admission_alignment"
        in metadata["corrections"]
    )
    assert "eicu_susp_inf_uses_antibiotic_event_time" in metadata["corrections"]
    assert metadata["release_gate"] == {
        "contract": sealer.RELEASE_GATE_CONTRACT,
        "contract_revision": sealer.CONTRACT_REVISION,
        "minimum_easyicu_commit": sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT,
        "required_corrections": list(sealer.REQUIRED_HARMONIZED_CORRECTIONS),
        "affected_database_runtime_commits": {
            database: sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT
            for database in sealer.DATABASES
        },
        "foundation": {
            "lock_finalized": True,
            "locked_for_extraction_run": "synthetic_finalized_release",
            "lock_sha256": _sha256(sealer.FOUNDATION_LOCK_PATH),
            **_foundation_runtime_provenance(),
        },
    }
    assert metadata["extraction_execution"]["profile"] == "server-adaptive"
    assert not metadata["extraction_execution"]["portable_16gb_validated"]
    assert set(metadata["extraction_execution"]["timing"]["databases"]) == set(
        sealer.DATABASES
    )
    module_timing = metadata["extraction_execution"]["module_timing"]
    assert module_timing["record_count"] == 114
    assert module_timing["database_count"] == 6
    assert module_timing["module_count"] == 19
    module_timing_path = run_root / module_timing["file"]
    assert _sha256(module_timing_path) == module_timing["sha256"]
    with module_timing_path.open(newline="", encoding="utf-8") as handle:
        module_timing_rows = list(csv.DictReader(handle))
    assert len(module_timing_rows) == 114
    assert {(row["database"], row["module"]) for row in module_timing_rows} == {
        (database, module) for database in sealer.DATABASES for module in sealer.MODULES
    }
    eicu_special = [
        row
        for row in module_timing_rows
        if row["database"] == "eicu"
        and row["module"] in sealer.SPECIAL_SHARED_TIMING_MODULES
    ]
    assert len(eicu_special) == 2
    assert {row["timing_scope"] for row in eicu_special} == {"shared_stage_wall_time"}
    assert {row["shared_stage_id"] for row in eicu_special} == {
        sealer.SPECIAL_SHARED_TIMING_STAGE_ID
    }
    assert all(
        row["timing_scope"] == "module_wall_time" and row["shared_stage_id"] == ""
        for row in module_timing_rows
        if row["module"] not in sealer.SPECIAL_SHARED_TIMING_MODULES
    )
    assert "count that elapsed time once" in module_timing["semantics"]
    assert metadata["easyicu_commit"] == sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT
    assert set(metadata["source_manifest_sha256"]) == set(sealer.DATABASES)
    assert metadata["validation"] == {
        "audited_parquet_count": 114,
        "valid_footer_count": 114,
        "primary_key_contract_verified_count": 114,
        "primary_key_uniqueness_verified_count": 114,
        "row_grain_contract_verified_count": 114,
        "null_time_semantics_verified_count": 114,
        "parquet_sha256_verified_count": 114,
        "parquet_bytes_verified_count": 114,
        "exact_schema_module_count": 19,
        "total_rows": 228,
        "total_parquet_bytes": sum(
            path.stat().st_size for path in (run_root / "exports").glob("*/*.parquet")
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
        sealer.seal_release(run_root=run_root, execution_profile="server-adaptive")

    assert destination.read_bytes() == original
    assert not list(run_root.glob(".run_metadata.json.*.tmp"))


def test_missing_module_timing_fails_closed(tmp_path: Path) -> None:
    run_root = tmp_path / "missing_module_timing"
    _build_synthetic_release(run_root)
    manifest_path = run_root / "exports" / "eicu" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["module_timings_seconds"]["respiratory"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(sealer.ReleaseValidationError, match="module_timings_seconds"):
        sealer.seal_release(run_root=run_root, execution_profile="server-adaptive")

    assert not (run_root / "module_extraction_timing.csv").exists()
    assert not (run_root / "run_metadata.json").exists()


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
        audit = sealer._actual_key_audit(connection, parquet, ["stay_id", "charttime"])
    finally:
        connection.close()

    assert audit == {
        "null_stay_id_rows": 0,
        "null_charttime_rows": 2,
        "duplicate_key_groups": 1,
    }


def test_release_gate_rejects_dynamic_values_at_null_charttime(tmp_path: Path) -> None:
    parquet = tmp_path / "renal.parquet"
    pq.write_table(
        pa.table(
            {
                "stay_id": pa.array([1], type=pa.int64()),
                "charttime": pa.array([None], type=pa.float64()),
                "rrt": pa.array([True], type=pa.bool_()),
            }
        ),
        parquet,
    )
    connection = duckdb.connect(":memory:")
    try:
        with pytest.raises(
            sealer.ReleaseValidationError,
            match="dynamic or undeclared",
        ):
            sealer._validate_null_time_semantics(
                connection,
                parquet_path=parquet,
                database="eicu",
                module="renal",
                concepts=["rrt"],
            )
    finally:
        connection.close()


def test_release_gate_allows_only_declared_admission_level_null_time_values(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "other_scores.parquet"
    pq.write_table(
        pa.table(
            {
                "stay_id": pa.array([1, 2], type=pa.int64()),
                "charttime": pa.array([None, 0.0], type=pa.float64()),
                "apache_iv": pa.array([72.0, None], type=pa.float64()),
                "news": pa.array([None, 3.0], type=pa.float64()),
            }
        ),
        parquet,
    )
    connection = duckdb.connect(":memory:")
    try:
        audit = sealer._validate_null_time_semantics(
            connection,
            parquet_path=parquet,
            database="eicu",
            module="other_scores",
            concepts=["apache_iv", "news"],
        )
    finally:
        connection.close()

    assert audit == {
        "null_charttime_rows": 1,
        "empty_null_charttime_rows": 0,
        "disallowed_null_charttime_rows": 0,
    }


def test_release_gate_rejects_value_less_null_time_artifact(tmp_path: Path) -> None:
    parquet = tmp_path / "other_scores_empty.parquet"
    pq.write_table(
        pa.table(
            {
                "stay_id": pa.array([1], type=pa.int64()),
                "charttime": pa.array([None], type=pa.float64()),
                "apache_iv": pa.array([None], type=pa.float64()),
            }
        ),
        parquet,
    )
    connection = duckdb.connect(":memory:")
    try:
        with pytest.raises(sealer.ReleaseValidationError, match="value-less"):
            sealer._validate_null_time_semantics(
                connection,
                parquet_path=parquet,
                database="eicu",
                module="other_scores",
                concepts=["apache_iv"],
            )
    finally:
        connection.close()


def test_stable_module_contract_matches_public_extractor() -> None:
    from easyicu.api import EXTRACT_MODULE_ORDER

    assert tuple(EXTRACT_MODULE_ORDER) == sealer.MODULES


def test_foundation_gate_rejects_unfinalized_lock() -> None:
    lock = json.loads(sealer.FOUNDATION_LOCK_PATH.read_text(encoding="utf-8"))
    lock["finalized"] = False
    sealer.FOUNDATION_LOCK_PATH.write_text(
        json.dumps(lock) + "\n", encoding="utf-8"
    )

    with pytest.raises(sealer.ReleaseValidationError, match="not finalized"):
        sealer._validate_foundation_lock()


def test_foundation_gate_rejects_dictionary_hash_drift() -> None:
    sealer.FOUNDATION_RESOURCE_PATHS[
        "concept_dictionary_sha256"
    ].write_text("drifted dictionary\n", encoding="utf-8")

    with pytest.raises(
        sealer.ReleaseValidationError,
        match="concept_dictionary_sha256.*does not match",
    ):
        sealer._validate_foundation_lock()


def test_release_gate_rejects_manifest_foundation_digest_drift(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "foundation_digest_drift"
    _build_synthetic_release(run_root)
    manifest_path = run_root / "exports" / "aumc" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_provenance"]["concept_dictionary_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(
        sealer.ReleaseValidationError,
        match="aumc: foundation provenance.*concept_dictionary_sha256",
    ):
        sealer.validate_release(run_root)


def test_harmonized_minimum_includes_aumc_rrt_interval_fix() -> None:
    assert sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT == (
        "187c6123ea59b4d904a2594d755de4186dc249b5"
    )
    assert (
        "aumc_rrt_derived_from_processitem_treatment_intervals"
        in sealer.REQUIRED_HARMONIZED_CORRECTIONS
    )


def test_sealer_expands_sofa2_owner_receipts_in_physical_column_order() -> None:
    assert sealer._physical_columns_from_manifest_concepts(
        ["stay_id", "charttime"],
        ["sofa2", "sofa2_cns_ascertainment", "hr"],
    ) == [
        "stay_id",
        "charttime",
        "sofa2",
        "sofa2_observed",
        "sofa2_available",
        "sofa2_cns_ascertainment",
        "sofa2_cns_ascertainment_observed",
        "sofa2_cns_ascertainment_available",
        "hr",
    ]


def test_harmonized_ancestry_rejects_old_and_unknown_runtime_commits() -> None:
    commits = {
        database: sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT
        for database in sealer.DATABASES
    }
    sealer._validate_harmonized_commit_ancestry(commits)

    commits = {
        database: "ebb802598c8c6d2ae55715dde1699075260f9429"
        for database in sealer.DATABASES
    }
    with pytest.raises(sealer.ReleaseValidationError, match="predates required"):
        sealer._validate_harmonized_commit_ancestry(commits)

    commits = {database: "f" * 40 for database in sealer.DATABASES}
    with pytest.raises(
        sealer.ReleaseValidationError,
        match="Fetch full history.*shallow",
    ):
        sealer._validate_harmonized_commit_ancestry(commits)


def test_harmonized_gate_requires_one_identical_six_database_commit() -> None:
    commits = {
        database: sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT
        for database in sealer.DATABASES
    }
    commits["sic"] = "ebb802598c8c6d2ae55715dde1699075260f9429"
    with pytest.raises(sealer.ReleaseValidationError, match="one identical clean"):
        sealer._validate_harmonized_commit_ancestry(commits)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, filename: str, content: str, message: str) -> str:
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_corrected_time_gate_uses_real_git_dag_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    root = _commit(repo, "history.txt", "root\n", "root")
    minimum = _commit(repo, "fix.txt", "corrected time\n", "corrected-time")
    descendant = _commit(repo, "runtime.txt", "runtime\n", "runtime")
    monkeypatch.setattr(sealer, "MINIMUM_HARMONIZED_EASYICU_COMMIT", minimum)

    commits = {database: descendant for database in sealer.DATABASES}
    sealer._validate_harmonized_commit_ancestry(
        commits,
        repository=repo,
    )

    _git(repo, "checkout", "--detach", root)
    divergent = _commit(repo, "divergent.txt", "old branch\n", "divergent")
    commits = {database: divergent for database in sealer.DATABASES}
    with pytest.raises(sealer.ReleaseValidationError, match="predates required"):
        sealer._validate_harmonized_commit_ancestry(
            commits,
            repository=repo,
        )

    commits = {database: "f" * 40 for database in sealer.DATABASES}
    with pytest.raises(sealer.ReleaseValidationError, match="not available"):
        sealer._validate_harmonized_commit_ancestry(
            commits,
            repository=repo,
        )


@pytest.mark.parametrize("bad", [None, "abc", "A" * 40])
def test_corrected_time_gate_rejects_missing_or_noncanonical_sha(bad) -> None:
    commits = {
        database: sealer.MINIMUM_HARMONIZED_EASYICU_COMMIT
        for database in sealer.DATABASES
    }
    commits["eicu"] = bad
    with pytest.raises(sealer.ReleaseValidationError, match="full lowercase"):
        sealer._validate_harmonized_commit_ancestry(commits)
