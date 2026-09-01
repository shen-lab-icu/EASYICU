"""Contract tests for the canonical six-database extraction controller."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[2] / "tools" / "reextract_native_export_v2.py"
SPEC = importlib.util.spec_from_file_location("reextract_native_export_v2", TOOL)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

COMMIT = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_native_export(root: Path, *, database: str = "miiv") -> dict:
    root.mkdir(parents=True)
    sidecar = root / "column_metadata.json"
    sidecar.write_text("{}\n", encoding="utf-8")
    entries = []
    for index, module in enumerate(launcher.MODULE_ORDER):
        parquet = root / f"{module}.parquet"
        parquet.write_bytes(b"PAR1" + module.encode("utf-8"))
        entries.append(
            {
                "module": module,
                "file": parquet.name,
                "rows": index + 1,
                "parquet_bytes": parquet.stat().st_size,
                "parquet_sha256": _sha256(parquet),
            }
        )
    manifest = {
        "schema_version": launcher.NATIVE_SCHEMA_VERSION,
        "database": database,
        "runtime_provenance": {
            "easyicu_git_commit": COMMIT,
            "easyicu_git_dirty": False,
        },
        "module_timings_seconds": {
            module: float(index + 1)
            for index, module in enumerate(launcher.MODULE_ORDER)
        },
        "module_peak_rss_mb": {
            module: 100.0 + index
            for index, module in enumerate(launcher.MODULE_ORDER)
        },
        "module_peak_working_set_mb": {
            module: 90.0 + index
            for index, module in enumerate(launcher.MODULE_ORDER)
        },
        "stream_retry_history": [],
        "column_metadata": {
            "file": sidecar.name,
            "sha256": _sha256(sidecar),
        },
        "files": entries,
    }
    (root / "_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return manifest


@pytest.mark.parametrize(
    ("total_gb", "available_gb", "expected"),
    [
        (16, 8, 1),
        (16, 16, 1),
        (64, 20, 1),
        (64, 32, 2),
        (128, 48, 3),
        (128, 96, 3),
    ],
)
def test_database_concurrency_is_memory_adaptive_and_capped_at_three(
    total_gb: int, available_gb: int, expected: int
) -> None:
    memory = {
        "effective_total_mb": total_gb * 1024,
        "effective_available_mb": available_gb * 1024,
    }
    assert launcher._database_worker_count(memory, 3) == expected


def test_single_worker_stream_planning_reserves_available_memory_only_once() -> None:
    memory = {
        "effective_total_mb": 16 * 1024,
        "effective_available_mb": 8 * 1024,
    }
    assigned = launcher._assigned_worker_memory_mb(memory, 1)

    assert assigned == 6 * 1024
    assert launcher._stream_planning_memory_mb(memory, 1, assigned) == 8 * 1024
    assert launcher._stream_planning_memory_mb(memory, 2, assigned) == assigned


def test_data_root_maps_database_aliases_and_overrides_one_path(tmp_path: Path) -> None:
    data_root = tmp_path / "databases"
    for directory in launcher.DATABASE_DIRECTORY_NAMES.values():
        (data_root / directory).mkdir(parents=True)
    alternate = tmp_path / "alternate_miiv"
    alternate.mkdir()
    args = launcher._parse_args(
        [
            "--output-root",
            str(tmp_path / "run"),
            "--data-root",
            str(data_root),
            "--data-path",
            f"miiv={alternate}",
        ]
    )

    paths = launcher._resolve_data_paths(args)

    assert paths["miiv"] == str(alternate.resolve())
    assert paths["mimic"] == str((data_root / "mimiciii").resolve())
    assert paths["eicu"] == str((data_root / "eicu").resolve())


def test_data_root_detects_versioned_mimiciii_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "databases"
    for directory in launcher.DATABASE_DIRECTORY_NAMES.values():
        (data_root / directory).mkdir(parents=True)
    versioned = data_root / "mimiciii" / "1.4"
    versioned.mkdir()
    (versioned / "ICUSTAYS.csv.gz").touch()
    args = launcher._parse_args(
        ["--output-root", str(tmp_path / "run"), "--data-root", str(data_root)]
    )

    paths = launcher._resolve_data_paths(args)

    assert paths["mimic"] == str(versioned.resolve())


def test_dirty_checkout_fails_closed() -> None:
    with pytest.raises(launcher.ExtractionRunError, match="clean EasyICU checkout"):
        launcher._require_clean_identity(
            {
                "dirty": True,
                "dirty_status": [" M src/easyicu/api/extraction.py"],
            }
        )


def test_native_package_validation_binds_19_module_time_peak_rows_and_bytes(
    tmp_path: Path,
) -> None:
    export = tmp_path / "export"
    _build_native_export(export)

    receipt = launcher._validate_export_package(export, COMMIT, "miiv")

    assert receipt["module_count"] == 19
    assert receipt["valid_parquet_count"] == 19
    assert receipt["total_rows"] == sum(range(1, 20))
    assert set(receipt["module_metrics"]) == set(launcher.MODULE_ORDER)
    assert receipt["module_metrics"]["respiratory"]["peak_rss_mb"] > 0


def test_native_package_validation_rejects_wrong_database_and_extra_parquet(
    tmp_path: Path,
) -> None:
    export = tmp_path / "export"
    _build_native_export(export)
    with pytest.raises(launcher.ExtractionRunError, match="database mismatch"):
        launcher._validate_export_package(export, COMMIT, "eicu")

    (export / "extra.parquet").write_bytes(b"PAR1extra")
    with pytest.raises(launcher.ExtractionRunError, match="Parquet set mismatch"):
        launcher._validate_export_package(export, COMMIT, "miiv")


def test_timing_csv_is_sealer_compatible_and_replaced_atomically(tmp_path: Path) -> None:
    path = tmp_path / "database_extraction_timing.csv"
    source = {
        "status": "complete",
        "elapsed_seconds": 61.5,
        "module_count": 19,
        "valid_parquet_count": 19,
        "total_rows": 123,
        "total_parquet_bytes": 456,
        "batch_strategy": "one_shot:10_stays",
        "process_exit_code": 0,
        "peak_process_tree_rss_mb": 1000.0,
        "peak_process_tree_pss_mb": 900.0,
        "initial_batch_size": 10,
        "planned_batch_count": 1,
        "stream_retry_count": 0,
        "attempt_count": 1,
        "easyicu_git_commit": COMMIT,
    }
    monitoring = {
        "backend": "psutil_process_tree",
        "process_tree_pss_supported": True,
        "release_sealable": True,
    }
    row = launcher._timing_row("miiv", source, monitoring=monitoring)

    launcher._atomic_write_timing_csv(path, [row])

    with path.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["status"] == "complete"
    assert persisted[0]["module_count"] == "19"
    assert persisted[0]["peak_process_tree_pss_mb"] == "900.0"
    assert not list(tmp_path.glob(".database_extraction_timing.csv.*.tmp"))


def test_monitored_worker_forces_current_source_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class _Process:
        pid = 123

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", _Process)
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")

    result = launcher._run_monitored_worker(
        spec_path=spec,
        log_path=tmp_path / "worker.log",
        psutil_module=None,
        sample_interval_seconds=0.02,
    )

    assert result["process_exit_code"] == 0
    assert captured["cwd"] == launcher.REPOSITORY_ROOT
    assert captured["env"]["PYTHONPATH"] == str(launcher.SOURCE_ROOT)
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"


def test_worker_passes_one_recorded_initial_batch_and_keeps_adaptation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as public_api
    import easyicu.api.extraction as extraction

    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    captured = {}

    monkeypatch.setattr(
        launcher,
        "_git_identity",
        lambda: {
            "repository_root": str(launcher.REPOSITORY_ROOT),
            "commit": COMMIT,
            "dirty": False,
            "dirty_status": [],
        },
    )
    monkeypatch.setattr(
        launcher,
        "_configure_worker_runtime",
        lambda _attempt, assigned: {"assigned_memory_mb": assigned},
    )
    monkeypatch.setattr(
        extraction,
        "_get_all_patient_ids",
        lambda *_args, **_kwargs: (list(range(61_532)), "stay_id"),
    )

    def fake_extract_database(database, **kwargs):
        captured["database"] = database
        captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir()
        return {
            "batch_size": kwargs["batch_size"],
            "stream_retry_history": [],
            "modules": {
                module: {"errors": []} for module in launcher.MODULE_ORDER
            },
        }

    monkeypatch.setattr(public_api, "extract_database", fake_extract_database)
    monkeypatch.setattr(
        launcher,
        "_validate_export_package",
        lambda *_args, **_kwargs: {"module_count": 19},
    )
    monkeypatch.setenv("PYTHONPATH", str(launcher.SOURCE_ROOT))
    spec_path = attempt_root / "worker_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "database": "mimic",
                "data_path": "/data/mimic",
                "attempt_root": str(attempt_root),
                "easyicu_git_commit": COMMIT,
                "assigned_memory_mb": 6 * 1024,
                "planning_memory_mb": 8 * 1024,
                "adaptive_core": True,
                "requested_batch_size": None,
            }
        ),
        encoding="utf-8",
    )

    assert launcher._worker_main(spec_path) == 0

    plan = json.loads((attempt_root / "worker_plan.json").read_text())
    result = json.loads((attempt_root / "worker_result.json").read_text())
    assert plan["assigned_memory_mb"] == 6 * 1024
    assert plan["planning_memory_mb"] == 8 * 1024
    assert plan["planned_initial_batch_size"] == 20_000
    assert captured["batch_size"] == plan["planned_initial_batch_size"]
    assert captured["adaptive_stream_batches"] is True
    assert result["batch_strategy"]["initial_batch_size"] == 20_000


def test_successful_database_is_atomically_promoted_and_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    (run_root / "exports").mkdir(parents=True)

    def fake_worker(**kwargs):
        spec = json.loads(Path(kwargs["spec_path"]).read_text(encoding="utf-8"))
        attempt = Path(spec["attempt_root"])
        export = attempt / "export"
        _build_native_export(export)
        package = launcher._validate_export_package(export, COMMIT, "miiv")
        (attempt / "worker_plan.json").write_text(
            json.dumps(
                {
                    "planned_initial_batch_size": 50_000,
                    "planned_batch_count": 2,
                    "adaptive_core": True,
                }
            ),
            encoding="utf-8",
        )
        (attempt / "worker_result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "package_receipt": package,
                    "batch_strategy": {
                        "label": "adaptive_streamed:50000_stays_x2;memory_retries=0",
                        "initial_batch_size": 50_000,
                        "planned_batch_count": 2,
                        "stream_retry_history": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "process_exit_code": 0,
            "elapsed_seconds": 2.0,
            "peak_process_tree_rss_mb": 1000.0,
            "peak_process_tree_pss_mb": 900.0,
            "monitor_errors": [],
        }

    monkeypatch.setattr(launcher, "_run_monitored_worker", fake_worker)
    source = launcher._execute_database(
        database="miiv",
        run_root=run_root,
        data_path="/data/miiv",
        git_commit=COMMIT,
        assigned_memory_mb=16 * 1024,
        adaptive_core=True,
        requested_batch_size=None,
        max_memory_retries=2,
        sample_interval_seconds=0.1,
        psutil_module=None,
        monitoring={"release_sealable": True},
        prior_source=None,
    )

    assert source["status"] == "complete"
    assert source["module_count"] == 19
    assert (run_root / "exports" / "miiv" / "vitals.parquet").is_file()
    assert not (
        run_root / ".orchestration" / "attempts" / "miiv" / "attempt-01" / "export"
    ).exists()

    with pytest.raises(launcher.ExtractionRunError, match="overwrite"):
        launcher._execute_database(
            database="miiv",
            run_root=run_root,
            data_path="/data/miiv",
            git_commit=COMMIT,
            assigned_memory_mb=16 * 1024,
            adaptive_core=True,
            requested_batch_size=None,
            max_memory_retries=0,
            sample_interval_seconds=0.1,
            psutil_module=None,
            monitoring={"release_sealable": True},
            prior_source=source,
        )


def test_process_oom_downbatches_only_failed_staging_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    (run_root / "exports").mkdir(parents=True)
    observed_batches = []

    def fake_worker(**kwargs):
        spec = json.loads(Path(kwargs["spec_path"]).read_text(encoding="utf-8"))
        attempt = Path(spec["attempt_root"])
        requested = spec.get("requested_batch_size")
        planned = 40_000 if requested is None else int(requested)
        observed_batches.append(planned)
        (attempt / "worker_plan.json").write_text(
            json.dumps(
                {
                    "planned_initial_batch_size": planned,
                    "planned_batch_count": 6,
                    "adaptive_core": requested is None,
                }
            ),
            encoding="utf-8",
        )
        if len(observed_batches) == 1:
            (attempt / "export").mkdir()
            (attempt / "worker_result.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error": (
                            "ExtractionRunError: module extraction errors: "
                            "streamed module export exhausted memory: other_scores"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            return {
                "process_exit_code": 1,
                "elapsed_seconds": 1.0,
                "peak_process_tree_rss_mb": 7000.0,
                "peak_process_tree_pss_mb": 6500.0,
                "monitor_errors": [],
            }
        export = attempt / "export"
        _build_native_export(export, database="eicu")
        package = launcher._validate_export_package(export, COMMIT, "eicu")
        (attempt / "worker_result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "package_receipt": package,
                    "batch_strategy": {
                        "label": "planned_streamed:30000_stays_x7;memory_retries=0",
                        "initial_batch_size": planned,
                        "planned_batch_count": 7,
                        "stream_retry_history": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "process_exit_code": 0,
            "elapsed_seconds": 2.0,
            "peak_process_tree_rss_mb": 5000.0,
            "peak_process_tree_pss_mb": 4500.0,
            "monitor_errors": [],
        }

    monkeypatch.setattr(launcher, "_run_monitored_worker", fake_worker)
    source = launcher._execute_database(
        database="eicu",
        run_root=run_root,
        data_path="/data/eicu",
        git_commit=COMMIT,
        assigned_memory_mb=8 * 1024,
        adaptive_core=True,
        requested_batch_size=None,
        max_memory_retries=2,
        sample_interval_seconds=0.1,
        psutil_module=None,
        monitoring={"release_sealable": True},
        prior_source=None,
    )

    assert observed_batches == [40_000, 30_000]
    assert source["attempt_count"] == 2
    assert source["elapsed_seconds"] == 3.0
    assert source["peak_process_tree_rss_mb"] == 7000.0
    assert (run_root / "exports" / "eicu" / "renal.parquet").is_file()


def test_streamed_python_memory_error_is_retryable() -> None:
    assert launcher._looks_like_memory_failure(
        1,
        "streamed module export exhausted memory: sofa1_score",
    )


def test_resume_recovers_completed_database_and_returns_only_missing(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    final = run_root / "exports" / "miiv"
    _build_native_export(final)
    package = launcher._validate_export_package(final, COMMIT, "miiv")
    source = {
        "status": "complete",
        "database": "miiv",
        "easyicu_git_commit": COMMIT,
        **package,
    }
    receipt = run_root / ".orchestration" / "receipts" / "miiv.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "easyicu_database_publication_receipt_v1",
                "staging_export": "unused",
                "source": source,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "database_order": ["miiv", "eicu"],
        "sources": {},
    }

    pending = launcher._pending_databases(
        manifest=manifest,
        run_root=run_root,
        expected_commit=COMMIT,
    )

    assert pending == ["eicu"]
    assert manifest["sources"]["miiv"]["status"] == "complete"


def test_psutil_absence_has_explicit_strict_and_unsealable_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    with pytest.raises(launcher.ExtractionRunError, match="requires psutil"):
        launcher._load_psutil("strict")

    module, capability = launcher._load_psutil("allow-unsealable")
    assert module is None
    assert capability["release_sealable"] is False
    assert capability["degradation"] == "psutil_unavailable"
