#!/usr/bin/env python3
"""Run one resumable, provenance-locked six-database native-v2 extraction.

The controller never imports EasyICU.  Every database is extracted by a new
Python interpreter whose ``PYTHONPATH`` is pinned to this checkout's ``src``
directory.  A database is published under ``exports/<database>`` only after
all 19 Parquet files and their native manifest have been verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPT_PATH = Path(__file__).resolve()

DATABASE_ORDER = ("miiv", "mimic", "eicu", "aumc", "hirid", "sic")
DATABASE_DIRECTORY_NAMES = {
    "miiv": "mimiciv",
    "mimic": "mimiciii",
    "eicu": "eicu",
    "aumc": "aumc",
    "hirid": "hirid",
    "sic": "sic",
}
MODULE_ORDER = (
    "vitals",
    "demographics",
    "outcome",
    "blood_gas",
    "chemistry",
    "hematology",
    "ventilator",
    "respiratory",
    "vasopressors",
    "medications",
    "neurological",
    "renal",
    "circulatory",
    "other_scores",
    "sepsis_shared",
    "sofa1_score",
    "sofa2_score",
    "sepsis3_sofa1",
    "sepsis3_sofa2",
)
RUN_SCHEMA_VERSION = "easyicu_full6_extraction_run_v2"
NATIVE_SCHEMA_VERSION = "easyicu_native_export_v2"

TIMING_FIELDS = (
    "database",
    "status",
    "elapsed_seconds",
    "module_count",
    "valid_parquet_count",
    "total_rows",
    "total_parquet_bytes",
    "batch_strategy",
    "error",
    "process_exit_code",
    "peak_process_tree_rss_mb",
    "peak_process_tree_pss_mb",
    "initial_batch_size",
    "planned_batch_count",
    "stream_retry_count",
    "attempt_count",
    "resource_monitoring",
    "pss_supported",
    "easyicu_git_commit",
)

_DATABASE_WORKER_MAX = 3
_DATABASE_WORKER_MEMORY_MB = 12 * 1024
_PORTABLE_TOTAL_MEMORY_MB = 18 * 1024
_LOW_AVAILABLE_MEMORY_MB = 16 * 1024
_MEMORY_RESERVE_MB = 2 * 1024
_MEMORY_RESERVE_FRACTION = 0.20
_STREAM_BATCH_MIN = 5_000
_STREAM_BATCH_QUANTUM = 5_000
_STREAM_RETRY_FACTOR = 0.75


class ExtractionRunError(RuntimeError):
    """Raised when the full-run contract cannot be satisfied safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync; unsupported platforms remain portable."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialised = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    _atomic_write_bytes(path, (serialised + "\n").encode("utf-8"))


def _atomic_write_timing_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in TIMING_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionRunError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtractionRunError(f"{label} must contain one JSON object: {path}")
    return value


def _git_command(repository: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ExtractionRunError(
            f"git {' '.join(arguments)} failed in {repository}: "
            f"{detail or result.returncode}"
        )
    return result.stdout.strip()


def _git_identity(repository: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    top_level = Path(
        _git_command(repository, ["rev-parse", "--show-toplevel"])
    ).resolve()
    if top_level != repository.resolve():
        raise ExtractionRunError(
            f"script repository mismatch: expected {repository.resolve()}, got {top_level}"
        )
    commit = _git_command(repository, ["rev-parse", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ExtractionRunError(f"Git did not return a full lowercase SHA: {commit!r}")
    status = _git_command(
        repository,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    return {
        "repository_root": str(repository.resolve()),
        "commit": commit,
        "dirty": bool(status),
        "dirty_status": status.splitlines(),
    }


def _require_clean_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("dirty") is not False:
        paths = identity.get("dirty_status") or []
        preview = "; ".join(str(value) for value in list(paths)[:8])
        raise ExtractionRunError(
            "full extraction requires one clean EasyICU checkout "
            f"(dirty paths: {preview or 'unknown'})"
        )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_output_does_not_dirty_checkout(output_root: Path) -> None:
    if not _is_within(output_root, REPOSITORY_ROOT):
        return
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(output_root)],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ExtractionRunError(
            "output root is inside the EasyICU checkout and is not Git-ignored; "
            "use an external data directory so workers remain clean"
        )


def _parse_key_value(
    values: Sequence[str],
    *,
    option: str,
    allowed_keys: Sequence[str],
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        key, separator, value = str(raw).partition("=")
        if (
            not separator
            or key not in allowed_keys
            or not value.strip()
            or key in parsed
        ):
            raise ExtractionRunError(f"invalid or duplicate {option}: {raw!r}")
        parsed[key] = value.strip()
    return parsed


def _resolve_data_paths(args: argparse.Namespace) -> dict[str, str]:
    paths: dict[str, Path] = {}
    if args.data_root is not None:
        root = Path(args.data_root).expanduser().resolve()
        paths.update(
            {
                database: root / DATABASE_DIRECTORY_NAMES[database]
                for database in DATABASE_ORDER
            }
        )
        # PhysioNet commonly stores MIMIC-III as ``mimiciii/1.4`` while the
        # other databases live directly below ``data_root``. Prefer the
        # versioned directory only when it contains the ICU-stay table and the
        # parent does not; an explicit --data-path below remains authoritative.
        mimic_parent = paths["mimic"]
        mimic_markers = ("icustays.parquet", "ICUSTAYS.csv.gz", "ICUSTAYS.csv")
        parent_is_dataset = any((mimic_parent / name).is_file() for name in mimic_markers)
        versioned = mimic_parent / "1.4"
        versioned_is_dataset = any((versioned / name).is_file() for name in mimic_markers)
        if not parent_is_dataset and versioned_is_dataset:
            paths["mimic"] = versioned
    overrides = _parse_key_value(
        args.data_path,
        option="--data-path",
        allowed_keys=DATABASE_ORDER,
    )
    paths.update(
        {database: Path(value).expanduser().resolve() for database, value in overrides.items()}
    )
    resolved: dict[str, str] = {}
    for database in args.databases:
        path = paths.get(database)
        if path is None:
            raise ExtractionRunError(
                f"no source path for {database}; pass --data-root or "
                f"--data-path {database}=PATH"
            )
        if not path.is_dir():
            raise ExtractionRunError(f"source data directory missing for {database}: {path}")
        resolved[database] = str(path)
    return resolved


def _resolve_batch_overrides(values: Sequence[str]) -> dict[str, int]:
    parsed = _parse_key_value(
        values,
        option="--database-batch-size",
        allowed_keys=DATABASE_ORDER,
    )
    result: dict[str, int] = {}
    for database, raw in parsed.items():
        try:
            value = int(raw)
        except ValueError as exc:
            raise ExtractionRunError(
                f"batch size for {database} must be an integer, got {raw!r}"
            ) from exc
        if value <= 0:
            raise ExtractionRunError(f"batch size for {database} must be positive")
        result[database] = value
    return result


def _load_psutil(resource_policy: str):
    try:
        import psutil
    except ImportError as exc:
        if resource_policy == "strict":
            raise ExtractionRunError(
                "strict resource evidence requires psutil>=5.9; install psutil "
                "before creating a release run"
            ) from exc
        return None, {
            "backend": "unavailable",
            "process_tree_rss_supported": False,
            "process_tree_pss_supported": False,
            "release_sealable": False,
            "degradation": "psutil_unavailable",
        }

    pss_supported = False
    try:
        pss = getattr(psutil.Process(os.getpid()).memory_full_info(), "pss")
        pss_supported = isinstance(pss, int) and pss > 0
    except (psutil.Error, AttributeError, OSError):
        pss_supported = False
    if resource_policy == "strict" and not pss_supported:
        raise ExtractionRunError(
            "this platform/psutil build cannot report real PSS; strict runs stop "
            "before extraction. Use --resource-policy allow-unsealable only when "
            "a non-sealable RSS-only export is explicitly acceptable."
        )
    return psutil, {
        "backend": "psutil_process_tree",
        "psutil_version": getattr(psutil, "__version__", None),
        "process_tree_rss_supported": True,
        "process_tree_pss_supported": pss_supported,
        "release_sealable": pss_supported,
        "degradation": None if pss_supported else "pss_unavailable",
    }


def _read_cgroup_v2_memory_mb() -> tuple[float | None, float | None]:
    if not sys.platform.startswith("linux"):
        return None, None
    relative = "/"
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, candidate = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                relative = candidate
                break
    except (OSError, ValueError):
        pass
    cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
    try:
        raw_limit = (cgroup / "memory.max").read_text(encoding="utf-8").strip()
        if raw_limit == "max":
            return None, None
        limit = int(raw_limit)
        current = int((cgroup / "memory.current").read_text(encoding="utf-8").strip())
        if limit <= 0:
            return None, None
    except (OSError, ValueError):
        return None, None
    return limit / (1024.0**2), max(0, current) / (1024.0**2)


def _detect_effective_memory(psutil_module) -> dict[str, Any]:
    if psutil_module is None:
        return {
            "source": "unavailable",
            "host_total_mb": None,
            "host_available_mb": None,
            "effective_total_mb": None,
            "effective_available_mb": None,
        }
    memory = psutil_module.virtual_memory()
    host_total = float(memory.total) / (1024.0**2)
    host_available = float(memory.available) / (1024.0**2)
    limit, current = _read_cgroup_v2_memory_mb()
    if limit is None:
        effective_total = host_total
        effective_available = min(host_total, host_available)
        source = "host"
    else:
        effective_total = min(host_total, limit)
        effective_available = min(
            host_available,
            effective_total,
            max(0.0, limit - float(current or 0.0)),
        )
        source = "cgroup_v2"
    return {
        "source": source,
        "host_total_mb": round(host_total, 1),
        "host_available_mb": round(host_available, 1),
        "effective_total_mb": round(effective_total, 1),
        "effective_available_mb": round(effective_available, 1),
        "cgroup_limit_mb": None if limit is None else round(limit, 1),
        "cgroup_current_mb": None if current is None else round(current, 1),
    }


def _database_worker_count(
    memory: Mapping[str, Any], requested_maximum: int
) -> int:
    """Choose 1--3 database workers from memory usable at this moment."""

    total = float(memory.get("effective_total_mb") or 0.0)
    available = float(memory.get("effective_available_mb") or 0.0)
    requested = max(1, min(_DATABASE_WORKER_MAX, int(requested_maximum)))
    if (
        total <= 0
        or available <= 0
        or total <= _PORTABLE_TOTAL_MEMORY_MB
        or available < _LOW_AVAILABLE_MEMORY_MB
    ):
        return 1
    reserve = max(_MEMORY_RESERVE_MB, available * _MEMORY_RESERVE_FRACTION)
    usable = max(0.0, available - reserve)
    capacity = max(1, int(usable // _DATABASE_WORKER_MEMORY_MB))
    return min(requested, capacity)


def _assigned_worker_memory_mb(memory: Mapping[str, Any], workers: int) -> float:
    available = float(memory.get("effective_available_mb") or 0.0)
    if available <= 0:
        return 8 * 1024.0
    reserve = max(_MEMORY_RESERVE_MB, available * _MEMORY_RESERVE_FRACTION)
    return max(4 * 1024.0, (available - reserve) / max(1, workers))


def _stream_planning_memory_mb(
    memory: Mapping[str, Any],
    workers: int,
    assigned_memory_mb: float,
) -> float:
    """Return one pre-reserve memory signal for the stream batch planner.

    With one database worker, the core stream planner itself retains the
    25%/2-GiB reserve.  Passing the already-reserved worker allocation would
    reserve twice (an 8-GiB launch was planned as 25k even though the core ran
    40k).  Concurrent waves instead use the per-worker allocation so one
    worker cannot plan against memory assigned to its peer.
    """

    available = float(memory.get("effective_available_mb") or 0.0)
    if int(workers) == 1 and available > 0:
        return available
    return max(0.0, float(assigned_memory_mb))


def _runtime_limits(available_memory_mb: float, cpu_count: int | None = None) -> dict[str, str]:
    available = max(0.0, float(available_memory_mb))
    cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    if available >= 64 * 1024:
        workers, memory_gb, cache_mb = min(8, cpus), 8, 8 * 1024
    elif available >= 32 * 1024:
        workers, memory_gb, cache_mb = min(4, cpus), 4, 6 * 1024
    elif available >= 13 * 1024:
        workers, memory_gb, cache_mb = min(2, cpus), 2, 2 * 1024
    else:
        workers, memory_gb, cache_mb = 1, 1, 256
    return {
        "duckdb_threads": str(workers),
        "duckdb_memory_limit": f"{memory_gb}GB",
        "parallel_max_workers": str(workers),
        "cache_budget_mb": str(cache_mb),
    }


def _nested_workset_budget_mb(available_memory_mb: float) -> int:
    quantum_mb = 512
    budget = int(max(0.0, available_memory_mb) / 3.0 / quantum_mb) * quantum_mb
    return max(quantum_mb, min(8 * 1024, budget))


def _configure_worker_runtime(attempt_root: Path, assigned_memory_mb: float) -> dict[str, Any]:
    runtime_tmp = attempt_root / "runtime_tmp"
    runtime_spill = attempt_root / "runtime_spill"
    runtime_tmp.mkdir(mode=0o700)
    runtime_spill.mkdir(mode=0o700)
    limits = _runtime_limits(assigned_memory_mb)
    os.environ.update(
        {
            "TMPDIR": str(runtime_tmp),
            "TMP": str(runtime_tmp),
            "TEMP": str(runtime_tmp),
            "EASYICU_DUCKDB_TEMP_DIR": str(runtime_spill),
            "EASYICU_DUCKDB_THREADS": limits["duckdb_threads"],
            "EASYICU_DUCKDB_MEMORY_LIMIT": limits["duckdb_memory_limit"],
            "EASYICU_PARALLEL_MAX_WORKERS": limits["parallel_max_workers"],
            "EASYICU_CACHE_BUDGET_MB": limits["cache_budget_mb"],
            "EASYICU_ONESHOT_BUDGET_MB": str(
                _nested_workset_budget_mb(assigned_memory_mb)
            ),
            "EASYICU_OVERRIDE_MEMORY_GB": f"{assigned_memory_mb / 1024.0:.6f}",
        }
    )
    tempfile.tempdir = str(runtime_tmp)
    return {
        "assigned_memory_mb": round(assigned_memory_mb, 1),
        **limits,
        "nested_workset_budget_mb": _nested_workset_budget_mb(assigned_memory_mb),
    }


def _safe_remove_attempt_payload(path: Path, attempt_root: Path) -> None:
    """Remove only bulky, unpublished data below one known attempt root."""

    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir() or path.parent.resolve() != attempt_root.resolve():
        raise ExtractionRunError(f"refusing unsafe attempt cleanup: {path}")
    shutil.rmtree(path)


def _sample_process_tree(psutil_module, process) -> tuple[float, float | None, list[str]]:
    if psutil_module is None or process is None:
        return 0.0, None, []
    errors: list[str] = []
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil_module.Error as exc:
        processes = [process]
        errors.append(type(exc).__name__)
    rss_bytes = 0
    pss_bytes = 0
    pss_observations = 0
    seen: set[int] = set()
    for candidate in processes:
        if candidate.pid in seen:
            continue
        seen.add(candidate.pid)
        try:
            rss_bytes += int(candidate.memory_info().rss)
        except psutil_module.Error as exc:
            errors.append(type(exc).__name__)
            continue
        try:
            value = getattr(candidate.memory_full_info(), "pss")
        except (psutil_module.Error, AttributeError, OSError) as exc:
            errors.append(type(exc).__name__)
        else:
            if isinstance(value, int) and value >= 0:
                pss_bytes += value
                pss_observations += 1
    pss_mb = pss_bytes / (1024.0**2) if pss_observations else None
    return rss_bytes / (1024.0**2), pss_mb, sorted(set(errors))


def _run_monitored_worker(
    *,
    spec_path: Path,
    log_path: Path,
    psutil_module,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    command = [sys.executable, str(SCRIPT_PATH), "--_worker-spec", str(spec_path)]
    started = time.monotonic()
    peak_rss_mb = 0.0
    peak_pss_mb: float | None = None
    monitor_errors: set[str] = set()
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        monitored_process = None
        if psutil_module is not None:
            try:
                monitored_process = psutil_module.Process(process.pid)
            except psutil_module.Error as exc:
                monitor_errors.add(type(exc).__name__)
        try:
            while True:
                rss_mb, pss_mb, errors = _sample_process_tree(
                    psutil_module, monitored_process
                )
                peak_rss_mb = max(peak_rss_mb, rss_mb)
                if pss_mb is not None:
                    peak_pss_mb = max(peak_pss_mb or 0.0, pss_mb)
                monitor_errors.update(errors)
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(sample_interval_seconds)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return {
        "command": command,
        "process_exit_code": int(return_code),
        "elapsed_seconds": max(0.001, time.monotonic() - started),
        "peak_process_tree_rss_mb": round(peak_rss_mb, 3),
        "peak_process_tree_pss_mb": (
            None if peak_pss_mb is None else round(peak_pss_mb, 3)
        ),
        "monitor_errors": sorted(monitor_errors),
    }


def _validate_nonnegative_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ExtractionRunError(f"{label} must be finite and non-negative")
    return float(value)


def _validate_export_package(
    export_root: Path,
    expected_commit: str,
    expected_database: str | None = None,
) -> dict[str, Any]:
    manifest_path = export_root / "_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ExtractionRunError(f"missing regular native manifest: {manifest_path}")
    manifest = _read_json_object(manifest_path, label="native manifest")
    if manifest.get("schema_version") != NATIVE_SCHEMA_VERSION:
        raise ExtractionRunError(
            f"unsupported native schema: {manifest.get('schema_version')!r}"
        )
    if expected_database is not None and manifest.get("database") != expected_database:
        raise ExtractionRunError(
            f"native manifest database mismatch: {manifest.get('database')!r}"
        )
    provenance = manifest.get("runtime_provenance")
    if not isinstance(provenance, dict):
        raise ExtractionRunError("native manifest lacks runtime_provenance")
    if provenance.get("easyicu_git_commit") != expected_commit:
        raise ExtractionRunError("native manifest was produced by a different commit")
    if provenance.get("easyicu_git_dirty") is not False:
        raise ExtractionRunError("native manifest reports a dirty extraction checkout")

    timings = manifest.get("module_timings_seconds")
    module_rss = manifest.get("module_peak_rss_mb")
    module_working_set = manifest.get("module_peak_working_set_mb")
    if not isinstance(timings, dict) or set(timings) != set(MODULE_ORDER):
        raise ExtractionRunError("native manifest must time exactly the 19 modules")
    if not isinstance(module_rss, dict) or set(module_rss) != set(MODULE_ORDER):
        raise ExtractionRunError("native manifest must bind RSS for all 19 modules")
    if not isinstance(module_working_set, dict) or set(module_working_set) != set(
        MODULE_ORDER
    ):
        raise ExtractionRunError(
            "native manifest must bind working-set peaks for all 19 modules"
        )

    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != len(MODULE_ORDER):
        raise ExtractionRunError("native manifest must contain 19 file receipts")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ExtractionRunError("native manifest file receipts must be objects")
    by_module = {entry.get("module"): entry for entry in entries}
    if len(by_module) != len(entries) or set(by_module) != set(MODULE_ORDER):
        raise ExtractionRunError("native manifest module set is not the 19-module contract")
    actual_parquets = {
        path.relative_to(export_root).as_posix()
        for path in export_root.rglob("*.parquet")
        if path.is_file()
    }
    expected_parquets = {f"{module}.parquet" for module in MODULE_ORDER}
    if actual_parquets != expected_parquets:
        raise ExtractionRunError(
            "native Parquet set mismatch: "
            f"missing={sorted(expected_parquets - actual_parquets)}, "
            f"extra={sorted(actual_parquets - expected_parquets)}"
        )

    total_rows = 0
    total_bytes = 0
    module_metrics: dict[str, dict[str, Any]] = {}
    for module in MODULE_ORDER:
        entry = by_module[module]
        parquet = export_root / f"{module}.parquet"
        if parquet.is_symlink() or not parquet.is_file():
            raise ExtractionRunError(f"{module}: Parquet must be a regular file")
        rows = entry.get("rows")
        parquet_bytes = entry.get("parquet_bytes")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ExtractionRunError(f"{module}: invalid manifest row count")
        if (
            isinstance(parquet_bytes, bool)
            or not isinstance(parquet_bytes, int)
            or parquet_bytes <= 0
            or parquet.stat().st_size != parquet_bytes
        ):
            raise ExtractionRunError(f"{module}: invalid or mismatched byte receipt")
        expected_sha = entry.get("parquet_sha256")
        if not isinstance(expected_sha, str) or _sha256(parquet) != expected_sha:
            raise ExtractionRunError(f"{module}: Parquet SHA-256 mismatch")
        elapsed = _validate_nonnegative_number(
            timings[module], label=f"{module}.elapsed"
        )
        peak_rss = _validate_nonnegative_number(
            module_rss[module], label=f"{module}.peak_rss_mb"
        )
        peak_working_set = _validate_nonnegative_number(
            module_working_set[module], label=f"{module}.peak_working_set_mb"
        )
        module_metrics[module] = {
            "elapsed_seconds": elapsed,
            "peak_rss_mb": peak_rss,
            "peak_working_set_mb": peak_working_set,
            "rows": rows,
            "parquet_bytes": parquet_bytes,
            "parquet_sha256": expected_sha,
        }
        total_rows += rows
        total_bytes += parquet_bytes

    sidecar = manifest.get("column_metadata")
    if not isinstance(sidecar, dict):
        raise ExtractionRunError("native manifest lacks typed column metadata receipt")
    sidecar_name = sidecar.get("file")
    sidecar_sha = sidecar.get("sha256")
    if not isinstance(sidecar_name, str) or not isinstance(sidecar_sha, str):
        raise ExtractionRunError("invalid typed column metadata receipt")
    sidecar_path = export_root / sidecar_name
    if (
        sidecar_path.parent.resolve() != export_root.resolve()
        or sidecar_path.is_symlink()
        or not sidecar_path.is_file()
        or _sha256(sidecar_path) != sidecar_sha
    ):
        raise ExtractionRunError("typed column metadata SHA-256 mismatch")

    return {
        "native_manifest_sha256": _sha256(manifest_path),
        "module_count": len(MODULE_ORDER),
        "valid_parquet_count": len(actual_parquets),
        "total_rows": total_rows,
        "total_parquet_bytes": total_bytes,
        "module_metrics": module_metrics,
        "stream_retry_history": list(manifest.get("stream_retry_history") or []),
    }


def _retarget_legacy_module_manifests(
    staging_export: Path, final_export: Path
) -> None:
    """Keep optional legacy side manifests usable after atomic directory rename."""

    old_prefix = str(staging_export)
    new_prefix = str(final_export)

    def replace_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_paths(item) for item in value]
        if isinstance(value, str) and value.startswith(old_prefix):
            return new_prefix + value[len(old_prefix) :]
        return value

    for module in MODULE_ORDER:
        path = staging_export / f"{module}.manifest.json"
        if not path.is_file():
            continue
        payload = _read_json_object(path, label=f"{module} legacy module manifest")
        _atomic_write_json(path, replace_paths(payload))


def _batch_label(
    *, num_stays: int, batch_size: int, adaptive_core: bool, retry_count: int
) -> str:
    batches = max(1, math.ceil(num_stays / max(1, batch_size)))
    if batches == 1:
        return f"one_shot:{num_stays}_stays"
    mode = "adaptive_streamed" if adaptive_core else "planned_streamed"
    return (
        f"{mode}:{batch_size}_stays_x{batches};"
        f"memory_retries={retry_count}"
    )


def _worker_main(spec_path: Path) -> int:
    """Internal clean-interpreter database worker."""

    spec = _read_json_object(spec_path, label="worker specification")
    attempt_root = Path(spec["attempt_root"]).resolve()
    result_path = attempt_root / "worker_result.json"
    plan_path = attempt_root / "worker_plan.json"
    output_root = attempt_root / "export"
    try:
        identity = _git_identity()
        _require_clean_identity(identity)
        if identity["commit"] != spec["easyicu_git_commit"]:
            raise ExtractionRunError("worker checkout changed after controller preflight")
        expected_pythonpath = str(SOURCE_ROOT)
        if os.environ.get("PYTHONPATH") != expected_pythonpath:
            raise ExtractionRunError(
                f"worker PYTHONPATH is not pinned to {expected_pythonpath}"
            )
        runtime = _configure_worker_runtime(
            attempt_root, float(spec["assigned_memory_mb"])
        )
        if output_root.exists() or output_root.is_symlink():
            raise ExtractionRunError(f"worker output staging root already exists: {output_root}")

        import easyicu
        from easyicu.api import extract_database
        from easyicu.api.extraction import (
            EXTRACT_MODULE_ORDER,
            _get_all_patient_ids,
            _resolve_stream_batch_size,
        )

        imported_package = Path(easyicu.__file__).resolve().parent
        expected_package = (SOURCE_ROOT / "easyicu").resolve()
        if imported_package != expected_package:
            raise ExtractionRunError(
                f"worker imported EasyICU from {imported_package}, expected {expected_package}"
            )
        if tuple(EXTRACT_MODULE_ORDER) != MODULE_ORDER:
            raise ExtractionRunError("worker 19-module catalog differs from launcher contract")

        database = str(spec["database"])
        data_path = str(spec["data_path"])
        patient_ids, id_column = _get_all_patient_ids(data_path, database, None)
        if not patient_ids:
            raise ExtractionRunError(f"no ICU stays found for {database}")
        num_stays = len(patient_ids)
        requested_batch_size = spec.get("requested_batch_size")
        planning_memory_mb = float(
            spec.get("planning_memory_mb", spec["assigned_memory_mb"])
        )
        planned_batch_size = _resolve_stream_batch_size(
            database,
            num_stays,
            requested_batch_size,
            available_memory_mb=planning_memory_mb,
        )
        adaptive_core = bool(spec.get("adaptive_core")) and requested_batch_size is None
        plan = {
            "database": database,
            "num_stays": num_stays,
            "id_column": id_column,
            "requested_batch_size": requested_batch_size,
            "planned_initial_batch_size": planned_batch_size,
            "planned_batch_count": math.ceil(num_stays / planned_batch_size),
            "adaptive_core": adaptive_core,
            "assigned_memory_mb": round(float(spec["assigned_memory_mb"]), 1),
            "planning_memory_mb": round(planning_memory_mb, 1),
            "runtime_limits": runtime,
            "easyicu_git_commit": identity["commit"],
        }
        _atomic_write_json(plan_path, plan)

        extraction = extract_database(
            database,
            data_path=data_path,
            output_dir=output_root,
            modules=list(MODULE_ORDER),
            patient_ids={id_column: patient_ids},
            batch_size=planned_batch_size,
            native_export_v2=True,
            stream_output_batches=True,
            verbose=True,
            adaptive_stream_batches=adaptive_core,
        )
        errors = {
            module: list((extraction["modules"].get(module) or {}).get("errors") or [])
            for module in MODULE_ORDER
            if (extraction["modules"].get(module) or {}).get("errors")
        }
        if errors:
            raise ExtractionRunError(f"module extraction errors: {errors}")
        receipt = _validate_export_package(
            output_root, identity["commit"], database
        )
        actual_batch_size = int(extraction.get("batch_size") or planned_batch_size)
        retries = list(extraction.get("stream_retry_history") or [])
        payload = {
            "status": "complete",
            "database": database,
            "num_stays": num_stays,
            "batch_strategy": {
                "label": _batch_label(
                    num_stays=num_stays,
                    batch_size=actual_batch_size,
                    adaptive_core=adaptive_core,
                    retry_count=len(retries),
                ),
                "adaptive_core": adaptive_core,
                "initial_batch_size": actual_batch_size,
                "planned_batch_count": math.ceil(num_stays / actual_batch_size),
                "stream_retry_history": retries,
            },
            "runtime_limits": runtime,
            "package_receipt": receipt,
        }
        _atomic_write_json(result_path, payload)
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "database": spec.get("database"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            _atomic_write_json(result_path, payload)
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        return 1


def _next_retry_batch_size(current_batch_size: int) -> int:
    current = max(_STREAM_BATCH_MIN, int(current_batch_size))
    proposed = int(current * _STREAM_RETRY_FACTOR)
    proposed = (proposed // _STREAM_BATCH_QUANTUM) * _STREAM_BATCH_QUANTUM
    if proposed >= current:
        proposed = current - _STREAM_BATCH_QUANTUM
    return max(_STREAM_BATCH_MIN, proposed)


def _looks_like_memory_failure(exit_code: int, error: str) -> bool:
    if exit_code in {-9, 9, 137, -1073740791, 3221226505}:
        return True
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "memoryerror",
            "out of memory",
            "cannot allocate memory",
            "bad_alloc",
            "oom",
            "killed",
            "streamed module export exhausted memory",
        )
    )


def _attempt_number(run_root: Path, database: str) -> int:
    database_root = run_root / ".orchestration" / "attempts" / database
    if not database_root.exists():
        return 1
    numbers = []
    for path in database_root.iterdir():
        match = re.fullmatch(r"attempt-(\d+)", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _read_optional_worker_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _max_metric(attempts: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(attempt[field])
        for attempt in attempts
        if isinstance(attempt.get(field), (int, float))
        and not isinstance(attempt.get(field), bool)
    ]
    return max(values) if values else None


def _execute_database(
    *,
    database: str,
    run_root: Path,
    data_path: str,
    git_commit: str,
    assigned_memory_mb: float,
    adaptive_core: bool,
    requested_batch_size: int | None,
    max_memory_retries: int,
    sample_interval_seconds: float,
    psutil_module,
    monitoring: Mapping[str, Any],
    prior_source: Mapping[str, Any] | None,
    planning_memory_mb: float | None = None,
) -> dict[str, Any]:
    final_export = run_root / "exports" / database
    if final_export.exists() or final_export.is_symlink():
        raise ExtractionRunError(
            f"refusing to overwrite existing database export: {final_export}"
        )
    attempts = list((prior_source or {}).get("attempts") or [])
    next_batch_size = requested_batch_size
    retry_index = 0
    final_error = "worker did not start"
    final_exit_code = 1
    while True:
        attempt_number = _attempt_number(run_root, database)
        attempt_root = (
            run_root
            / ".orchestration"
            / "attempts"
            / database
            / f"attempt-{attempt_number:02d}"
        )
        attempt_root.mkdir(mode=0o700, parents=True)
        spec = {
            "schema_version": "easyicu_database_worker_spec_v1",
            "database": database,
            "data_path": data_path,
            "attempt_root": str(attempt_root),
            "easyicu_git_commit": git_commit,
            "assigned_memory_mb": round(assigned_memory_mb, 1),
            "planning_memory_mb": round(
                assigned_memory_mb
                if planning_memory_mb is None
                else planning_memory_mb,
                1,
            ),
            "adaptive_core": adaptive_core and retry_index == 0,
            "requested_batch_size": next_batch_size,
        }
        spec_path = attempt_root / "worker_spec.json"
        _atomic_write_json(spec_path, spec)
        monitor_result = _run_monitored_worker(
            spec_path=spec_path,
            log_path=attempt_root / "worker.log",
            psutil_module=psutil_module,
            sample_interval_seconds=sample_interval_seconds,
        )
        worker_result = _read_optional_worker_json(attempt_root / "worker_result.json")
        worker_plan = _read_optional_worker_json(attempt_root / "worker_plan.json")
        error = str(worker_result.get("error") or "")
        if not error and monitor_result["process_exit_code"] != 0:
            error = "worker exited without a structured error"
        attempt_record = {
            "attempt": attempt_number,
            "status": worker_result.get("status", "failed"),
            "started_batch_size": worker_plan.get("planned_initial_batch_size"),
            "planned_batch_count": worker_plan.get("planned_batch_count"),
            "adaptive_core": worker_plan.get("adaptive_core"),
            "assigned_memory_mb": round(assigned_memory_mb, 1),
            "planning_memory_mb": worker_plan.get("planning_memory_mb"),
            "elapsed_seconds": round(monitor_result["elapsed_seconds"], 3),
            "process_exit_code": monitor_result["process_exit_code"],
            "peak_process_tree_rss_mb": monitor_result[
                "peak_process_tree_rss_mb"
            ],
            "peak_process_tree_pss_mb": monitor_result[
                "peak_process_tree_pss_mb"
            ],
            "monitor_errors": monitor_result["monitor_errors"],
            "error": error,
            "log": str((attempt_root / "worker.log").relative_to(run_root)),
        }
        attempts.append(attempt_record)
        final_error = error
        final_exit_code = int(monitor_result["process_exit_code"])

        strict_monitor_ok = (
            monitoring.get("release_sealable") is not True
            or (
                monitor_result["peak_process_tree_rss_mb"] > 0
                and isinstance(
                    monitor_result["peak_process_tree_pss_mb"], (int, float)
                )
                and monitor_result["peak_process_tree_pss_mb"] > 0
            )
        )
        success = (
            final_exit_code == 0
            and worker_result.get("status") == "complete"
            and strict_monitor_ok
        )
        if success:
            staging_export = attempt_root / "export"
            _retarget_legacy_module_manifests(staging_export, final_export)
            package_receipt = _validate_export_package(
                staging_export, git_commit, database
            )
            worker_receipt = worker_result.get("package_receipt")
            if worker_receipt != package_receipt:
                raise ExtractionRunError(
                    f"{database}: controller and worker package receipts disagree"
                )
            strategy = worker_result.get("batch_strategy") or {}
            elapsed_seconds = sum(float(item["elapsed_seconds"]) for item in attempts)
            source = {
                "status": "complete",
                "database": database,
                "completed_at": _utc_now(),
                "easyicu_git_commit": git_commit,
                "data_path": data_path,
                "elapsed_seconds": round(max(0.001, elapsed_seconds), 3),
                "successful_attempt_elapsed_seconds": round(
                    monitor_result["elapsed_seconds"], 3
                ),
                "process_exit_code": 0,
                "peak_process_tree_rss_mb": _max_metric(
                    attempts, "peak_process_tree_rss_mb"
                ),
                "peak_process_tree_pss_mb": _max_metric(
                    attempts, "peak_process_tree_pss_mb"
                ),
                "batch_strategy": strategy.get("label"),
                "initial_batch_size": strategy.get("initial_batch_size"),
                "planned_batch_count": strategy.get("planned_batch_count"),
                "stream_retry_count": len(strategy.get("stream_retry_history") or []),
                "attempt_count": len(attempts),
                "attempts": attempts,
                **package_receipt,
            }
            receipt_path = run_root / ".orchestration" / "receipts" / f"{database}.json"
            _atomic_write_json(
                receipt_path,
                {
                    "schema_version": "easyicu_database_publication_receipt_v1",
                    "staging_export": str(staging_export),
                    "source": source,
                },
            )
            if final_export.exists() or final_export.is_symlink():
                raise ExtractionRunError(
                    f"refusing to overwrite existing database export: {final_export}"
                )
            os.replace(staging_export, final_export)
            _fsync_directory(final_export.parent)
            _safe_remove_attempt_payload(attempt_root / "runtime_tmp", attempt_root)
            _safe_remove_attempt_payload(attempt_root / "runtime_spill", attempt_root)
            return source

        if monitoring.get("release_sealable") and not strict_monitor_ok:
            final_error = "strict process-tree RSS/PSS evidence was not captured"
            retryable = False
        else:
            retryable = _looks_like_memory_failure(final_exit_code, final_error)
        planned = worker_plan.get("planned_initial_batch_size")
        can_downbatch = (
            retryable
            and retry_index < max_memory_retries
            and isinstance(planned, int)
            and planned > _STREAM_BATCH_MIN
        )
        _safe_remove_attempt_payload(attempt_root / "export", attempt_root)
        _safe_remove_attempt_payload(attempt_root / "runtime_tmp", attempt_root)
        _safe_remove_attempt_payload(attempt_root / "runtime_spill", attempt_root)
        if not can_downbatch:
            break
        next_batch_size = _next_retry_batch_size(planned)
        adaptive_core = False
        retry_index += 1

    return {
        "status": "failed",
        "database": database,
        "failed_at": _utc_now(),
        "easyicu_git_commit": git_commit,
        "data_path": data_path,
        "elapsed_seconds": round(
            max(0.001, sum(float(item["elapsed_seconds"]) for item in attempts)),
            3,
        ),
        "process_exit_code": final_exit_code,
        "peak_process_tree_rss_mb": _max_metric(attempts, "peak_process_tree_rss_mb"),
        "peak_process_tree_pss_mb": _max_metric(attempts, "peak_process_tree_pss_mb"),
        "batch_strategy": "failed",
        "initial_batch_size": attempts[-1].get("started_batch_size") if attempts else None,
        "planned_batch_count": attempts[-1].get("planned_batch_count") if attempts else None,
        "stream_retry_count": 0,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "error": final_error,
    }


def _timing_row(
    database: str,
    source: Mapping[str, Any] | None,
    *,
    monitoring: Mapping[str, Any],
) -> dict[str, Any]:
    if source is None:
        return {"database": database, "status": "pending"}
    complete = source.get("status") == "complete"
    if complete and monitoring.get("release_sealable"):
        status = "complete"
    elif complete:
        status = "complete_unsealable"
    else:
        status = "failed"
    return {
        "database": database,
        "status": status,
        "elapsed_seconds": source.get("elapsed_seconds", ""),
        "module_count": source.get("module_count", 0),
        "valid_parquet_count": source.get("valid_parquet_count", 0),
        "total_rows": source.get("total_rows", 0),
        "total_parquet_bytes": source.get("total_parquet_bytes", 0),
        "batch_strategy": source.get("batch_strategy", ""),
        "error": "" if complete else source.get("error", "unknown failure"),
        "process_exit_code": source.get("process_exit_code", ""),
        "peak_process_tree_rss_mb": source.get("peak_process_tree_rss_mb", ""),
        "peak_process_tree_pss_mb": source.get("peak_process_tree_pss_mb", ""),
        "initial_batch_size": source.get("initial_batch_size", ""),
        "planned_batch_count": source.get("planned_batch_count", ""),
        "stream_retry_count": source.get("stream_retry_count", ""),
        "attempt_count": source.get("attempt_count", ""),
        "resource_monitoring": monitoring.get("backend", "unavailable"),
        "pss_supported": str(
            bool(monitoring.get("process_tree_pss_supported"))
        ).lower(),
        "easyicu_git_commit": source.get("easyicu_git_commit", ""),
    }


def _persist_run_state(run_root: Path, manifest: Mapping[str, Any]) -> None:
    _atomic_write_json(run_root / "run_manifest.json", manifest)
    sources = manifest.get("sources") or {}
    rows = [
        _timing_row(
            database,
            sources.get(database),
            monitoring=manifest["resource_monitoring"],
        )
        for database in manifest["database_order"]
    ]
    _atomic_write_timing_csv(run_root / "database_extraction_timing.csv", rows)


def _new_run_manifest(
    *,
    databases: Sequence[str],
    data_paths: Mapping[str, str],
    identity: Mapping[str, Any],
    monitoring: Mapping[str, Any],
    memory: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "database_order": list(databases),
        "module_order": list(MODULE_ORDER),
        "output_layout": "exports/{database}/{module}.parquet",
        "source_checkout": {
            "repository_root": identity["repository_root"],
            "easyicu_git_commit": identity["commit"],
            "easyicu_git_dirty": False,
            "pythonpath": str(SOURCE_ROOT),
            "launcher": str(SCRIPT_PATH.relative_to(REPOSITORY_ROOT)),
            "launcher_sha256": _sha256(SCRIPT_PATH),
        },
        "data_paths": dict(data_paths),
        "resource_policy": args.resource_policy,
        "resource_monitoring": dict(monitoring),
        "launch_memory": dict(memory),
        "scheduler": {
            "maximum_database_workers": args.max_database_workers,
            "hard_database_worker_cap": _DATABASE_WORKER_MAX,
            "eicu_exclusive": True,
            "portable_and_low_available_memory_serial": True,
            "memory_retry_limit": args.max_memory_retries,
            "sample_interval_seconds": args.sample_interval,
        },
        "sources": {},
    }


def _load_resume_manifest(
    *,
    run_root: Path,
    databases: Sequence[str],
    data_paths: Mapping[str, str],
    identity: Mapping[str, Any],
    resource_policy: str,
) -> dict[str, Any]:
    manifest = _read_json_object(run_root / "run_manifest.json", label="run manifest")
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ExtractionRunError("resume root has an unsupported run manifest")
    if manifest.get("database_order") != list(databases):
        raise ExtractionRunError("resume database list/order differs from the original run")
    if manifest.get("module_order") != list(MODULE_ORDER):
        raise ExtractionRunError("resume run uses a different 19-module contract")
    checkout = manifest.get("source_checkout") or {}
    if (
        checkout.get("easyicu_git_commit") != identity["commit"]
        or checkout.get("easyicu_git_dirty") is not False
    ):
        raise ExtractionRunError("resume requires the exact original clean EasyICU commit")
    if manifest.get("data_paths") != dict(data_paths):
        raise ExtractionRunError("resume source data paths differ from the original run")
    if manifest.get("resource_policy") != resource_policy:
        raise ExtractionRunError("resume resource policy differs from the original run")
    return manifest


def _recover_published_source(
    *, run_root: Path, database: str, expected_commit: str
) -> dict[str, Any] | None:
    final_export = run_root / "exports" / database
    receipt_path = run_root / ".orchestration" / "receipts" / f"{database}.json"
    if not final_export.exists() and not receipt_path.is_file():
        return None
    if final_export.exists() and not receipt_path.is_file():
        raise ExtractionRunError(
            f"{database}: existing export has no atomic publication receipt; "
            "refusing overwrite"
        )
    receipt = _read_json_object(receipt_path, label=f"{database} publication receipt")
    source = receipt.get("source")
    if not isinstance(source, dict) or source.get("status") != "complete":
        raise ExtractionRunError(f"{database}: invalid publication receipt")
    if final_export.exists():
        package = _validate_export_package(
            final_export, expected_commit, database
        )
        for key in (
            "native_manifest_sha256",
            "module_count",
            "valid_parquet_count",
            "total_rows",
            "total_parquet_bytes",
            "module_metrics",
        ):
            if source.get(key) != package.get(key):
                raise ExtractionRunError(
                    f"{database}: published export disagrees with its receipt ({key})"
                )
        return source
    staging = Path(str(receipt.get("staging_export") or ""))
    expected_parent = run_root / ".orchestration" / "attempts" / database
    if not _is_within(staging, expected_parent):
        raise ExtractionRunError(f"{database}: unsafe staging path in receipt")
    package = _validate_export_package(staging, expected_commit, database)
    if source.get("native_manifest_sha256") != package["native_manifest_sha256"]:
        raise ExtractionRunError(f"{database}: staging export disagrees with receipt")
    os.replace(staging, final_export)
    _fsync_directory(final_export.parent)
    return source


def _pending_databases(
    *, manifest: dict[str, Any], run_root: Path, expected_commit: str
) -> list[str]:
    sources = manifest.setdefault("sources", {})
    pending: list[str] = []
    for database in manifest["database_order"]:
        recovered = _recover_published_source(
            run_root=run_root,
            database=database,
            expected_commit=expected_commit,
        )
        if recovered is not None:
            sources[database] = recovered
            continue
        final_export = run_root / "exports" / database
        if final_export.exists() or final_export.is_symlink():
            raise ExtractionRunError(
                f"{database}: existing export cannot be proven complete; refusing overwrite"
            )
        pending.append(database)
    return pending


def _run_non_eicu_segment(
    *,
    segment: Sequence[str],
    args: argparse.Namespace,
    run_root: Path,
    manifest: dict[str, Any],
    data_paths: Mapping[str, str],
    batch_overrides: Mapping[str, int],
    psutil_module,
) -> None:
    remaining = list(segment)
    while remaining:
        memory = _detect_effective_memory(psutil_module)
        worker_count = min(
            len(remaining),
            _database_worker_count(memory, args.max_database_workers),
        )
        wave = remaining[:worker_count]
        remaining = remaining[worker_count:]
        assigned_memory_mb = _assigned_worker_memory_mb(memory, worker_count)
        planning_memory_mb = _stream_planning_memory_mb(
            memory,
            worker_count,
            assigned_memory_mb,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _execute_database,
                    database=database,
                    run_root=run_root,
                    data_path=data_paths[database],
                    git_commit=manifest["source_checkout"]["easyicu_git_commit"],
                    assigned_memory_mb=assigned_memory_mb,
                    adaptive_core=worker_count == 1,
                    requested_batch_size=batch_overrides.get(database),
                    max_memory_retries=args.max_memory_retries,
                    sample_interval_seconds=args.sample_interval,
                    psutil_module=psutil_module,
                    monitoring=manifest["resource_monitoring"],
                    prior_source=manifest["sources"].get(database),
                    planning_memory_mb=planning_memory_mb,
                ): database
                for database in wave
            }
            for future in as_completed(futures):
                database = futures[future]
                try:
                    source = future.result()
                except Exception as exc:
                    source = {
                        "status": "failed",
                        "database": database,
                        "failed_at": _utc_now(),
                        "easyicu_git_commit": manifest["source_checkout"][
                            "easyicu_git_commit"
                        ],
                        "data_path": data_paths[database],
                        "elapsed_seconds": 0.001,
                        "process_exit_code": 1,
                        "attempt_count": 0,
                        "attempts": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                manifest["sources"][database] = source
                manifest["updated_at"] = _utc_now()
                _persist_run_state(run_root, manifest)


def _run_pending(
    *,
    pending: Sequence[str],
    args: argparse.Namespace,
    run_root: Path,
    manifest: dict[str, Any],
    data_paths: Mapping[str, str],
    batch_overrides: Mapping[str, int],
    psutil_module,
) -> None:
    segment: list[str] = []
    for database in [*pending, None]:
        if database not in {"eicu", None}:
            segment.append(str(database))
            continue
        if segment:
            _run_non_eicu_segment(
                segment=segment,
                args=args,
                run_root=run_root,
                manifest=manifest,
                data_paths=data_paths,
                batch_overrides=batch_overrides,
                psutil_module=psutil_module,
            )
            segment = []
        if database == "eicu":
            memory = _detect_effective_memory(psutil_module)
            assigned_memory_mb = _assigned_worker_memory_mb(memory, 1)
            try:
                source = _execute_database(
                    database="eicu",
                    run_root=run_root,
                    data_path=data_paths["eicu"],
                    git_commit=manifest["source_checkout"]["easyicu_git_commit"],
                    assigned_memory_mb=assigned_memory_mb,
                    adaptive_core=True,
                    requested_batch_size=batch_overrides.get("eicu"),
                    max_memory_retries=args.max_memory_retries,
                    sample_interval_seconds=args.sample_interval,
                    psutil_module=psutil_module,
                    monitoring=manifest["resource_monitoring"],
                    prior_source=manifest["sources"].get("eicu"),
                    planning_memory_mb=_stream_planning_memory_mb(
                        memory,
                        1,
                        assigned_memory_mb,
                    ),
                )
            except Exception as exc:
                source = {
                    "status": "failed",
                    "database": "eicu",
                    "failed_at": _utc_now(),
                    "easyicu_git_commit": manifest["source_checkout"][
                        "easyicu_git_commit"
                    ],
                    "data_path": data_paths["eicu"],
                    "elapsed_seconds": 0.001,
                    "process_exit_code": 1,
                    "attempt_count": 0,
                    "attempts": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            manifest["sources"]["eicu"] = source
            manifest["updated_at"] = _utc_now()
            _persist_run_state(run_root, manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    if not 1 <= args.max_database_workers <= _DATABASE_WORKER_MAX:
        raise ExtractionRunError("--max-database-workers must be between 1 and 3")
    if not 0 <= args.max_memory_retries <= 3:
        raise ExtractionRunError("--max-memory-retries must be between 0 and 3")
    if not 0.02 <= args.sample_interval <= 5.0:
        raise ExtractionRunError("--sample-interval must be between 0.02 and 5 seconds")

    identity = _git_identity()
    _require_clean_identity(identity)
    psutil_module, monitoring = _load_psutil(args.resource_policy)
    memory = _detect_effective_memory(psutil_module)
    data_paths = _resolve_data_paths(args)
    batch_overrides = _resolve_batch_overrides(args.database_batch_size)
    unused_overrides = set(batch_overrides) - set(args.databases)
    if unused_overrides:
        raise ExtractionRunError(
            f"batch overrides name databases outside this run: {sorted(unused_overrides)}"
        )

    run_root = Path(args.output_root).expanduser().resolve()
    _require_output_does_not_dirty_checkout(run_root)
    if args.resume:
        if run_root.is_symlink() or not run_root.is_dir():
            raise ExtractionRunError(f"resume root must be an existing directory: {run_root}")
        manifest = _load_resume_manifest(
            run_root=run_root,
            databases=args.databases,
            data_paths=data_paths,
            identity=identity,
            resource_policy=args.resource_policy,
        )
        manifest["resource_monitoring"] = monitoring
    else:
        if run_root.exists() or run_root.is_symlink():
            raise ExtractionRunError(
                f"output root must be new and non-symlink: {run_root}; "
                "use --resume only for this launcher's incomplete run root"
            )
        run_root.mkdir(mode=0o700, parents=True)
        (run_root / "exports").mkdir(mode=0o700)
        (run_root / ".orchestration" / "attempts").mkdir(mode=0o700, parents=True)
        (run_root / ".orchestration" / "receipts").mkdir(mode=0o700, parents=True)
        manifest = _new_run_manifest(
            databases=args.databases,
            data_paths=data_paths,
            identity=identity,
            monitoring=monitoring,
            memory=memory,
            args=args,
        )

    pending = _pending_databases(
        manifest=manifest,
        run_root=run_root,
        expected_commit=identity["commit"],
    )
    manifest["status"] = "running"
    manifest["updated_at"] = _utc_now()
    _persist_run_state(run_root, manifest)
    if pending:
        _run_pending(
            pending=pending,
            args=args,
            run_root=run_root,
            manifest=manifest,
            data_paths=data_paths,
            batch_overrides=batch_overrides,
            psutil_module=psutil_module,
        )

    failures = [
        database
        for database in args.databases
        if (manifest["sources"].get(database) or {}).get("status") != "complete"
    ]
    if failures:
        manifest["status"] = "failed"
        manifest["failed_databases"] = failures
    elif monitoring.get("release_sealable"):
        manifest["status"] = "complete"
        manifest.pop("failed_databases", None)
    else:
        manifest["status"] = "complete_unsealable"
        manifest.pop("failed_databases", None)
    manifest["completed_at"] = _utc_now()
    manifest["updated_at"] = manifest["completed_at"]
    _persist_run_state(run_root, manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="new run root, or this launcher's existing root together with --resume",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify completed databases and run only missing/failed databases",
    )
    parser.add_argument(
        "--databases",
        nargs="+",
        choices=DATABASE_ORDER,
        default=list(DATABASE_ORDER),
        help="database order (default: all six)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="root containing mimiciv, mimiciii, eicu, aumc, hirid, and sic",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        metavar="DATABASE=PATH",
        help="override one source path; may be repeated",
    )
    parser.add_argument(
        "--database-batch-size",
        action="append",
        default=[],
        metavar="DATABASE=STAYS",
        help="expert per-database override; automatic memory planning is the default",
    )
    parser.add_argument(
        "--max-database-workers",
        type=int,
        default=3,
        help="upper bound only; memory planner may select fewer (maximum: 3)",
    )
    parser.add_argument(
        "--max-memory-retries",
        type=int,
        default=2,
        help="whole-database retries after a process-level OOM (default: 2)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.10,
        help="process-tree RSS/PSS sampling interval in seconds (default: 0.10)",
    )
    parser.add_argument(
        "--resource-policy",
        choices=("strict", "allow-unsealable"),
        default="strict",
        help="strict requires real psutil process-tree RSS and PSS evidence",
    )
    args = parser.parse_args(argv)
    if args._worker_spec is None and args.output_root is None:
        parser.error("--output-root is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args._worker_spec is not None:
        return _worker_main(args._worker_spec)
    try:
        manifest = run(args)
    except (ExtractionRunError, OSError, ValueError) as exc:
        print(f"full native-v2 extraction failed: {exc}", file=sys.stderr)
        return 1
    summary = {
        "status": manifest["status"],
        "output_root": str(args.output_root),
        "complete_databases": [
            database
            for database, source in manifest["sources"].items()
            if source.get("status") == "complete"
        ],
        "failed_databases": manifest.get("failed_databases", []),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if manifest["status"] in {"complete", "complete_unsealable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
