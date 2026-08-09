"""Conversion, export, registration, and orchestration for demo sources.

Secure cache/archive work is supplied through an immutable operation contract.
This owner composes the stages, translates every non-cancellation failure into
one phase-attributable diagnostic, and preserves lower-layer causes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store
from easyicu.webserver.demo_source_contracts import (
    DemoSourceCancelled,
    DemoSourceError,
    DemoSourcePaths,
    DemoSourceSpec,
    check_cancelled,
    is_cancel_requested,
)
from easyicu.webserver.demo_source_storage import (
    export_ready,
    parquet_ready,
    read_marker,
    write_marker,
)

DownloadOperation = Callable[
    [DemoSourceSpec, DemoSourcePaths, Any],
    tuple[str, bool],
]
ExtractOperation = Callable[
    [DemoSourceSpec, DemoSourcePaths, str, Any],
    bool,
]
ConvertOperation = Callable[
    [DemoSourceSpec, DemoSourcePaths, str, Any],
    tuple[dict[str, int], bool],
]
ExportOperation = Callable[
    [DemoSourceSpec, DemoSourcePaths, str, Any],
    tuple[dict[str, Any], bool],
]
RegisterOperation = Callable[
    [DemoSourceSpec, DemoSourcePaths, Any],
    dict[str, Any],
]


@dataclass(frozen=True)
class PrepareOperations:
    """Explicit dependency contract for the five preparation stages."""

    download: DownloadOperation
    extract: ExtractOperation
    convert: ConvertOperation
    export: ExportOperation
    register: RegisterOperation


class ExportJobProxy:
    """Add an export phase and remove local paths from nested runner events."""

    def __init__(self, job: Any) -> None:
        self._job = job

    @property
    def cancel_requested(self) -> bool:
        return is_cancel_requested(self._job)

    def emit(self, event: Dict[str, Any]) -> bool:
        payload = dict(event)
        nested_phase = payload.pop("phase", None)
        for field in ("path", "data_path", "out_dir"):
            payload.pop(field, None)
        payload["phase"] = "export"
        if nested_phase and "stage" not in payload:
            payload["stage"] = str(nested_phase)
        else:
            payload.setdefault("stage", "module")
        return bool(self._job.emit(payload))


def convert_dataset(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    archive_sha256: str,
    job: Any,
) -> tuple[dict[str, int], bool]:
    """Convert extracted official tables through the canonical converter."""

    if parquet_ready(paths, source):
        marker = read_marker(paths.converted_marker, source) or {}
        summary = dict(marker.get("conversion") or {})
        job.emit(
            {
                "type": "progress",
                "phase": "convert",
                "stage": "reused",
                **summary,
            }
        )
        return {
            "converted": int(summary.get("converted") or 0),
            "failed": int(summary.get("failed") or 0),
            "skipped": int(summary.get("skipped") or 0),
            "total_files": int(summary.get("total_files") or 0),
        }, True

    from easyicu.io.data_converter import ConversionStatus, DataConverter

    check_cancelled(job, "convert")
    converter = DataConverter(
        data_path=paths.raw,
        database=source.database,
        verbose=False,
    )
    counts = {"converted": 0, "failed": 0, "skipped": 0}
    job.emit(
        {
            "type": "progress",
            "phase": "convert",
            "stage": "starting",
            "database": source.database,
        }
    )

    def progress(info: Dict[str, Any]) -> None:
        status = info.get("status")
        if status == ConversionStatus.FAILED:
            counts["failed"] += 1
        elif status == ConversionStatus.SKIPPED:
            counts["skipped"] += 1
        else:
            counts["converted"] += 1
        result = info.get("result") or {}
        job.emit(
            {
                "type": "progress",
                "phase": "convert",
                "stage": "table",
                "current": info.get("current"),
                "total": info.get("total"),
                "file": Path(str(info.get("file") or "")).name,
                "status": status,
                "rows": result.get("row_count"),
                "shards": result.get("shards"),
                "counts": dict(counts),
            }
        )

    results = converter.convert_all(force=False, progress_callback=progress)
    check_cancelled(job, "convert")
    failed = sum(
        1
        for result in results.values()
        if isinstance(result, dict) and result.get("status") == ConversionStatus.FAILED
    )
    if failed or counts["failed"]:
        raise DemoSourceError(
            f"Demo conversion failed for {max(failed, counts['failed'])} table(s)"
        )
    if not any(paths.raw.rglob("*.parquet")):
        raise DemoSourceError("Demo archive contained no convertible ICU tables")
    summary = {
        "converted": counts["converted"],
        "failed": 0,
        "skipped": counts["skipped"],
        "total_files": len(results),
    }
    write_marker(
        paths.converted_marker,
        source,
        archive_sha256=archive_sha256,
        conversion=summary,
    )
    job.emit(
        {
            "type": "progress",
            "phase": "convert",
            "stage": "complete",
            **summary,
        }
    )
    return summary, False


def export_dataset(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    archive_sha256: str,
    job: Any,
) -> tuple[dict[str, Any], bool]:
    """Run the canonical all-module export and persist its readiness marker."""

    if export_ready(paths, source):
        marker = read_marker(paths.prepared_marker, source) or {}
        summary = dict(marker.get("export") or {})
        job.emit(
            {
                "type": "progress",
                "phase": "export",
                "stage": "reused",
                **summary,
            }
        )
        return summary, True

    check_cancelled(job, "export")
    paths.export.mkdir(parents=True, exist_ok=True, mode=0o700)
    export_runner = dataio.make_export_runner(
        data_path=str(paths.raw),
        database=source.database,
        modules=None,
        concepts=None,
        export_format="parquet",
        merge=False,
        out_dir=str(paths.export),
        create_run_subdir=False,
        max_patients=None,
        cohort=None,
        include_feature_definitions=True,
    )
    result = export_runner(ExportJobProxy(job)) or {}
    check_cancelled(job, "export")
    if result.get("cancelled_at") or not result.get("manifest"):
        raise DemoSourceError("All-module demo export did not produce a manifest")
    summary = {
        "file_count": int(result.get("file_count") or 0),
        "total_rows": int(result.get("total_rows") or 0),
        "manifest": str(result.get("manifest") or "_manifest.json"),
        "format": "parquet",
        "scope": "all_modules",
    }
    write_marker(
        paths.prepared_marker,
        source,
        archive_sha256=archive_sha256,
        export=summary,
    )
    job.emit(
        {
            "type": "progress",
            "phase": "export",
            "stage": "complete",
            **summary,
        }
    )
    return summary, False


def register_export(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    job: Any,
) -> dict[str, Any]:
    """Register the prepared export as an active local cross-database source."""

    check_cancelled(job, "register")
    job.emit({"type": "progress", "phase": "register", "stage": "starting"})
    registry = source_store.register_source(
        str(paths.export),
        label=f"{source.title} v{source.version}",
        active=True,
        crossdb=True,
    )
    if not registry.get("ok"):
        raise DemoSourceError("Prepared demo export could not be registered")
    result = {
        "ok": True,
        "active": True,
        "source_count": len(registry.get("sources") or []),
    }
    job.emit(
        {
            "type": "progress",
            "phase": "register",
            "stage": "complete",
            **result,
        }
    )
    return result


_PREPARE_LOCKS: dict[str, threading.Lock] = {}
_PREPARE_LOCKS_GUARD = threading.Lock()


def prepare_lock(source_id: str) -> threading.Lock:
    """Return the process-local idempotency lock for one allowlisted release."""

    with _PREPARE_LOCKS_GUARD:
        return _PREPARE_LOCKS.setdefault(source_id, threading.Lock())


def lower_layer_diagnostic(phase: str, exc: Exception) -> DemoSourceError:
    """Translate a stage exception without flattening structured detail."""

    cause_code: str | None = None
    cause_detail: Any = None
    try:
        raw_code = getattr(exc, "error", None) or getattr(exc, "code", None)
    except Exception:  # noqa: BLE001 - hostile properties stay unstructured.
        raw_code = None
    if isinstance(raw_code, str) and raw_code.strip():
        cause_code = raw_code.strip()
        try:
            cause_detail = getattr(exc, "detail", None)
        except Exception:  # noqa: BLE001 - preserve the code and safe message.
            cause_detail = None

    structured = cause_code is not None
    if cause_detail is None:
        cause_detail = {"message": str(exc)}
    return DemoSourceError(
        f"Demo source {phase} failed",
        code=f"demo_source_{phase}_failed",
        detail={
            "phase": phase,
            "cause": {
                "type": type(exc).__name__,
                "structured": structured,
                "code": cause_code or "unstructured_exception",
                "detail": cause_detail,
            },
        },
    )


def is_phase_diagnostic(phase: str, exc: DemoSourceError) -> bool:
    """Return whether an error already carries this owner's canonical envelope."""

    if exc.code != f"demo_source_{phase}_failed":
        return False
    detail = exc.detail
    return (
        isinstance(detail, dict)
        and detail.get("phase") == phase
        and isinstance(detail.get("cause"), dict)
    )


def run_prepare_stage(phase: str, operation: Callable[..., Any], *args: Any) -> Any:
    """Run one stage and establish the canonical DemoSource error boundary."""

    try:
        return operation(*args)
    except DemoSourceCancelled:
        raise
    except DemoSourceError as exc:
        if is_phase_diagnostic(phase, exc):
            raise
        raise lower_layer_diagnostic(phase, exc) from exc
    except Exception as exc:
        raise lower_layer_diagnostic(phase, exc) from exc


def build_prepare_runner(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    operations: PrepareOperations,
):
    """Build an idempotent background runner from explicit owner operations."""

    def runner(job: Any) -> dict[str, Any]:
        job.emit(
            {
                "type": "start",
                "phase": "prepare",
                "source_id": source.id,
                "title": source.title,
                "version": source.version,
                "scope": source.scope_summary,
                "bytes_total": source.size_bytes,
            }
        )
        reused_stages: list[str] = []
        try:
            with prepare_lock(source.id):
                check_cancelled(job, "prepare")
                archive_sha256, reused = run_prepare_stage(
                    "download", operations.download, source, paths, job
                )
                if reused:
                    reused_stages.append("download")
                if run_prepare_stage(
                    "extract",
                    operations.extract,
                    source,
                    paths,
                    archive_sha256,
                    job,
                ):
                    reused_stages.append("extract")
                conversion, reused = run_prepare_stage(
                    "convert",
                    operations.convert,
                    source,
                    paths,
                    archive_sha256,
                    job,
                )
                if reused:
                    reused_stages.append("convert")
                export, reused = run_prepare_stage(
                    "export",
                    operations.export,
                    source,
                    paths,
                    archive_sha256,
                    job,
                )
                if reused:
                    reused_stages.append("export")
                registered = run_prepare_stage(
                    "register", operations.register, source, paths, job
                )
        except DemoSourceCancelled as exc:
            return {
                "source_id": source.id,
                "cancelled_at": str(exc),
                "reused": bool(reused_stages),
                "reused_stages": reused_stages,
            }
        return {
            "source_id": source.id,
            "version": source.version,
            "database": source.database,
            "scope": source.scope_summary,
            "reused": bool(reused_stages),
            "reused_stages": reused_stages,
            "conversion": conversion,
            "export": export,
            "registered_source": registered,
            "provenance": {
                "provider": "PhysioNet",
                "landing_page": source.landing_page,
                "citation_url": source.citation_url,
                "license": "ODbL 1.0",
                "attribution": source.attribution,
            },
        }

    return runner


__all__ = [
    "ExportJobProxy",
    "PrepareOperations",
    "build_prepare_runner",
    "convert_dataset",
    "export_dataset",
    "is_phase_diagnostic",
    "lower_layer_diagnostic",
    "prepare_lock",
    "register_export",
    "run_prepare_stage",
]
