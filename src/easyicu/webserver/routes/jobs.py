"""Background job submission and lifecycle API routes."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from easyicu.webserver import crossdb_review
from easyicu.webserver import dataio
from easyicu.webserver import jobs as job_store
from easyicu.webserver import settings as settings_store
from easyicu.webserver import sources as source_store
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.routes.request_parsing import body_bool

submission_router = APIRouter()
lifecycle_router = APIRouter()


def submit_job(kind: str, runner: Any):
    """Submit a local job while preserving the public capacity error contract."""
    try:
        return job_store.MANAGER.submit(kind, runner)
    except job_store.JobCapacityError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "job_capacity_exceeded",
                "running": exc.running,
                "max_running": exc.max_running,
                "reason": "Wait for a running local job to finish before retrying.",
            },
        ) from exc


@submission_router.post("/api/jobs/convert")
def jobs_convert(body: Dict[str, Any]) -> dict:
    """Start a raw->Parquet conversion as a background job. Returns ``{job_id}``;
    progress streams from ``/api/jobs/{id}/events``."""
    path = str(body.get("path", ""))
    database = str(body.get("database") or "")
    if not path or not database:
        raise HTTPException(status_code=400, detail="path and database are required")
    job = submit_job("convert", dataio.make_convert_runner(path, database))
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


@submission_router.post("/api/jobs/extract")
def jobs_extract(body: Dict[str, Any]) -> dict:
    """Start a feature-module extraction/export as a background job. Returns
    ``{job_id}``; progress streams from ``/api/jobs/{id}/events``."""
    path = str(body.get("path", ""))
    database = str(body.get("database") or "")
    if not path or not database:
        raise HTTPException(status_code=400, detail="path and database are required")
    settings = settings_store.load_settings()
    out_dir = body.get("out_dir") or settings.get("export_dir")
    export_runner = dataio.make_export_runner(
        data_path=path,
        database=database,
        modules=body.get("modules"),
        concepts=body.get("concepts"),
        export_format=str(body.get("format") or "csv"),
        merge=body_bool(body, "merge"),
        out_dir=out_dir,
        create_run_subdir=True,
        max_patients=body.get("max_patients"),
        cohort=body.get("cohort"),
        include_feature_definitions=body_bool(
            body, "include_feature_definitions", True
        ),
    )

    study_context_id = str(body.get("study_context_id") or "").strip()
    expected_revision = body.get("study_context_revision")
    study_context = None
    if study_context_id:
        try:
            study_context = context_store.get_context(study_context_id)
        except context_store.StudyContextError as exc:
            raise HTTPException(status_code=400, detail=exc.detail) from exc
        if study_context is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "study_context_not_found",
                    "study_context_id": study_context_id,
                },
            )
        current_revision = int(study_context.get("revision") or 0)
        if expected_revision is not None and expected_revision != current_revision:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "study_context_revision_conflict",
                    "study_context_id": study_context_id,
                    "expected_revision": expected_revision,
                    "current_revision": current_revision,
                },
            )
        bound_source = study_context.get("data_source")
        bound_source = bound_source if isinstance(bound_source, dict) else {}
        try:
            bound_path = context_store.normalize_path(bound_source.get("path"))
            requested_path = context_store.normalize_path(path)
        except context_store.StudyContextError as exc:
            raise HTTPException(status_code=400, detail=exc.detail) from exc
        if not bound_path or bound_path != requested_path:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "study_context_source_mismatch",
                    "study_context_id": study_context_id,
                },
            )
        if str(bound_source.get("database") or "").strip() != database:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "study_context_database_mismatch",
                    "study_context_id": study_context_id,
                },
            )

    start_gate = threading.Event()
    start_abort: Dict[str, Any] = {}

    def runner(job: Any) -> dict:
        start_gate.wait()
        if start_abort:
            error = str(start_abort.get("error") or "study_context_sync_failed")
            raise RuntimeError(f"extract_start_blocked:{error}")
        terminal_stage = "extract_failed"
        result: Dict[str, Any] | None = None
        try:
            result = export_runner(job)
            out_path = str((result or {}).get("out_dir") or "")
            if out_path and result.get("manifest") and not result.get("cancelled_at"):
                registry = source_store.register_source(
                    out_path,
                    label=body.get("label"),
                    active=True,
                    crossdb=True,
                )
                result["registered_source"] = {
                    "ok": bool(registry.get("ok")),
                    "active_path": registry.get("active_path"),
                    "source_count": len(registry.get("sources") or []),
                }
            terminal_stage = (
                "extract_cancelled" if job.cancel_requested else "extract_review"
            )
            return result
        finally:
            if study_context is not None:
                try:
                    cleanup = context_store.clear_active_job_if(
                        study_context_id,
                        job.id,
                        current_stage=terminal_stage,
                        last_route="extract",
                    )
                    if isinstance(result, dict):
                        result["study_context_revision"] = int(
                            cleanup["context"].get("revision") or 0
                        )
                except Exception:
                    pass

    job = submit_job("extract", runner)
    synced_context = None
    try:
        if study_context is not None:
            synced_context = context_store.handoff_context(
                study_context_id,
                current_stage="extract",
                last_route="extract",
                active_job_id=job.id,
                expected_revision=int(study_context.get("revision") or 0),
            )
    except context_store.StudyContextError as exc:
        start_abort.update(exc.detail)
    except Exception as exc:
        start_abort.update(
            {
                "error": "study_context_active_job_sync_failed",
                "reason": type(exc).__name__,
            }
        )
    finally:
        start_gate.set()
    if start_abort:
        error = str(start_abort.get("error") or "study_context_sync_failed")
        status_code = 409 if error.startswith("study_context_revision_") else 500
        raise HTTPException(
            status_code=status_code,
            detail={**start_abort, "job_id": job.id, "job_started": False},
        )
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "study_context_id": study_context_id or None,
        "study_context_revision": (
            int(synced_context.get("revision") or 0) if synced_context else None
        ),
    }


@submission_router.post("/api/jobs/crossdb-summary")
def jobs_crossdb_summary(body: Dict[str, Any]) -> dict:
    """Start a leased, cancellable registered-export Cross-DB summary job."""
    try:
        runner = crossdb_review.make_crossdb_review_summary_runner(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc

    lease_receipt = runner.source_lease_receipt()
    try:
        job = submit_job("crossdb-summary", runner)
    except Exception:
        runner.release()
        raise
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "selection_receipt": runner.selection_receipt,
        "source_lease": lease_receipt,
        "deadline_seconds": runner.deadline_seconds,
        "deadline_at": runner.deadline_at,
    }


@submission_router.post("/api/jobs/crossdb-raw-distribution")
def jobs_crossdb_raw_distribution(body: Dict[str, Any]) -> dict:
    """Start a raw local Cross-DB density aggregation job.

    Full raw-database density comparisons can span several ICU databases and
    many concepts, so the native UI runs them through the same local job/SSE
    model as extraction instead of a foreground request.
    """
    job = submit_job(
        "crossdb-raw-distribution",
        crossdb_review.make_crossdb_raw_distribution_runner(body),
    )
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


@lifecycle_router.get("/api/jobs/{job_id}")
def jobs_get(job_id: str) -> dict:
    """Full snapshot of a job (events history + terminal state) for reconnect."""
    job = job_store.MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.snapshot()


@lifecycle_router.post("/api/jobs/{job_id}/cancel")
def jobs_cancel(job_id: str, body: Optional[Dict[str, Any]] = None) -> dict:
    """Request cooperative cancellation for a running local job."""
    job = job_store.MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    reason = str((body or {}).get("reason") or "user_requested")
    requested = job.request_cancel(reason=reason)
    snap = job.snapshot()
    snap["cancel_request_accepted"] = requested
    return snap


@lifecycle_router.get("/api/jobs/{job_id}/events")
async def jobs_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events: replay the job's event history, then tail live events
    until the job reaches a terminal status."""
    job = job_store.MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    async def gen():
        sent = 0
        while True:
            # Read the event slice and status under the Job's per-instance lock.
            events, status = job.events_since(sent)
            for ev in events:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            sent += len(events)
            if status != "running":
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(gen(), media_type="text/event-stream")


__all__ = ["lifecycle_router", "submission_router", "submit_job"]
