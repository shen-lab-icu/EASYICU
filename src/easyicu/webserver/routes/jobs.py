"""Background job submission and lifecycle API routes."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from easyicu.webserver import crossdb_review
from easyicu.webserver import dataio
from easyicu.webserver import jobs as job_store
from easyicu.webserver import settings as settings_store
from easyicu.webserver import sources as source_store
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

    def runner(job: Any) -> dict:
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
        return result

    job = submit_job("extract", runner)
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


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
