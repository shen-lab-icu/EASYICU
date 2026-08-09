"""Official demo-source catalog and background preparation routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import demo_sources as demo_store
from easyicu.webserver.routes.jobs import submit_job

catalog_router = APIRouter()
submission_router = APIRouter()


@catalog_router.get("/api/demo-sources")
def get_demo_sources() -> dict:
    """Return the allowlisted PhysioNet demo catalog and local readiness flags."""

    return demo_store.demo_sources_catalog()


@submission_router.post("/api/jobs/demo-source-prepare")
def jobs_demo_source_prepare(body: Dict[str, Any]) -> dict:
    """Download, convert, all-module export and register one official demo."""

    unexpected = sorted(set(body) - {"source_id"})
    if unexpected:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_demo_source_request",
                "reason": "Only source_id is accepted; URLs and paths are not allowed.",
                "unexpected_fields": unexpected,
            },
        )
    source_id = body.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "source_id_required",
                "allowed_source_ids": list(demo_store.allowed_source_ids()),
            },
        )
    try:
        source = demo_store.get_source(source_id.strip())
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_demo_source",
                "allowed_source_ids": list(demo_store.allowed_source_ids()),
            },
        ) from exc
    runner = demo_store.make_prepare_runner(source.id)
    job = submit_job("demo-source-prepare", runner)
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


__all__ = ["catalog_router", "submission_router"]
