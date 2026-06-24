"""EasyICU web server — FastAPI backend for the native (non-Streamlit) UI.

Migration target (route C, see WEBAPP_MIGRATION_PLAN.md): the static frontend
under ``static/`` is the real product UI (vendored from the easyicu_ui design
repo and evolved here). Python lives behind ``/api/*`` instead of rendering DOM.

Runs locally to preserve the local-first contract (no data upload, local
filesystem access for data roots). This module is Stage 0+1: it serves the
frontend and the first real read-only endpoint, ``/api/catalog``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from easyicu.webserver import agent_runs
from easyicu.webserver import cohort_review
from easyicu.webserver import crossdb_review
from easyicu.webserver import dataio
from easyicu.webserver import extraction_filters
from easyicu.webserver import patient_drilldown
from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
from easyicu.webserver import sources as source_store
from easyicu.webserver.catalog import build_catalog
from easyicu.webserver.jobs import MANAGER

STATIC_DIR = Path(__file__).with_name("static")

app = FastAPI(title="EasyICU", version="0.1.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/catalog")
def catalog() -> dict:
    """The concept catalog the Data Dictionary screen renders."""
    return build_catalog()


@app.get("/api/settings")
def get_settings() -> dict:
    """Local settings + read-only environment facts for the Settings screen."""
    return {**settings_store.load_settings(), "about": settings_store.about()}


@app.post("/api/settings")
def post_settings(patch: Dict[str, Any]) -> dict:
    """Merge-update known settings keys and persist locally."""
    return {**settings_store.update_settings(patch), "about": settings_store.about()}


@app.get("/api/fs/list")
def fs_list(path: str | None = None) -> dict:
    """Server-side directory listing for the data-folder picker (local-first)."""
    return dataio.list_dir(path)


@app.post("/api/data/scan")
def data_scan(body: Dict[str, Any]) -> dict:
    """Inspect a folder: detect database, layout, and extraction readiness."""
    return dataio.scan_path(str(body.get("path", "")), body.get("source"))


@app.post("/api/workspace/summary")
def workspace_summary(body: Dict[str, Any]) -> dict:
    """Summarise an EasyICU export folder for Patient/Cohort review screens."""
    path = str(body.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = dataio.summarize_export_workspace(path)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/patient-review/drilldown")
def patient_review_drilldown(body: Dict[str, Any]) -> dict:
    """Return bounded real Patient Review aggregates plus one entity drilldown."""
    try:
        return patient_drilldown.patient_review_drilldown(body)
    except patient_drilldown.PatientReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@app.post("/api/cohort-review/summary")
def cohort_review_summary(body: Dict[str, Any]) -> dict:
    """Return bounded real Cohort Review aggregates for the active export."""
    try:
        return cohort_review.cohort_review_summary(body)
    except cohort_review.CohortReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@app.post("/api/crossdb-review/summary")
def crossdb_review_summary(body: Dict[str, Any]) -> dict:
    """Return bounded real Cross-DB descriptive aggregates for registered exports."""
    try:
        return crossdb_review.crossdb_review_summary(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@app.post("/api/extraction/filter-options")
def extraction_filter_options(body: Dict[str, Any]) -> dict:
    """Return bounded real-source filter metadata for Data Extraction."""
    try:
        return extraction_filters.filter_options(body)
    except extraction_filters.ExtractionFilterError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@app.post("/api/extraction/filter-preview")
def extraction_filter_preview(body: Dict[str, Any]) -> dict:
    """Apply supported extraction metadata filters; unsupported filters fail closed."""
    try:
        result = extraction_filters.filter_preview(body)
    except extraction_filters.ExtractionFilterError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/workspaces/crossdb-summary")
def workspaces_crossdb_summary(body: Dict[str, Any]) -> dict:
    """Summarise two or more local EasyICU exports for Cross-DB preview."""
    paths = body.get("paths")
    if not isinstance(paths, list):
        raise HTTPException(status_code=400, detail="paths must be a list")
    result = dataio.summarize_crossdb_workspaces([str(p) for p in paths])
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/api/workspaces/registry")
def workspaces_registry() -> dict:
    """Local export-source registry shared by Review, Cross-DB, Agent, Copilot."""
    return source_store.load_registry()


@app.post("/api/workspaces/registry")
def post_workspaces_registry(patch: Dict[str, Any]) -> dict:
    """Merge-update local export-source registry selections."""
    return source_store.save_registry(patch)


@app.post("/api/workspaces/register")
def post_workspaces_register(body: Dict[str, Any]) -> dict:
    """Validate and register one local EasyICU export folder."""
    path = str(body.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = source_store.register_source(
        path,
        label=body.get("label"),
        active=bool(body.get("active", True)),
        crossdb=bool(body.get("crossdb", True)),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/workspaces/rename")
def post_workspaces_rename(body: Dict[str, Any]) -> dict:
    """Rename one registered local export source in registry metadata only."""
    path = str(body.get("path", ""))
    label = str(body.get("label", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = source_store.rename_source(path, label)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/workspaces/remove")
def post_workspaces_remove(body: Dict[str, Any]) -> dict:
    """Unregister one source. This never deletes export files from disk."""
    path = str(body.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = source_store.remove_source(path)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/jobs/convert")
def jobs_convert(body: Dict[str, Any]) -> dict:
    """Start a raw->Parquet conversion as a background job. Returns ``{job_id}``;
    progress streams from ``/api/jobs/{id}/events``."""
    path = str(body.get("path", ""))
    database = str(body.get("database") or "")
    if not path or not database:
        raise HTTPException(status_code=400, detail="path and database are required")
    job = MANAGER.submit("convert", dataio.make_convert_runner(path, database))
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


@app.post("/api/jobs/extract")
def jobs_extract(body: Dict[str, Any]) -> dict:
    """Start a feature-module extraction/export as a background job. Returns
    ``{job_id}``; progress streams from ``/api/jobs/{id}/events``."""
    path = str(body.get("path", ""))
    database = str(body.get("database") or "")
    if not path or not database:
        raise HTTPException(status_code=400, detail="path and database are required")
    runner = dataio.make_export_runner(
        data_path=path,
        database=database,
        modules=body.get("modules"),
        export_format=str(body.get("format") or "csv"),
        merge=bool(body.get("merge")),
        out_dir=body.get("out_dir"),
        max_patients=body.get("max_patients"),
    )
    job = MANAGER.submit("extract", runner)
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


@app.post("/api/jobs/agent-run")
def jobs_agent_run(body: Dict[str, Any]) -> dict:
    """Start a registry-backed local Research Agent run.

    The default run is deterministic and local: it consumes the active export
    summary, writes bounded artifacts, and stops at an evidence gate. The
    optional ``run_type=full`` path can use either the offline mock provider or
    a configured external provider after canonical AI opt-in, per-run opt-in,
    and credential checks.
    """
    path = str(body.get("path") or "")
    if not path:
        path = str(source_store.load_registry().get("active_path") or "")
    if not path:
        raise HTTPException(status_code=400, detail={"error": "no_active_export"})

    desc = dataio.describe_export_source(path)
    if not desc.get("ok"):
        raise HTTPException(status_code=400, detail=desc)

    run_type = str(body.get("run_type") or ("full" if body.get("full_run") else "preflight"))
    llm_provider = str(body.get("llm_provider") or body.get("provider") or "mock")
    external_llm_opt_in = bool(body.get("external_llm_opt_in"))
    settings = settings_store.load_settings()
    try:
        agent_runs.validate_agent_run_config(
            run_type=run_type,
            llm_provider=llm_provider,
            external_llm_opt_in=external_llm_opt_in,
            ai_enabled=bool(settings.get("ai_enabled")),
        )
    except agent_runs.AgentRunConfigError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc

    runner = agent_runs.make_agent_run_runner(
        export_path=path,
        study_id=str(body.get("study_id") or "study"),
        mode=str(body.get("mode") or "analysis"),
        question=body.get("question"),
        project_root=body.get("project_root"),
        run_type=run_type,
        llm_provider=llm_provider,
        external_llm_opt_in=external_llm_opt_in,
        ai_enabled=bool(settings.get("ai_enabled")),
    )
    job = MANAGER.submit("agent-run", runner)
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


@app.get("/api/agent-runs/provider-status")
def get_agent_run_provider_status(provider: str = "openai") -> dict:
    """Return sanitized external-provider readiness for UI controls.

    This endpoint never constructs a model client and never returns credential
    values. It only reports whether the expected environment variables are
    present and whether global AI opt-in is enabled.
    """
    settings = settings_store.load_settings()
    return {
        "ok": True,
        "provider_status": provider_adapter.provider_readiness(
            provider,
            ai_enabled=bool(settings.get("ai_enabled")),
        ),
    }


@app.post("/api/agent-runs/review")
def post_agent_run_review(body: Dict[str, Any]) -> dict:
    """Read a bounded local artifact bundle for human review."""
    result = agent_runs.read_run_review(str(body.get("project_dir") or ""))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/agent-runs/signoff")
def post_agent_run_signoff(body: Dict[str, Any]) -> dict:
    """Write a local human signoff artifact without unlocking the draft."""
    result = agent_runs.create_human_signoff(
        str(body.get("project_dir") or ""),
        reviewer=body.get("reviewer"),
        confirmations=body.get("confirmations") if isinstance(body.get("confirmations"), list) else [],
        note=body.get("note"),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/agent-runs/history")
def post_agent_run_history(body: Dict[str, Any]) -> dict:
    """List local agent run directories by reading bounded artifacts only."""
    return agent_runs.list_run_history(
        study_id=body.get("study_id"),
        project_root=body.get("project_root"),
        limit=int(body.get("limit") or 50),
    )


@app.post("/api/agent-runs/artifact")
def post_agent_run_artifact(body: Dict[str, Any]) -> dict:
    """Return a bounded JSON viewer payload for one whitelisted artifact."""
    result = agent_runs.read_run_artifact(
        str(body.get("project_dir") or ""),
        str(body.get("artifact") or body.get("name") or ""),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/agent-runs/download-artifact")
def post_agent_run_download_artifact(body: Dict[str, Any]) -> Response:
    """Download one whitelisted artifact as its original local JSON bytes."""
    result = agent_runs.read_run_artifact_bytes(
        str(body.get("project_dir") or ""),
        str(body.get("artifact") or body.get("name") or ""),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    filename = str(result.get("name") or "artifact.json").replace('"', "")
    return Response(
        content=result["content"],
        media_type=str(result.get("media_type") or "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/agent-runs/download-bundle")
def post_agent_run_download_bundle(body: Dict[str, Any]) -> Response:
    """Download a zip bundle of whitelisted artifacts for one local run."""
    result = agent_runs.build_run_bundle(str(body.get("project_dir") or ""))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    filename = str(result.get("name") or "agent_run_artifacts.zip").replace('"', "")
    return Response(
        content=result["content"],
        media_type=str(result.get("media_type") or "application/zip"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/jobs/{job_id}")
def jobs_get(job_id: str) -> dict:
    """Full snapshot of a job (events history + terminal state) for reconnect."""
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.snapshot()


@app.get("/api/jobs/{job_id}/events")
async def jobs_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events: replay the job's event history, then tail live events
    until the job reaches a terminal status."""
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    async def gen():
        sent = 0
        while True:
            # Flush any events not yet streamed (covers both replay and live).
            while sent < len(job.events):
                ev = job.events[sent]
                sent += 1
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if job.status != "running":
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(gen(), media_type="text/event-stream")


# Static frontend last, mounted at root, with HTML serving so "/" -> index.html.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:  # console-script entry candidate
    import uvicorn

    uvicorn.run("easyicu.webserver.app:app", host="127.0.0.1", port=8502, reload=False)


if __name__ == "__main__":
    main()
