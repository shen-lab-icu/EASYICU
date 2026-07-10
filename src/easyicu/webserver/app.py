"""EasyICU web server — FastAPI backend for the native UI.

Migration target (route C, see WEBAPP_MIGRATION_PLAN.md): the static frontend
under ``static/`` is the real product UI (vendored from the easyicu_ui design
repo and evolved here). Python lives behind ``/api/*`` instead of rendering DOM.

Runs locally to preserve the local-first contract (no data upload, local
filesystem access for data roots). This module is Stage 0+1: it serves the
frontend and the first real read-only endpoint, ``/api/catalog``.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
from pathlib import Path

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from easyicu.webserver import agent_runs
from easyicu.webserver import capabilities
from easyicu.webserver import cohort_review
from easyicu.webserver import crossdb_review
from easyicu.webserver import dataio
from easyicu.webserver import extraction_filters
from easyicu.webserver import patient_drilldown
from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
from easyicu.webserver import science_workbench
from easyicu.webserver import sources as source_store
from easyicu.webserver.jobs import MANAGER, JobCapacityError
from easyicu.webserver.host_security import AllowedHostsMiddleware
from easyicu.webserver.routes.copilot import router as copilot_router
from easyicu.webserver.routes.guided import router as guided_router
from easyicu.webserver.routes.ideas import router as ideas_router
from easyicu.webserver.routes.local_data import router as local_data_router
from easyicu.webserver.routes.page_guide import router as page_guide_router
from easyicu.webserver.routes.request_parsing import body_bool as _body_bool
from easyicu.webserver.routes.system import router as system_router
from easyicu.webserver.routes.workspaces import router as workspaces_router

STATIC_DIR = Path(__file__).with_name("static")

app = FastAPI(title="EasyICU", version="0.1.0")


def _web_allowed_hosts() -> list[str]:
    configured = [
        host.strip()
        for host in os.getenv("EASYICU_WEB_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    ]
    allow_any = os.getenv("EASYICU_WEB_ALLOW_ANY_HOST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if "*" in configured and not allow_any:
        configured = [host for host in configured if host != "*"]
    return configured or ["127.0.0.1", "localhost", "[::1]", "testserver"]


WEB_ALLOWED_HOSTS = _web_allowed_hosts()
app.add_middleware(AllowedHostsMiddleware, allowed_hosts=WEB_ALLOWED_HOSTS)


def _is_local_web_client(request: Request) -> bool:
    peer = request.client.host if request.client else ""
    if peer in {"testclient", "testserver"}:
        return True
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return bool(
        address.version == 6 and address.ipv4_mapped and address.ipv4_mapped.is_loopback
    )


def _submit_job(kind: str, runner: Any):
    try:
        return MANAGER.submit(kind, runner)
    except JobCapacityError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "job_capacity_exceeded",
                "running": exc.running,
                "max_running": exc.max_running,
                "reason": "Wait for a running local job to finish before retrying.",
            },
        ) from exc


@app.middleware("http")
async def local_clients_only(request: Request, call_next):
    """Keep filesystem and job APIs local until the product has remote auth."""
    if not _is_local_web_client(request):
        return JSONResponse(
            status_code=403,
            content={"detail": "EasyICU WebApp accepts loopback clients only."},
        )
    return await call_next(request)


@app.middleware("http")
async def no_store_native_ui_assets(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in {"/", "/index.html"} or path.startswith(("/js/", "/css/")):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


app.include_router(system_router)
app.include_router(local_data_router)


@app.post("/api/patient-review/drilldown")
def patient_review_drilldown(body: Dict[str, Any]) -> dict:
    """Return bounded real Patient Review aggregates plus one entity drilldown."""
    try:
        return patient_drilldown.patient_review_drilldown(body)
    except patient_drilldown.PatientReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@app.post("/api/patient-review/sources")
def patient_review_sources(body: Dict[str, Any] | None = None) -> dict:
    """Return metadata-only local export candidates for Patient Review."""
    return patient_drilldown.patient_review_sources(body or {})


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


@app.post("/api/crossdb-review/raw-distribution")
def crossdb_raw_distribution(body: Dict[str, Any]) -> dict:
    """Return bounded real Cross-DB density aggregates from a local ICU data root."""
    try:
        return crossdb_review.crossdb_raw_distribution(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@app.post("/api/crossdb-review/raw-root-scan")
def crossdb_raw_root_scan(body: Dict[str, Any]) -> dict:
    """Preflight a local raw ICU data root before launching Cross-DB loading."""
    try:
        return crossdb_review.crossdb_raw_root_scan(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@app.post("/api/crossdb-review/demo-distribution")
def crossdb_demo_distribution(body: Dict[str, Any]) -> dict:
    """Return bounded legacy-seeded Cross-DB demo density aggregates."""
    try:
        return crossdb_review.crossdb_demo_distribution(body)
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


app.include_router(workspaces_router)


@app.post("/api/jobs/convert")
def jobs_convert(body: Dict[str, Any]) -> dict:
    """Start a raw->Parquet conversion as a background job. Returns ``{job_id}``;
    progress streams from ``/api/jobs/{id}/events``."""
    path = str(body.get("path", ""))
    database = str(body.get("database") or "")
    if not path or not database:
        raise HTTPException(status_code=400, detail="path and database are required")
    job = _submit_job("convert", dataio.make_convert_runner(path, database))
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


@app.post("/api/jobs/extract")
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
        merge=_body_bool(body, "merge"),
        out_dir=out_dir,
        create_run_subdir=True,
        max_patients=body.get("max_patients"),
        cohort=body.get("cohort"),
        include_feature_definitions=_body_bool(
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

    job = _submit_job("extract", runner)
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


@app.post("/api/jobs/crossdb-raw-distribution")
def jobs_crossdb_raw_distribution(body: Dict[str, Any]) -> dict:
    """Start a raw local Cross-DB density aggregation job.

    Full raw-database density comparisons can span several ICU databases and
    many concepts, so the native UI runs them through the same local job/SSE
    model as extraction instead of a foreground request.
    """
    job = _submit_job(
        "crossdb-raw-distribution",
        crossdb_review.make_crossdb_raw_distribution_runner(body),
    )
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
    project_seed_dir = str(
        body.get("project_seed_dir") or body.get("project_seed_path") or ""
    ).strip()
    if project_seed_dir:
        seed_check = _validate_agent_project_seed_for_run(project_seed_dir, path)
        if not seed_check.get("ok"):
            raise HTTPException(status_code=400, detail=seed_check)

    run_type = str(
        body.get("run_type")
        or ("full" if _body_bool(body, "full_run") else "preflight")
    )
    llm_provider = str(body.get("llm_provider") or body.get("provider") or "mock")
    external_llm_opt_in = _body_bool(body, "external_llm_opt_in")
    settings = settings_store.load_settings()
    compute = capabilities.validate_compute_target(body)
    if not compute.get("ok"):
        raise HTTPException(status_code=400, detail=compute)
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
        project_root=body.get("project_root") or _agent_seed_run_root(project_seed_dir),
        run_type=run_type,
        llm_provider=llm_provider,
        external_llm_opt_in=external_llm_opt_in,
        ai_enabled=bool(settings.get("ai_enabled")),
    )
    job = _submit_job("agent-run", runner)
    capabilities.record_tool_event(
        "agent_run_submitted",
        {
            "job_id": job.id,
            "run_type": run_type,
            "llm_provider": llm_provider,
            "compute_target": compute.get("compute_target"),
        },
    )
    return {"job_id": job.id, "kind": job.kind, "status": job.status}


def _validate_agent_project_seed_for_run(
    project_seed_dir: str, export_path: str
) -> Dict[str, Any]:
    """Fail closed when an Idea-derived Agent seed is not ready to run."""

    seed_path = Path(project_seed_dir).expanduser() / "project_seed.json"
    try:
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "ok": False,
            "error": "agent_project_seed_not_found",
            "project_seed_dir": project_seed_dir,
        }
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "agent_project_seed_invalid_json",
            "project_seed_dir": project_seed_dir,
        }
    except (OSError, UnicodeDecodeError):
        # NotADirectoryError/IsADirectoryError/PermissionError/bad encoding:
        # keep the structured seed-error contract instead of raising a 500.
        return {
            "ok": False,
            "error": "agent_project_seed_unreadable",
            "project_seed_dir": project_seed_dir,
        }

    gate = seed.get("execution_gate") or {}
    if _idea_seed_requires_gate(seed) and not gate:
        return {
            "ok": False,
            "error": "agent_project_execution_gate_missing",
            "blockers": [
                "refresh Agent project from Idea Mining so preflight checks are available"
            ],
        }
    blockers = [str(item) for item in gate.get("blockers") or [] if item]
    if gate and not gate.get("agent_run_ready_after_human_confirmation"):
        return {
            "ok": False,
            "error": "agent_project_execution_gate_blocked",
            "blockers": blockers,
            "execution_gate": gate,
        }

    contract = seed.get("active_export_contract") or {}
    if contract.get("demo_like"):
        return {
            "ok": False,
            "error": "agent_project_demo_export_blocked",
            "blockers": ["prepare or select a real EasyICU export"],
            "active_export_contract": contract,
        }
    contract_status = str(contract.get("status") or "").lower()
    if contract_status and contract_status != "ready":
        return {
            "ok": False,
            "error": "agent_project_export_contract_not_ready",
            "blockers": ["re-extract or confirm missing required concepts"],
            "active_export_contract": contract,
        }

    expected_hash = str(contract.get("path_hash") or "")
    if expected_hash:
        active_hash = hashlib.sha256(
            str(export_path or "").encode("utf-8")
        ).hexdigest()[:16]
        if active_hash != expected_hash:
            return {
                "ok": False,
                "error": "agent_project_active_export_changed",
                "blockers": ["select the same active export used by Idea Mining"],
                "expected_path_hash": expected_hash,
                "active_path_hash": active_hash,
            }
    return {"ok": True}


def _idea_seed_requires_gate(seed: Dict[str, Any]) -> bool:
    return bool(
        seed.get("source_run_id")
        or seed.get("source_idea_id")
        or seed.get("status") == "seeded_from_idea"
    )


def _agent_seed_run_root(project_seed_dir: str) -> str | None:
    if not project_seed_dir:
        return None
    return str(Path(project_seed_dir).expanduser() / "runs")


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


@app.post("/api/agent-runs/provider-config")
def post_agent_run_provider_config(body: Dict[str, Any]) -> dict:
    """Persist a private provider config without returning secrets."""
    provider = str(body.get("provider") or "openai")
    try:
        meta = provider_adapter.write_provider_config(
            provider,
            api_key=str(body.get("api_key") or ""),
            base_url=str(body.get("base_url") or ""),
            model=str(body.get("model") or ""),
            max_tokens=str(body.get("max_tokens") or ""),
            json_format_style=str(body.get("json_format_style") or ""),
            force=True,
        )
    except provider_adapter.ProviderAdapterError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc

    # Fail closed: writing provider credentials must not silently flip the
    # global AI opt-in — only an explicit enable_ai=true may enable it.
    updates: Dict[str, Any] = {"ai_enabled": _body_bool(body, "enable_ai")}
    if updates["ai_enabled"]:
        updates["agent_model_mode"] = "external"
    settings = settings_store.update_settings(updates)
    return {
        **meta,
        "settings": {**settings, "about": settings_store.about()},
        "provider_status": provider_adapter.provider_readiness(
            provider,
            ai_enabled=bool(settings.get("ai_enabled")),
        ),
        "secrets_returned": False,
    }


@app.post("/api/agent-runs/review")
def post_agent_run_review(body: Dict[str, Any]) -> dict:
    """Read a bounded local artifact bundle for human review."""
    result = agent_runs.read_run_review(str(body.get("project_dir") or ""))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/agent-runs/science-workbench")
def post_agent_run_science_workbench(body: Dict[str, Any]) -> dict:
    """Return Claude-Science-style artifact history/reviewer summaries.

    The endpoint is a bounded presentation adapter over local Agent artifacts.
    It does not create runs, read raw patient rows, or unlock manuscript drafts.
    """
    result = science_workbench.build_science_workbench(
        str(body.get("project_dir") or "").strip() or None
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    capabilities.record_tool_event(
        "science_workbench_loaded",
        {"project_dir": str(body.get("project_dir") or "").strip() or None},
    )
    return result


@app.post("/api/agent-runs/signoff")
def post_agent_run_signoff(body: Dict[str, Any]) -> dict:
    """Write a local human signoff artifact without unlocking the draft."""
    result = agent_runs.create_human_signoff(
        str(body.get("project_dir") or ""),
        reviewer=body.get("reviewer"),
        confirmations=(
            body.get("confirmations")
            if isinstance(body.get("confirmations"), list)
            else []
        ),
        note=body.get("note"),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/agent-runs/history")
def post_agent_run_history(body: Dict[str, Any]) -> dict:
    """List local agent run directories by reading bounded artifacts only."""
    # Idea-seeded projects store runs under <seed>/runs (see jobs_agent_run);
    # derive the same root here so their history survives a reload.
    seed_dir = str(
        body.get("project_seed_dir") or body.get("project_seed_path") or ""
    ).strip()
    return agent_runs.list_run_history(
        study_id=body.get("study_id"),
        project_root=body.get("project_root") or _agent_seed_run_root(seed_dir),
        limit=int(body.get("limit") or 50),
    )


app.include_router(guided_router)


app.include_router(copilot_router)
app.include_router(page_guide_router)
app.include_router(ideas_router)


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


@app.post("/api/jobs/{job_id}/cancel")
def jobs_cancel(job_id: str, body: Optional[Dict[str, Any]] = None) -> dict:
    """Request cooperative cancellation for a running local job."""
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    reason = str((body or {}).get("reason") or "user_requested")
    requested = job.request_cancel(reason=reason)
    snap = job.snapshot()
    snap["cancel_request_accepted"] = requested
    return snap


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
