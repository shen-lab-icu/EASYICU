"""Research Agent control, review, and artifact API routes."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from easyicu.webserver import agent_runs
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import capabilities
from easyicu.webserver import dataio
from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
from easyicu.webserver import science_workbench
from easyicu.webserver import sources as source_store
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.ideas.mining import EXECUTION_GATE_BLOCKERS
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore
from easyicu.webserver.routes.jobs import submit_job
from easyicu.webserver.routes.request_parsing import body_bool

control_router = APIRouter()
artifact_router = APIRouter()


def _provider_environment_for_agent_run(
    *,
    credential_source: str,
    engine: str,
    run_type: str,
    external_llm_opt_in: bool,
) -> Optional[Mapping[str, str]]:
    """Resolve one credential authority without returning secret values."""

    source = str(credential_source or "scientific_provider").strip().lower()
    if source == "scientific_provider":
        return None
    if source != "pi_verified":
        raise HTTPException(
            status_code=400,
            detail={"error": "agent_provider_credential_source_invalid"},
        )
    if engine != "research_agent_pipeline" or run_type != "full":
        raise HTTPException(
            status_code=400,
            detail={"error": "pi_provider_research_pipeline_only"},
        )
    try:
        return PiProviderConfigStore().research_agent_environment(
            external_llm_opt_in=external_llm_opt_in,
        )
    except PiCopilotError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@control_router.post("/api/jobs/agent-run")
def jobs_agent_run(body: Dict[str, Any]) -> dict:
    """Start a registry-backed local Research Agent run.

    The default run is deterministic and local: it consumes the active export
    summary, writes bounded artifacts, and stops at an evidence gate. The
    optional ``run_type=full`` path can use either the legacy bounded summary
    profile or the real ResearchAgentPipeline engine after canonical AI
    opt-in, per-run opt-in, and credential checks.
    """
    path = str(body.get("path") or "")
    if not path:
        path = str(source_store.load_registry().get("active_path") or "")
    if not path:
        raise HTTPException(status_code=400, detail={"error": "no_active_export"})

    desc = dataio.describe_export_source(path)
    if not desc.get("ok"):
        raise HTTPException(status_code=400, detail=desc)
    study_context_id = str(body.get("study_context_id") or "").strip()
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
    project_seed_dir = str(
        body.get("project_seed_dir") or body.get("project_seed_path") or ""
    ).strip()
    if project_seed_dir:
        seed_check = _validate_agent_project_seed_for_run(project_seed_dir, path)
        if not seed_check.get("ok"):
            raise HTTPException(status_code=400, detail=seed_check)

    run_type = str(
        body.get("run_type") or ("full" if body_bool(body, "full_run") else "preflight")
    )
    engine = str(body.get("engine") or "native_summary").strip().lower()
    if engine not in {"native_summary", "research_agent_pipeline"}:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_agent_run_engine", "engine": engine},
        )
    if engine == "research_agent_pipeline" and run_type != "full":
        raise HTTPException(
            status_code=400,
            detail={"error": "research_pipeline_requires_full_run"},
        )
    if engine == "research_agent_pipeline" and study_context is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "research_pipeline_study_context_required"},
        )
    llm_provider = str(body.get("llm_provider") or body.get("provider") or "mock")
    external_llm_opt_in = body_bool(body, "external_llm_opt_in")
    literature_search_authorized = body_bool(
        body,
        "literature_search_authorized",
    )
    if literature_search_authorized and (
        engine != "research_agent_pipeline"
        or run_type != "full"
        or not external_llm_opt_in
    ):
        raise HTTPException(
            status_code=400,
            detail={"error": "literature_search_authorization_scope_invalid"},
        )
    credential_source = str(
        body.get("credential_source") or "scientific_provider"
    ).strip()
    provider_environment = _provider_environment_for_agent_run(
        credential_source=credential_source,
        engine=engine,
        run_type=run_type,
        external_llm_opt_in=external_llm_opt_in,
    )
    settings = settings_store.load_settings()
    compute = capabilities.validate_compute_target(body)
    if not compute.get("ok"):
        raise HTTPException(status_code=400, detail=compute)
    try:
        provider_meta = agent_runs.resolve_agent_provider_config(
            run_type=run_type,
            llm_provider=llm_provider,
            external_llm_opt_in=external_llm_opt_in,
            ai_enabled=bool(settings.get("ai_enabled")),
            environ=provider_environment,
        )
    except agent_runs.AgentRunConfigError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc

    try:
        if study_context is not None:
            # Validate before a background job is created. Do not persist an
            # ``analyze`` stage yet: capacity rejection must leave the prior
            # context untouched.
            context_store.build_agent_context_binding(
                study_context,
                export_path=path,
                request_question=body.get("question"),
            )
        project_root = body.get("project_root") or _agent_seed_run_root(
            project_seed_dir
        )
        if engine == "research_agent_pipeline":
            runner_kwargs: Dict[str, Any] = {
                "export_path": path,
                "study_context": study_context or {},
                "project_root": project_root,
                "provider": provider_meta,
                "provider_environment": provider_environment,
            }
            if literature_search_authorized:
                runner_kwargs["literature_search_authorized"] = True
            plan_revision_source_run_id = str(
                body.get("plan_revision_source_run_id") or ""
            ).strip()
            if plan_revision_source_run_id:
                runner_kwargs["plan_revision_source_run_id"] = (
                    plan_revision_source_run_id
                )
            base_runner = agent_pipeline_runs.make_research_pipeline_run_runner(
                **runner_kwargs,
            )
        else:
            base_runner = agent_runs.make_agent_run_runner(
                export_path=path,
                study_id=str(
                    (study_context or {}).get("id") or body.get("study_id") or "study"
                ),
                mode=str(body.get("mode") or "analysis"),
                question=body.get("question"),
                project_root=project_root,
                run_type=run_type,
                llm_provider=llm_provider,
                external_llm_opt_in=external_llm_opt_in,
                ai_enabled=bool(settings.get("ai_enabled")),
                study_context=study_context,
            )
    except context_store.StudyContextError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    except agent_pipeline_runs.ResearchPipelineRunError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": exc.code,
                "message": str(exc),
                **({"details": exc.details} if exc.details else {}),
            },
        ) from exc
    start_gate = threading.Event()
    start_abort: Dict[str, Any] = {}

    def runner(job: Any) -> dict:
        # ``submit`` starts its daemon thread immediately. Hold execution until
        # the route has bound the returned job id to StudyContext, then always
        # clear that active pointer at the terminal boundary.
        start_gate.wait()
        if start_abort:
            error = str(start_abort.get("error") or "study_context_sync_failed")
            raise RuntimeError(f"agent_run_start_blocked:{error}")
        terminal_stage = "agent_failed"
        result: Dict[str, Any] | None = None
        try:
            result = base_runner(job)
            if job.cancel_requested:
                terminal_stage = "agent_cancelled"
            elif (result.get("gate") or {}).get("status") == "blocked":
                terminal_stage = "review_blocked"
            else:
                terminal_stage = "review"
            return result
        finally:
            if study_context is not None:
                try:
                    cleanup = context_store.clear_active_job_if(
                        study_context_id,
                        job.id,
                        current_stage=terminal_stage,
                        last_route="agent",
                    )
                    if isinstance(result, dict):
                        result["study_context_revision"] = int(
                            cleanup["context"].get("revision") or 0
                        )
                except Exception:
                    # Metadata cleanup must never turn a completed analysis
                    # into a failed job. The job snapshot remains authoritative.
                    pass

    job = submit_job("agent-run", runner)
    synced_context = None
    try:
        if study_context is not None:
            synced_context = context_store.handoff_context(
                study_context_id,
                current_stage="analyze",
                last_route="agent",
                active_job_id=job.id,
                expected_revision=int(study_context.get("revision") or 0),
            )
    except context_store.StudyContextError as exc:
        if study_context is not None:
            start_abort.update(exc.detail)
    except Exception as exc:
        if study_context is not None:
            # A failure to record the active job is not UI metadata. Without
            # that reservation the context does not know an analysis is
            # running, a second request can start another one against the same
            # revision, and the terminal cleanup has no job id to match. The
            # run is blocked rather than allowed to proceed unbound —
            # including for OSError, which used to be downgraded to a warning
            # here while the runner was released anyway.
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
            detail={
                **start_abort,
                "job_id": job.id,
                "job_started": False,
            },
        )

    audit_warning = None
    try:
        capabilities.record_tool_event(
            "agent_run_submitted",
            {
                "job_id": job.id,
                "run_type": run_type,
                "engine": engine,
                "llm_provider": llm_provider,
                "compute_target": compute.get("compute_target"),
                "study_context_id": study_context_id or None,
            },
        )
    except Exception:
        # The analysis is already running; an audit-log filesystem failure is
        # a warning, not a false 500 that loses the job id for the caller. The
        # audit helper also reads JSON settings, so malformed local metadata
        # can raise non-I/O exceptions after submission.
        audit_warning = {
            "error": "agent_run_audit_write_failed",
            "job_id": job.id,
        }
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "engine": engine,
        "study_context_id": study_context_id or None,
        "study_context_revision": (
            int(synced_context.get("revision") or 0) if synced_context else None
        ),
        "audit_warning": audit_warning,
    }


@control_router.post("/api/jobs/agent-run-review")
def jobs_agent_run_review(body: Dict[str, Any]) -> dict:
    """Resume one same-process digest-bound Research Agent plan review."""

    run_id = str(body.get("run_id") or "").strip()
    study_context_id = str(body.get("study_context_id") or "").strip()
    decision = str(body.get("decision") or "").strip().lower()
    if not run_id or not study_context_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "research_pipeline_review_coordinates_required"},
        )
    if decision not in {"approved", "rejected"}:
        raise HTTPException(
            status_code=400,
            detail={"error": "research_pipeline_review_decision_invalid"},
        )
    if not body_bool(body, "external_llm_opt_in"):
        raise HTTPException(
            status_code=400,
            detail={"error": "research_pipeline_review_external_opt_in_required"},
        )
    if not bool(settings_store.load_settings().get("ai_enabled")):
        raise HTTPException(
            status_code=400,
            detail={"error": "ai_not_enabled"},
        )
    pending = agent_pipeline_runs.pending_review(run_id)
    if not pending:
        raise HTTPException(
            status_code=409,
            detail={"error": "research_pipeline_review_not_resumable"},
        )
    if pending.get("study_id") != study_context_id:
        raise HTTPException(
            status_code=409,
            detail={"error": "research_pipeline_review_study_mismatch"},
        )
    if not pending.get("resumable_here"):
        raise HTTPException(
            status_code=409,
            detail={"error": "research_pipeline_review_not_resumable_here"},
        )
    provider_environment = _provider_environment_for_agent_run(
        credential_source=str(
            pending.get("credential_source") or "scientific_provider"
        ),
        engine="research_agent_pipeline",
        run_type="full",
        external_llm_opt_in=True,
    )
    try:
        study_context = context_store.get_context(study_context_id)
    except context_store.StudyContextError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    if study_context is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "study_context_not_found"},
        )
    current_scientific_digest = context_store.scientific_configuration_sha256(
        study_context
    )
    if pending.get("scientific_configuration_sha256") != current_scientific_digest:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "research_pipeline_review_configuration_superseded",
                "run_id": run_id,
            },
        )
    start_gate = threading.Event()
    start_abort: Dict[str, Any] = {}

    def runner(job: Any) -> dict:
        start_gate.wait()
        if start_abort:
            raise RuntimeError(
                "agent_review_start_blocked:"
                + str(start_abort.get("error") or "study_context_sync_failed")
            )
        result: Dict[str, Any] | None = None
        terminal_stage = "agent_failed"
        try:
            result = agent_pipeline_runs.resume_research_pipeline(
                run_id=run_id,
                study_context_id=study_context_id,
                decision=decision,
                reviewer=str(body.get("reviewer") or "local_web_reviewer"),
                note=str(body.get("note") or ""),
                job=job,
                current_study_context=study_context,
                provider_environment=provider_environment,
            )
            terminal_stage = (
                "review_blocked"
                if (result.get("gate") or {}).get("status") == "blocked"
                else "review"
            )
            return result
        finally:
            try:
                cleanup = context_store.clear_active_job_if(
                    study_context_id,
                    job.id,
                    current_stage=terminal_stage,
                    last_route="agent",
                )
                if isinstance(result, dict):
                    result["study_context_revision"] = int(
                        cleanup["context"].get("revision") or 0
                    )
            except Exception:
                pass

    job = submit_job("agent-run", runner)
    synced_context = None
    try:
        synced_context = context_store.handoff_context(
            study_context_id,
            current_stage="analyze",
            last_route="agent",
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
        raise HTTPException(
            status_code=409 if error.startswith("study_context_revision_") else 500,
            detail={**start_abort, "job_id": job.id, "job_started": False},
        )
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "engine": "research_agent_pipeline",
        "review_run_id": run_id,
        "study_context_id": study_context_id,
        "study_context_revision": int(synced_context.get("revision") or 0),
    }


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
            "blocker_codes": ["seed_gate_missing"],
        }
    blockers = [str(item) for item in gate.get("blockers") or [] if item]
    if gate and not gate.get("agent_run_ready_after_human_confirmation"):
        return {
            "ok": False,
            "error": "agent_project_execution_gate_blocked",
            "blockers": blockers,
            # Seeds written before mining.py emitted codes carry only the
            # sentence; recover the code by exact lookup rather than letting
            # the UI regex-match prose to guess what to tell the user.
            "blocker_codes": _blocker_codes(gate, blockers),
            "execution_gate": gate,
        }

    contract = seed.get("active_export_contract") or {}
    if contract.get("demo_like"):
        return {
            "ok": False,
            "error": "agent_project_demo_export_blocked",
            "blockers": ["prepare or select a real EasyICU export"],
            "blocker_codes": ["export_not_real"],
            "active_export_contract": contract,
        }
    contract_status = str(contract.get("status") or "").lower()
    if contract_status and contract_status != "ready":
        return {
            "ok": False,
            "error": "agent_project_export_contract_not_ready",
            "blockers": ["re-extract or confirm missing required concepts"],
            "blocker_codes": ["required_concepts_missing"],
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
                "blocker_codes": ["active_export_changed"],
                "expected_path_hash": expected_hash,
                "active_path_hash": active_hash,
            }
    return {"ok": True}


def _blocker_codes(gate: Dict[str, Any], blockers: List[str]) -> List[str]:
    """Prefer the gate's own codes; recover legacy seeds by exact sentence."""
    codes = [str(code) for code in gate.get("blocker_codes") or [] if code]
    if codes:
        return codes
    legacy = {text: code for code, text in EXECUTION_GATE_BLOCKERS.items()}
    # An unrecognised sentence stays unrecognised. Guessing a remedy from
    # prose is how the UI ended up regex-matching English to decide what to
    # tell a clinician to do next.
    return [legacy[text] for text in blockers if text in legacy]


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


@control_router.get("/api/agent-runs/provider-status")
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


@control_router.post("/api/agent-runs/provider-config")
def post_agent_run_provider_config(body: Dict[str, Any]) -> dict:
    """Persist a private provider config without returning secrets."""
    provider = str(body.get("provider") or "openai")
    enable_ai = body_bool(body, "enable_ai")
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
    settings = settings_store.update_settings({"ai_enabled": enable_ai})
    return {
        **meta,
        "settings": {**settings, "about": settings_store.about()},
        "provider_status": provider_adapter.provider_readiness(
            provider,
            ai_enabled=bool(settings.get("ai_enabled")),
        ),
        "secrets_returned": False,
    }


@control_router.post("/api/agent-runs/review")
def post_agent_run_review(body: Dict[str, Any]) -> dict:
    """Read a bounded local artifact bundle for human review."""
    result = agent_runs.read_run_review(str(body.get("project_dir") or ""))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@control_router.post("/api/agent-runs/science-workbench")
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


@control_router.post("/api/agent-runs/signoff")
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


@control_router.post("/api/agent-runs/history")
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


@artifact_router.post("/api/agent-runs/artifact")
def post_agent_run_artifact(body: Dict[str, Any]) -> dict:
    """Return a bounded JSON viewer payload for one whitelisted artifact."""
    result = agent_runs.read_run_artifact(
        str(body.get("project_dir") or ""),
        str(body.get("artifact") or body.get("name") or ""),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@artifact_router.post("/api/agent-runs/download-artifact")
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


@artifact_router.post("/api/agent-runs/download-bundle")
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


__all__ = ["artifact_router", "control_router"]
