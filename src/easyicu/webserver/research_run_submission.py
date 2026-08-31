"""Owner for validated ResearchAgentPipeline run submission.

HTTP routes and Copilot tools are adapters.  This module owns the scientific
run-submission decisions that must be identical regardless of which adapter
initiated the run: source authority, provider authority, Planner budget,
resume coordinates, workspace selection, and StudyContext job binding.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import agent_runs
from easyicu.webserver import capabilities
from easyicu.webserver import dataio
from easyicu.webserver import jobs as job_store
from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
from easyicu.webserver import sources as source_store
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.input_validation import parse_bool
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore
from easyicu.webserver.pi_copilot.run_authority import (
    list_bound_run_history,
    research_pipeline_workspace,
    resumable_planner_checkpoint_job_id,
)


_DEVELOPMENT_REVIEWED_EXECUTION_ENV = "EASYICU_DEVELOPMENT_REVIEWED_EXECUTION"


class ResearchRunSubmissionError(RuntimeError):
    """Typed adapter boundary for a rejected research run submission."""

    def __init__(self, detail: Mapping[str, Any], *, status_code: int = 400) -> None:
        self.detail = dict(detail)
        self.status_code = status_code
        super().__init__(
            str(self.detail.get("error") or "research_run_submission_failed")
        )


def _reject(detail: Mapping[str, Any], *, status_code: int = 400) -> None:
    raise ResearchRunSubmissionError(detail, status_code=status_code)


def _body_bool(body: Mapping[str, Any], key: str, default: bool = False) -> bool:
    try:
        return parse_bool(body.get(key), default=default)
    except ValueError as exc:
        raise ResearchRunSubmissionError(
            {"error": "invalid_boolean", "field": key}
        ) from exc


def server_research_pipeline_budget_mode() -> str:
    """Resolve the server-owned development launch mode."""

    raw = str(os.environ.get(_DEVELOPMENT_REVIEWED_EXECUTION_ENV) or "").strip()
    if not raw:
        return "planner_canary"
    if raw == "1":
        return "full_reviewed"
    _reject({"error": "research_pipeline_development_mode_invalid"}, status_code=500)
    raise AssertionError("unreachable")


def research_pipeline_budget_mode_for_source(
    *,
    prepared_manifest: Optional[Path],
    metadata_only_planning_authorized: bool,
) -> str:
    """Keep candidate planning from silently acquiring execution authority."""

    if metadata_only_planning_authorized:
        return "planner_canary"
    if prepared_manifest is not None:
        return "full_reviewed"
    return server_research_pipeline_budget_mode()


def provider_environment_for_agent_run(
    *,
    credential_source: str,
    engine: str,
    run_type: str,
    external_llm_opt_in: bool,
    llm_provider: str = "",
    account_environment: Optional[Mapping[str, str]] = None,
) -> Optional[Mapping[str, str]]:
    """Resolve one credential authority without returning secret values."""

    source = str(credential_source or "scientific_provider").strip().lower()
    account_provider = provider_adapter.is_user_account_provider(llm_provider)
    if source == "codex_user_auth":
        if run_type != "full":
            _reject({"error": "codex_user_auth_full_run_only"})
        if not account_provider:
            _reject({"error": "codex_user_auth_provider_required"})
        try:
            return provider_adapter.account_provider_environment(
                llm_provider,
                environ=account_environment,
            )
        except provider_adapter.ProviderAdapterError as exc:
            raise ResearchRunSubmissionError(exc.detail) from exc
    if engine == "research_agent_pipeline" and account_provider:
        _reject({"error": "research_pipeline_codex_user_auth_required"})
    if account_provider:
        _reject({"error": "codex_user_auth_required"})
    if engine == "research_agent_pipeline" and source != "pi_verified":
        _reject({"error": "research_pipeline_pi_verified_credentials_required"})
    if source == "scientific_provider":
        return None
    if source != "pi_verified":
        _reject({"error": "agent_provider_credential_source_invalid"})
    if engine != "research_agent_pipeline" or run_type != "full":
        _reject({"error": "pi_provider_research_pipeline_only"})
    try:
        return PiProviderConfigStore().research_agent_environment(
            external_llm_opt_in=external_llm_opt_in,
        )
    except PiCopilotError as exc:
        raise ResearchRunSubmissionError(exc.detail) from exc


def _submit_job(kind: str, runner: Any) -> Any:
    try:
        return job_store.MANAGER.submit(kind, runner)
    except job_store.JobCapacityError as exc:
        raise ResearchRunSubmissionError(
            {
                "error": "job_capacity_exceeded",
                "running": exc.running,
                "max_running": exc.max_running,
                "reason": "Wait for a running local job to finish before retrying.",
            },
            status_code=429,
        ) from exc


def submit_research_run(
    body: Dict[str, Any],
    *,
    account_environment: Optional[Mapping[str, str]] = None,
    metadata_only_planning_authorized: bool = False,
) -> dict:
    """Validate, bind, and start one ResearchAgentPipeline run."""

    engine = str(body.get("engine") or "").strip().lower()
    if engine != "research_agent_pipeline":
        _reject({"error": "research_run_submission_engine_required"})

    path = str(body.get("path") or "")
    if not path:
        path = str(source_store.load_registry().get("active_path") or "")
    if not path:
        _reject({"error": "no_active_export"})
    desc = dataio.describe_export_source(path)
    if not desc.get("ok"):
        _reject(desc)

    study_context_id = str(body.get("study_context_id") or "").strip()
    if not study_context_id:
        _reject({"error": "research_pipeline_study_context_required"})
    try:
        study_context = context_store.get_context(study_context_id)
    except context_store.StudyContextError as exc:
        raise ResearchRunSubmissionError(exc.detail) from exc
    if study_context is None:
        _reject(
            {
                "error": "study_context_not_found",
                "study_context_id": study_context_id,
            }
        )

    run_type = str(
        body.get("run_type")
        or ("full" if _body_bool(body, "full_run") else "preflight")
    )
    if run_type != "full":
        _reject({"error": "research_pipeline_requires_full_run"})
    planner_start_mode = str(body.get("planner_start_mode") or "auto").strip().lower()
    if planner_start_mode not in {"auto", "fresh", "resume_checkpoint"}:
        _reject(
            {
                "error": "planner_start_mode_invalid",
                "planner_start_mode": planner_start_mode,
            }
        )
    if "budget_mode" in body:
        _reject({"error": "research_pipeline_budget_mode_server_owned"})

    source = study_context.get("data_source")
    database = source.get("database") if isinstance(source, Mapping) else None
    prepared_manifest = dataio.prepared_export_manifest_path(Path(path).expanduser())
    budget_mode = research_pipeline_budget_mode_for_source(
        prepared_manifest=prepared_manifest,
        metadata_only_planning_authorized=metadata_only_planning_authorized,
    )
    if budget_mode == "full_reviewed" or prepared_manifest is not None:
        try:
            dataio.validate_research_pipeline_source(path, database=database)
        except dataio.ExportCohortError as exc:
            raise ResearchRunSubmissionError(exc.detail) from exc

    llm_provider = str(body.get("llm_provider") or body.get("provider") or "mock")
    external_llm_opt_in = _body_bool(body, "external_llm_opt_in")
    literature_search_authorized = _body_bool(
        body,
        "literature_search_authorized",
    )
    if literature_search_authorized and not external_llm_opt_in:
        _reject({"error": "literature_search_authorization_scope_invalid"})
    default_credential_source = (
        "codex_user_auth"
        if provider_adapter.is_user_account_provider(llm_provider)
        else "scientific_provider"
    )
    credential_source = str(
        body.get("credential_source") or default_credential_source
    ).strip()
    if (
        provider_adapter.is_user_account_provider(llm_provider)
        and account_environment is None
    ):
        _reject({"error": "codex_auth_login_required"})
    provider_environment = provider_environment_for_agent_run(
        credential_source=credential_source,
        engine=engine,
        run_type=run_type,
        external_llm_opt_in=external_llm_opt_in,
        llm_provider=llm_provider,
        account_environment=account_environment,
    )
    settings = settings_store.load_settings()
    compute = capabilities.validate_compute_target(body)
    if not compute.get("ok"):
        _reject(compute)
    try:
        provider_meta = agent_runs.resolve_agent_provider_config(
            run_type=run_type,
            llm_provider=llm_provider,
            external_llm_opt_in=external_llm_opt_in,
            ai_enabled=bool(settings.get("ai_enabled")),
            environ=provider_environment,
        )
        context_store.build_agent_context_binding(
            study_context,
            export_path=path,
            request_question=body.get("question"),
        )
        workspace = research_pipeline_workspace()
        project_root = str(workspace.project_root(str(study_context.get("id") or "")))
        runner_kwargs: Dict[str, Any] = {
            "export_path": path,
            "study_context": study_context,
            "project_root": project_root,
            "provider": provider_meta,
            "provider_environment": provider_environment,
            "credential_source": credential_source,
            "budget_mode": budget_mode,
        }
        if literature_search_authorized:
            runner_kwargs["literature_search_authorized"] = True
        plan_revision_source_run_id = str(
            body.get("plan_revision_source_run_id") or ""
        ).strip()
        execution_resume_source_run_id = str(
            body.get("execution_resume_source_run_id") or ""
        ).strip()
        development_resume_source_job_id = str(
            body.get("development_resume_source_job_id") or ""
        ).strip()
        if planner_start_mode == "fresh" and (
            development_resume_source_job_id
            or plan_revision_source_run_id
            or execution_resume_source_run_id
        ):
            _reject({"error": "fresh_plan_resume_coordinate_forbidden"})
        if planner_start_mode == "resume_checkpoint" and (
            plan_revision_source_run_id or execution_resume_source_run_id
        ):
            _reject({"error": "planner_checkpoint_resume_coordinate_conflict"})
        if plan_revision_source_run_id:
            runner_kwargs["plan_revision_source_run_id"] = plan_revision_source_run_id
        if execution_resume_source_run_id:
            runner_kwargs["execution_resume_source_run_id"] = (
                execution_resume_source_run_id
            )
        if (
            not development_resume_source_job_id
            and not plan_revision_source_run_id
            and not execution_resume_source_run_id
            and planner_start_mode != "fresh"
        ):
            development_resume_source_job_id = resumable_planner_checkpoint_job_id(
                study=study_context,
                rows=list_bound_run_history(
                    study_context_id=str(study_context.get("id") or ""),
                    project_root=project_root,
                    limit=50,
                ),
                project_root=project_root,
            )
        if (
            planner_start_mode == "resume_checkpoint"
            and not development_resume_source_job_id
        ):
            _reject({"error": "planner_checkpoint_not_available"})
        if development_resume_source_job_id:
            runner_kwargs["development_resume_source_job_id"] = (
                development_resume_source_job_id
            )
        base_runner = agent_pipeline_runs.make_research_pipeline_run_runner(
            **runner_kwargs
        )
    except agent_runs.AgentRunConfigError as exc:
        raise ResearchRunSubmissionError(exc.detail) from exc
    except context_store.StudyContextError as exc:
        raise ResearchRunSubmissionError(exc.detail) from exc
    except PiCopilotError as exc:
        raise ResearchRunSubmissionError(exc.detail) from exc
    except agent_pipeline_runs.ResearchPipelineRunError as exc:
        raise ResearchRunSubmissionError(
            {
                "error": exc.code,
                "message": str(exc),
                **({"details": exc.details} if exc.details else {}),
            }
        ) from exc

    start_gate = threading.Event()
    start_abort: Dict[str, Any] = {}

    def runner(job: Any) -> dict:
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

    job = _submit_job("agent-run", runner)
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
        status_code = 409 if error.startswith("study_context_revision_") else 500
        _reject(
            {**start_abort, "job_id": job.id, "job_started": False},
            status_code=status_code,
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
                "budget_mode": budget_mode,
                "compute_target": compute.get("compute_target"),
                "study_context_id": study_context_id,
                "planner_start_mode": planner_start_mode,
                "development_resume_source_job_id": (
                    development_resume_source_job_id or None
                ),
            },
        )
    except Exception:
        audit_warning = {
            "error": "agent_run_audit_write_failed",
            "job_id": job.id,
        }
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "engine": engine,
        "study_context_id": study_context_id,
        "study_context_revision": (
            int(synced_context.get("revision") or 0) if synced_context else None
        ),
        "planner_start_mode": planner_start_mode,
        "development_resume_source_job_id": development_resume_source_job_id or None,
        "audit_warning": audit_warning,
    }


__all__ = [
    "ResearchRunSubmissionError",
    "provider_environment_for_agent_run",
    "research_pipeline_budget_mode_for_source",
    "server_research_pipeline_budget_mode",
    "submit_research_run",
]
