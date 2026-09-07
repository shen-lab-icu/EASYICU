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
from typing import Any, Callable, Dict, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import agent_runs
from easyicu.webserver import capabilities
from easyicu.webserver import dataio
from easyicu.webserver import jobs as job_store
from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore
from easyicu.webserver.pi_copilot.run_authority import (
    list_bound_run_history,
    research_pipeline_workspace,
    resumable_planner_checkpoint_job_id,
)
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot


_DEVELOPMENT_REVIEWED_EXECUTION_ENV = "EASYICU_DEVELOPMENT_REVIEWED_EXECUTION"

_NEXT_STEP_SUMMARIES = {
    "research_pipeline_manifest_required": (
        "The bound directory is a converted ICU database, not a prepared "
        "EasyICU export package, so the Research Agent has nothing "
        "manifest-backed to bind its evidence to. This is not a bad data "
        "source and not a permission problem: run easyicu_start_extraction to "
        "prepare an export package from it first, then generate the plan. Do "
        "not ask the user to re-pick or validate their folder."
    ),
    "no_export_files": (
        "The bound directory has no readable EasyICU export files. Prepare the "
        "package before planning."
    ),
    "planner_checkpoint_not_available": (
        "The unchanged study has no validated Planner checkpoint to continue. "
        "Generate a fresh candidate plan instead."
    ),
}

RunIntent = Literal["candidate_plan", "reviewed_analysis"]
PlannerStartMode = Literal["auto", "fresh", "resume_checkpoint"]


class ResearchRunSubmissionRequest(BaseModel):
    """Path-free adapter request for one governed pipeline submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_context_id: str
    provider: str
    credential_source: str
    external_llm_opt_in: bool
    intent: RunIntent = "reviewed_analysis"
    planner_start_mode: PlannerStartMode = "auto"
    plan_revision_source_run_id: str = ""
    execution_resume_source_run_id: str = ""
    literature_search_authorized: bool = False
    compute_target: Literal["local"] = "local"


class ResearchRunSubmissionReceipt(BaseModel):
    """Immutable, path-free launch receipt shared by every adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.research-run-submission/1"] = (
        "easyicu.research-run-submission/1"
    )
    job_id: str
    kind: str
    status: str
    engine: Literal["research_agent_pipeline"] = "research_agent_pipeline"
    study_context_id: str
    study_context_revision: int
    budget_mode: Literal["planner_canary", "full_reviewed"]
    planner_start_mode: PlannerStartMode
    resume_source_job_id: Optional[str] = None
    run_id_status: Literal["pending_pipeline_start"] = "pending_pipeline_start"
    audit_warning: Optional[Mapping[str, Any]] = None


class ResearchRunSubmissionError(RuntimeError):
    """Typed adapter boundary for a rejected research run submission."""

    def __init__(self, detail: Mapping[str, Any], *, status_code: int = 400) -> None:
        self.detail = dict(detail)
        self.status_code = status_code
        self.code = str(self.detail.get("error") or "research_run_submission_failed")
        self.summary = str(
            self.detail.get("message")
            or _NEXT_STEP_SUMMARIES.get(self.code)
            or "The existing EasyICU run submission boundary rejected the request."
        )
        self.owner = str(
            self.detail.get("owner") or "easyicu.webserver.research_run_submission"
        )
        super().__init__(
            self.code
        )


def _reject(detail: Mapping[str, Any], *, status_code: int = 400) -> None:
    raise ResearchRunSubmissionError(detail, status_code=status_code)


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
    request: ResearchRunSubmissionRequest,
    *,
    account_environment: Optional[Mapping[str, str]] = None,
    authorize: Optional[Callable[[], None]] = None,
) -> ResearchRunSubmissionReceipt:
    """Validate, bind, and start one ResearchAgentPipeline run."""

    if not isinstance(request, ResearchRunSubmissionRequest):
        raise TypeError("submit_research_run requires ResearchRunSubmissionRequest")
    engine = "research_agent_pipeline"
    run_type = "full"
    study_context_id = request.study_context_id.strip()
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

    source = study_context.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    path = str(source.get("path") or "").strip()
    if not path:
        _reject({"error": "study_context_source_required"})
    desc = dataio.describe_export_source(path)
    if not desc.get("ok"):
        _reject(desc)
    readiness = build_research_workflow_snapshot(
        study=study_context,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    if readiness.planning_prerequisites_missing:
        missing = list(readiness.planning_prerequisites_missing)
        eligibility_required = "cohort_eligibility" in missing
        _reject(
            {
                "error": (
                    "cohort_eligibility_confirmation_required"
                    if eligibility_required
                    else "study_setup_incomplete"
                ),
                "message": (
                    "Confirm one cohort eligibility option before generating the candidate plan."
                    if eligibility_required
                    else "Bind the research question and data source before generating the candidate plan."
                ),
                "owner": (
                    "easyicu.webserver.pi_copilot.cohort_eligibility"
                    if eligibility_required
                    else "easyicu.webserver.pi_copilot.workflow"
                ),
                "next_action_code": readiness.next_action_code,
                "planning_prerequisites_missing": missing,
                "missing_setup_fields": list(readiness.missing_setup_fields),
            }
        )

    database = source.get("database") if isinstance(source, Mapping) else None
    prepared_manifest = dataio.prepared_export_manifest_path(Path(path).expanduser())
    budget_mode = research_pipeline_budget_mode_for_source(
        prepared_manifest=prepared_manifest,
        metadata_only_planning_authorized=request.intent == "candidate_plan",
    )
    if budget_mode == "full_reviewed" or prepared_manifest is not None:
        try:
            dataio.validate_research_pipeline_source(path, database=database)
        except dataio.ExportCohortError as exc:
            raise ResearchRunSubmissionError(exc.detail) from exc

    llm_provider = request.provider.strip() or "mock"
    external_llm_opt_in = request.external_llm_opt_in
    literature_search_authorized = request.literature_search_authorized
    if literature_search_authorized and not external_llm_opt_in:
        _reject({"error": "literature_search_authorization_scope_invalid"})
    default_credential_source = (
        "codex_user_auth"
        if provider_adapter.is_user_account_provider(llm_provider)
        else "scientific_provider"
    )
    credential_source = request.credential_source.strip() or default_credential_source
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
    compute = capabilities.validate_compute_target(
        {"compute_target": request.compute_target}
    )
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
            request_question=study_context.get("question"),
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
        planner_start_mode = request.planner_start_mode
        plan_revision_source_run_id = request.plan_revision_source_run_id.strip()
        execution_resume_source_run_id = request.execution_resume_source_run_id.strip()
        development_resume_source_job_id = ""
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

    if authorize is not None:
        authorize()

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
    return ResearchRunSubmissionReceipt(
        job_id=job.id,
        kind=job.kind,
        status=job.status,
        study_context_id=study_context_id,
        study_context_revision=int(synced_context.get("revision") or 0),
        budget_mode=budget_mode,
        planner_start_mode=planner_start_mode,
        resume_source_job_id=development_resume_source_job_id or None,
        audit_warning=audit_warning,
    )


__all__ = [
    "ResearchRunSubmissionError",
    "ResearchRunSubmissionReceipt",
    "ResearchRunSubmissionRequest",
    "provider_environment_for_agent_run",
    "research_pipeline_budget_mode_for_source",
    "server_research_pipeline_budget_mode",
    "submit_research_run",
]
