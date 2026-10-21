"""Wire provider recovery to the same Planner and execution launch owners."""

from __future__ import annotations

from dataclasses import fields
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

from easyicu.webserver import research_launch_resume, study_contexts
from easyicu.webserver.provider_job_recovery import (
    CheckpointContinuation,
    ProviderJobRecovery,
    recovery_for_job,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


def _continuation(
    recovery: ProviderJobRecovery, launch: Mapping[str, Any],
) -> CheckpointContinuation | None:
    from easyicu.webserver import agent_pipeline_runs

    wrapper = recovery.wrapper_dir
    if wrapper is None:
        return None
    study = launch["study_context"]
    digest = study_contexts.scientific_configuration_sha256(study)
    current = study_contexts.get_context(study["id"])
    if current is None or study_contexts.scientific_configuration_sha256(current) != digest:
        return None
    kwargs = dict(launch)
    if recovery.execution_resume_run_id:
        kwargs.update(execution_resume_source_run_id=recovery.execution_resume_run_id,
                      development_resume_source_job_id="", plan_revision_source_run_id="", plan_change_request=None)
    if kwargs.get("execution_resume_source_run_id"):
        checkpoint = agent_pipeline_runs.execution_resume_checkpoint(
            study=study, project_root=kwargs["project_root"],
            source_run_id=kwargs["execution_resume_source_run_id"],
        )
    else:
        checkpoint = research_launch_resume.development_resume_checkpoint(
            wrapper_dir=wrapper, study_id=study["id"], scientific_configuration_sha256=digest,
        )
        kwargs.update(
            development_resume_source_job_id=wrapper.name.removeprefix("run_"),
            plan_revision_source_run_id="", plan_change_request=None,
        )
    if checkpoint is None:
        return None
    try:
        runner = agent_pipeline_runs.make_research_pipeline_run_runner(**kwargs)
    except ResearchPipelineRunError:
        return None
    return CheckpointContinuation(checkpoint=checkpoint, run=runner)


def wrap_pipeline_runner(runner: Callable, request: Any) -> Callable:
    launch = {field.name: getattr(request, field.name) for field in fields(request)}

    def run(job: Any) -> Any:
        if recovery_for_job(job) is not None:
            return runner(job)
        recovery = ProviderJobRecovery(job)
        return recovery.run(runner, lambda exc: _continuation(recovery, launch))

    return run


def bind_provider_recovery(job: Any, *, wrapper_dir: Path, task: Any, client: Any) -> None:
    recovery = recovery_for_job(job)
    if recovery is not None:
        from easyicu.webserver.research_run_usage import bind_shared_provider_task

        bind_shared_provider_task(wrapper_dir, task)
        recovery.bind(wrapper_dir=wrapper_dir, task=task, client=client)


def existing_provider_task(job: Any) -> Any:
    recovery = recovery_for_job(job)
    return recovery.task if recovery is not None else None


def recover_review(operation: Callable) -> Callable:
    """An approval is submitted once; any continuation uses execution launch."""
    @wraps(operation)
    def run(*, job: Any, **kwargs: Any) -> Any:
        recovery = ProviderJobRecovery(job)
        recovery.review_launch = None
        return recovery.run(
            lambda bound_job: operation(job=bound_job, **kwargs),
            lambda exc: _continuation(recovery, recovery.review_launch)
            if recovery.review_launch is not None else None,
        )
    return run


def bind_review_recovery(job: Any, entry: Any, provider_environment: Any) -> None:
    from easyicu.webserver.agent_review_recovery import load_recovery_seed

    recovery = recovery_for_job(job)
    client = getattr(entry, "provider_client", None)
    if recovery is None or client is None or entry.provider_hard_stop is None:
        return
    seed = load_recovery_seed(entry.wrapper_dir)
    if seed is None:
        return
    recovery.bind(wrapper_dir=entry.wrapper_dir, task=entry.provider_hard_stop, client=client)
    recovery.review_launch = {
        "study_context": entry.study, "project_root": str(entry.wrapper_dir.parent.parent),
        "export_path": (entry.study.get("data_source") or {}).get("path", ""),
        "provider": seed.provider_meta, "provider_environment": provider_environment,
        "credential_source": entry.credential_source, "budget_mode": entry.budget_mode,
        "execution_resume_source_run_id": entry.pending.run_id,
    }


class RecordedProviderStop(ResearchPipelineRunError):
    """A sealed execution failure, already projected by the pipeline owner."""


def recoverable_execution_stop(job: Any, run_dir: Path, projection: Mapping[str, Any]) -> None:
    """Recover only a failed step whose authoritative record names transport."""
    from easyicu.research_agent.authority.runtime_artifacts import (
        RunArtifactAuthorityError, current_step_records, load_run_artifact_authority,
    )
    from easyicu.research_agent.reporting.writer_stop import writer_stop

    recovery = recovery_for_job(job)
    if recovery is None or (projection.get("gate") or {}).get("status") != "blocked":
        return
    try:
        authority = load_run_artifact_authority(run_dir)
    except RunArtifactAuthorityError:
        return
    records = current_step_records((authority or {}).get("per_step_records") or [])
    failures = [row for row in records if row.get("status") not in {"ok", "skipped_dependency_failed"}]
    writer = writer_stop((projection.get("gate") or {}).get("detail"))
    if not failures and writer:
        failure = {"reason_code": writer.get("provider_failure_code", "provider_http_error"),
                   "status_code": writer.get("provider_http_status")}
    elif failures and all(row.get("status") == "execution_raised" and row.get("transient_provider_failure") for row in failures):
        failure = failures[0]["transient_provider_failure"]
    else:
        return
    code = failure.get("reason_code")
    if code == "provider_http_error" and type(failure.get("status_code")) is int and 500 <= failure["status_code"] <= 599:
        cause = RuntimeError("Recorded transient provider HTTP failure")
        cause.status_code = failure["status_code"]
    elif code == "provider_timeout":
        cause = TimeoutError("Recorded provider timeout")
    elif code == "provider_connection_failure":
        cause = ConnectionError("Recorded provider connection failure")
    else:
        return
    recovery.execution_resume_run_id = run_dir.name
    raise RecordedProviderStop("research_pipeline_execution_failed", "The provider stopped an approved execution step.") from cause
