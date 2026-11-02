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
        if not kwargs.get("execution_resume_source_run_id"):
            recovery.record_continuation_refusal("research_pipeline_development_resume_checkpoint_invalid")
        return None
    try:
        runner = agent_pipeline_runs.make_research_pipeline_run_runner(**kwargs)
    except ResearchPipelineRunError as exc:
        if exc.code.startswith("research_pipeline_development_resume_"):
            recovery.record_continuation_refusal(exc.code)
        return None
    return CheckpointContinuation(checkpoint=checkpoint, run=runner)


def wrap_pipeline_runner(runner: Callable, request: Any) -> Callable:
    launch = {field.name: getattr(request, field.name) for field in fields(request)}

    def run(job: Any) -> Any:
        from easyicu.research_agent.providers.structured_retry import automatic_retry_scope

        disabled = getattr(request, "no_automatic_retries", False)
        with automatic_retry_scope(disabled=disabled):
            if recovery_for_job(job) is not None:
                return runner(job)
            recovery = ProviderJobRecovery(job)
            return recovery.run(
                runner, lambda exc: None if disabled else _continuation(recovery, launch)
            )

    return run


def bind_provider_recovery(job: Any, *, wrapper_dir: Path, task: Any, client: Any) -> None:
    from easyicu.webserver.durable_research import bind_provider_task

    bind_provider_task(job, wrapper_dir=wrapper_dir, task=task)
    recovery = recovery_for_job(job)
    if recovery is not None:
        from easyicu.webserver.research_run_usage import bind_shared_provider_task

        bind_shared_provider_task(wrapper_dir, task)
        lineage = getattr(recovery, "controlled_candidate_lineage", None)
        if lineage is not None:
            from easyicu.research_agent.canonical_json import canonical_json_bytes, canonical_sha256
            from easyicu.webserver.raw_preparation_decisions import _publish_once
            payload = {**lineage, "schema_version": "easyicu.controlled-candidate-lineage/1",
                       "task_id": task.task_id, "declaration_sha256": task.ledger.snapshot()["declaration_sha256"]}
            payload["sha256"] = canonical_sha256(payload)
            from easyicu.research_agent.authority.filesystem import AnchoredDirectory
            with AnchoredDirectory.open(wrapper_dir / ".runtime") as directory:
                name = "controlled_candidate_lineage.json"
                data = canonical_json_bytes(payload)
                try:
                    _publish_once(directory, name, data)
                except FileExistsError:
                    # The launch owner binds the raw client and then its guarded
                    # transport to the same task. Reuse only identical authority.
                    if directory.read_bytes(name, max_bytes=len(data)) != data:
                        raise ValueError("controlled_candidate_lineage_mismatch") from None
                directory.assert_still_selected()
        recovery.bind(wrapper_dir=wrapper_dir, task=task, client=client)


def existing_provider_task(job: Any) -> Any:
    recovery = recovery_for_job(job)
    return recovery.task if recovery is not None else None


def recover_review(operation: Callable) -> Callable:
    """An approval is submitted once; any continuation uses execution launch."""
    @wraps(operation)
    def run(*, job: Any, **kwargs: Any) -> Any:
        if recovery_for_job(job) is not None:
            return operation(job=job, **kwargs)
        recovery = ProviderJobRecovery(job)
        recovery.review_launch = None
        return recovery.run(
            lambda bound_job: operation(job=bound_job, **kwargs),
            lambda exc: _continuation(recovery, recovery.review_launch)
            if recovery.review_launch is not None and not getattr(recovery, "no_automatic_retries", False) else None,
        )
    return run


def bind_review_recovery(job: Any, entry: Any, provider_environment: Any) -> None:
    from easyicu.webserver.agent_review_recovery import load_recovery_seed

    recovery = recovery_for_job(job)
    if recovery is not None:
        recovery.no_automatic_retries = getattr(
            getattr(getattr(entry, "pipeline", None), "_config", None), "no_automatic_retries", False
        ) is True
    client = getattr(entry, "provider_client", None)
    if recovery is None or client is None or entry.provider_hard_stop is None:
        return
    seed = load_recovery_seed(entry.wrapper_dir)
    if seed is None:
        return
    from easyicu.webserver.durable_research import bind_provider_task

    bind_provider_task(job, wrapper_dir=entry.wrapper_dir, task=entry.provider_hard_stop)
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


def run_controlled_candidate(*, runner: Callable, request: Any, job: Any, task: Any,
                             original_job_id: str | None = None,
                             historical_worker_tree: str | None = None,
                             _successor: Any = None) -> Any:
    """Run one fresh candidate against an already validated, paused provider task.

    The caller owns authorization and must acquire the original task through
    the verified recovery seed. This function never loads or changes a saved
    pipeline checkpoint. The normal runner binds the shared ledger to its new
    wrapper and the original task continues to enforce the cumulative limits.
    """
    from easyicu.research_agent.providers.structured_retry import automatic_retry_scope

    if getattr(request, "no_automatic_retries", None) is not True:
        raise ValueError("controlled_candidate_retry_policy_required")
    if getattr(request, "allow_environment_resume", None) is not False:
        raise ValueError("controlled_candidate_environment_resume_forbidden")
    if getattr(request, "plan_change_request", None) is not None:
        raise ValueError("controlled_candidate_must_be_fresh")
    if any(getattr(request, name, "") for name in (
        "execution_resume_source_run_id", "development_resume_source_job_id",
        "plan_revision_source_run_id",
    )):
        raise ValueError("controlled_candidate_must_be_fresh")
    ledger = task.ledger.snapshot()
    if ledger.get("declaration_sha256") != study_contexts.scientific_configuration_sha256(request.study_context):
        raise ValueError("controlled_candidate_scientific_scope_changed")
    rows = [row for row in ledger.get("tasks", []) if row.get("task_id") == task.task_id]
    if len(rows) != 1 or rows[0].get("status") != ("failed" if _successor else "paused") or any(
        call.get("state") == "in_progress" for call in rows[0].get("calls", [])
    ):
        raise ValueError("controlled_candidate_original_task_not_quiescent")
    recovery = ProviderJobRecovery(job)
    recovery.task = task
    if original_job_id is not None or historical_worker_tree is not None:
        if not original_job_id or not historical_worker_tree:
            raise ValueError("controlled_candidate_lineage_incomplete")
        _historical_worker_identity(historical_worker_tree)
        recovery.controlled_candidate_lineage = {
            "original_job_id": original_job_id, "historical_worker_tree": historical_worker_tree,
            "candidate_job_id": str(job.id),
        }
        if _successor is not None:
            recovery.controlled_candidate_lineage.update(
                failed_candidate_job_id=_successor.failed_job_id,
                predecessor_failure_sha256=_successor.expected_failure_sha256,
            )
    # This is the launch boundary: resume retains all previous accounting and
    # elapsed time. A failed operation is closed by ProviderJobRecovery.
    if _successor is None:
        task.claim_controlled_candidate(str(job.id))
    else:
        task.ledger.claim_preprovider_successor(task.task_id, failed_job_id=_successor.failed_job_id,
            successor_job_id=str(job.id), expected_calls_sha256=_successor.expected_calls_sha256,
            expected_task_sha256=_successor.expected_task_sha256)
    with automatic_retry_scope(disabled=True):
        return recovery.run(runner, lambda exc: None)


def _historical_worker_identity(tree: str) -> str:
    """Hash the worker bytes in a caller-identified immutable Git tree."""
    import hashlib
    import re
    import subprocess
    if not re.fullmatch(r"[a-f0-9]{40}", tree):
        raise ValueError("controlled_candidate_historical_tree_invalid")
    root = Path(__file__).resolve().parents[3]
    try:
        kind = subprocess.run(["git", "cat-file", "-t", tree], cwd=root, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10).stdout.strip()
        if kind != b"tree":
            raise ValueError("controlled_candidate_historical_tree_invalid")
        blob = subprocess.run(["git", "show", f"{tree}:src/easyicu/webserver/metadata_provider_worker.py"],
                              cwd=root, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("controlled_candidate_historical_worker_unavailable") from exc
    if not blob or len(blob) > 1024 * 1024:
        raise ValueError("controlled_candidate_historical_worker_invalid")
    return hashlib.sha256(blob).hexdigest()


def prepare_controlled_candidate_from_original_job(*, service: Any, original_job_id: str,
                                                   historical_worker_tree: str, _successor: Any = None) -> Any:
    """Development UAT only: verify a durable host-owned original task reference.

    Does not resume the task, construct a client, dispatch, or read patient data.
    """
    from types import SimpleNamespace
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent, read_host_execution_receipt
    from easyicu.webserver import research_auto_resume
    from easyicu.webserver import metadata_provider_concurrency as slots

    record = service.store.get(original_job_id)
    if record is None or record.get("status") != "terminal":
        raise ValueError("controlled_candidate_original_job_not_terminal")
    original = service.manager.get(original_job_id)
    if original is None or original.snapshot().get("status") == "running":
        raise ValueError("controlled_candidate_original_job_not_quiescent")
    descriptor = record["payload"]
    # A new candidate inherits the accounting, never the old engine checkpoint.
    study, seed, wrapper, row = (research_auto_resume._authority(descriptor, SimpleNamespace()) if _successor is None
        else research_auto_resume._authority(descriptor, SimpleNamespace(), _allow_failed_successor=True))
    if _successor is not None:
        import hashlib
        import json
        from easyicu.research_agent.authority.filesystem import AnchoredDirectory
        from easyicu.research_agent.canonical_json import canonical_sha256
        failed_record = service.store.get(_successor.failed_job_id)
        failed_job = service.manager.get(_successor.failed_job_id)
        if (_successor.authorized is not True
                or not failed_record or failed_record.get("status") != "terminal"
                or not isinstance(failed_record.get("payload"), dict)
                or failed_record["payload"].get("study_context_id") != seed.study.get("id")
                or failed_record["payload"].get("scientific_configuration_sha256") != seed.scientific_configuration_sha256
                or failed_job is None or failed_job.snapshot().get("status") != "failed"
                or row.get("status") != "failed"
                or row.get("controlled_candidate_job_id") != _successor.failed_job_id
                or row.get("preprovider_successor_job_id") is not None
                or row.get("deferred_stage_job_id") is not None
                or not isinstance(row.get("calls"), list) or not row["calls"]
                or canonical_sha256(row.get("calls")) != _successor.expected_calls_sha256
                or canonical_sha256(row) != _successor.expected_task_sha256):
            raise ValueError("controlled_candidate_successor_not_authorized")
        failed_descriptor = failed_record["payload"]
        failed_binding = failed_descriptor.get("provider_binding")
        if (not isinstance(failed_binding, dict)
                or any(str(failed_binding.get(name) or "") != str(getattr(seed, name))
                       for name in ("hard_stop_ledger_path", "hard_stop_task_id", "hard_stop_declaration_sha256"))):
            raise ValueError("controlled_candidate_failed_binding_changed")
        failed_wrapper = research_auto_resume._owned_wrapper(failed_descriptor, failed_binding)
        if (failed_wrapper.name != "run_" + _successor.failed_job_id
                or (failed_wrapper / "pipeline").exists() or (failed_wrapper / "pipeline").is_symlink()
                or (failed_wrapper / ".runtime" / ".metadata-provider-execution").exists()
                or (failed_wrapper / ".runtime" / ".metadata-provider-execution").is_symlink()):
            raise ValueError("controlled_candidate_failed_worker_unverified")
        with AnchoredDirectory.open(failed_wrapper / ".runtime") as directory:
            lineage = json.loads(directory.read_bytes("controlled_candidate_lineage.json", max_bytes=4096))
            directory.assert_still_selected()
        digest = lineage.pop("sha256", None)
        if (digest != canonical_sha256(lineage)
                or lineage.get("schema_version") != "easyicu.controlled-candidate-lineage/1"
                or lineage.get("original_job_id") != original_job_id
                or lineage.get("candidate_job_id") != _successor.failed_job_id
                or lineage.get("historical_worker_tree") != historical_worker_tree
                or lineage.get("task_id") != seed.hard_stop_task_id
                or lineage.get("declaration_sha256") != seed.hard_stop_declaration_sha256):
            raise ValueError("controlled_candidate_failed_lineage_changed")
        with AnchoredDirectory.open(failed_wrapper / "diagnostics") as directory:
            failure = directory.read_bytes("research_pipeline_failure.json", max_bytes=16384)
            directory.assert_still_selected()
        if hashlib.sha256(failure).hexdigest() != _successor.expected_failure_sha256:
            raise ValueError("controlled_candidate_failure_evidence_changed")
        failure_payload = json.loads(failure)
        frames = failure_payload.get("traceback_frames", []) if isinstance(failure_payload, dict) else []
        frame_pairs = [(frame.get("file"), frame.get("function")) for frame in frames if isinstance(frame, dict)]
        if (not isinstance(failure_payload, dict) or failure_payload.get("status") != "failed"
                or ("research_job_recovery.py", "bind_provider_recovery") not in frame_pairs
                or ("raw_preparation_decisions.py", "_publish_once") not in frame_pairs
                or ("filesystem.py", "publish_temporary_no_clobber") not in frame_pairs):
            raise ValueError("controlled_candidate_failure_stage_invalid")
    if ((row.get("status") != "paused" or row.get("controlled_candidate_job_id") is not None)
            if _successor is None else False) or any(call.get("state") == "in_progress" for call in row.get("calls", [])):
        raise ValueError("controlled_candidate_original_task_not_quiescent")
    historical_identity = _historical_worker_identity(historical_worker_tree)
    guard_root = wrapper / ".runtime" / ".metadata-provider-execution"
    calls = row.get("calls", [])
    if not calls:
        raise ValueError("controlled_candidate_original_worker_evidence_missing")
    for call in calls:
        attempt = str(call.get("attempt_id") or "")
        if not __import__("re").fullmatch(r"[a-f0-9]{32}", attempt):
            raise ValueError("controlled_candidate_original_worker_identity_invalid")
        if slots.get(attempt) is not None:
            raise ValueError("controlled_candidate_original_worker_slot_held")
        guard = guard_root / attempt
        receipt = read_host_execution_receipt(guard)
        if receipt is None or receipt.get("script_sha256") != historical_identity:
            raise ValueError("controlled_candidate_original_worker_identity_changed")
        assert_host_execution_quiescent(guard, allow_missing=False)
        import json
        cleanup_path = guard / "worker-cleanup.json"
        if cleanup_path.is_symlink() or not cleanup_path.is_file() or cleanup_path.stat().st_size > 4096:
            raise ValueError("controlled_candidate_original_worker_cleanup_missing")
        try:
            cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("controlled_candidate_original_worker_cleanup_invalid") from None
        if (not isinstance(cleanup, dict) or cleanup.get("nonce") != receipt.get("nonce")
                or cleanup.get("worker") != receipt.get("worker") or cleanup.get("exitcode") != 0
                or receipt.get("state") != "quiesced" or receipt.get("reason") != "worker_completed"):
            raise ValueError("controlled_candidate_original_worker_cleanup_invalid")
    task = (research_auto_resume._original_task(seed) if _successor is None
            else research_auto_resume._original_task(seed, _failed_successor=True))
    return SimpleNamespace(task=task, study=study, seed=seed, original_job_id=original_job_id,
                           historical_worker_tree=historical_worker_tree, successor=_successor)


def submit_controlled_candidate_from_original_job(*, original_job_id: str, historical_worker_tree: str,
                                                  _successor: Any = None) -> Any:
    """Development UAT host entry; not a route and not checkpoint recovery."""
    from easyicu.webserver.durable_research import current_service
    from easyicu.webserver.research_run_submission import ResearchRunSubmissionRequest, submit_research_run

    service = current_service()
    if service is None:
        raise ValueError("controlled_candidate_durable_service_required")
    binding = prepare_controlled_candidate_from_original_job(service=service, original_job_id=original_job_id,
                                                             historical_worker_tree=historical_worker_tree, _successor=_successor)
    seed = binding.seed
    if seed.budget_mode != "planner_canary":
        raise ValueError("controlled_candidate_budget_mode_changed")
    request = ResearchRunSubmissionRequest(
        study_context_id=seed.study["id"], provider=str(seed.provider_meta.get("provider") or ""),
        credential_source=seed.credential_source, external_llm_opt_in=True,
        intent="candidate_plan", planner_start_mode="fresh",
    )
    return submit_research_run(request, no_automatic_retries=True, _controlled_candidate=binding)


def run_deferred_review_stage(*, runner: Callable, job: Any, task: Any,
                              expected_candidate_job_id: str) -> Any:
    """Run the distinct review stage on the original no-send budget task."""
    from easyicu.research_agent.providers.structured_retry import automatic_retry_scope

    recovery = ProviderJobRecovery(job)
    recovery.task = task
    task.claim_deferred_review_stage(str(job.id), expected_candidate_job_id=expected_candidate_job_id)
    with automatic_retry_scope(disabled=True):
        return recovery.run(runner, lambda exc: None)


def prepare_deferred_review_stage(*, service: Any, preparation_run_id: str,
                                  study_context_id: str, project_id: str,
                                  session_id: str) -> Any:
    """Prepare a no-send stage from a host-owned candidate and consumed preparation.

    This internal admission does not publish a review or read patient rows. The
    fixed stage owner rechecks the signed preparation before compiling and again
    before it may persist a reviewable recipe.
    """
    from types import SimpleNamespace
    import json
    from easyicu.research_agent.authority.filesystem import AnchoredDirectory
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.research_context.typed import parse_research_context_json
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent, read_host_execution_receipt
    from easyicu.webserver import agent_pipeline_runs, metadata_provider_concurrency, research_auto_resume
    from easyicu.webserver.durable_research_store import runtime_digest
    from easyicu.webserver.first_observed_deferred_recipe import FirstObservedDeferredReviewStage
    from easyicu.webserver.metadata_provider_worker import identity as current_worker_identity

    pending, study = agent_pipeline_runs.raw_preparation_review_owner(preparation_run_id, study_context_id)
    record = agent_pipeline_runs.get_review_recovery_record(preparation_run_id)
    if record is None or record.budget_mode != 'planner_canary':
        raise ValueError('deferred_stage_source_not_canary')
    wrapper = Path(record.wrapper_dir)
    candidate_job_id = wrapper.name.removeprefix('run_')
    command = service.store.get(candidate_job_id)
    if (not command or command.get('status') != 'terminal'
            or command.get('code_sha256') != runtime_digest()):
        raise ValueError('deferred_stage_source_engine_changed')
    _, seed, verified_wrapper, row = research_auto_resume._authority(
        command['payload'], SimpleNamespace(), allow_single_attempt_source=True)
    if (verified_wrapper != wrapper.resolve() or seed.hard_stop_task_id != record.hard_stop_task_id
            or row.get('status') != 'paused' or row.get('controlled_candidate_job_id') != candidate_job_id
            or row.get('deferred_stage_job_id') is not None
            or any(call.get('state') == 'in_progress' for call in row.get('calls', []))):
        raise ValueError('deferred_stage_source_task_unavailable')
    with AnchoredDirectory.open(wrapper / '.runtime') as directory:
        lineage = json.loads(directory.read_bytes('controlled_candidate_lineage.json', max_bytes=4096))
        directory.assert_still_selected()
    digest = lineage.pop('sha256', None)
    if (digest != canonical_sha256(lineage) or lineage.get('schema_version') != 'easyicu.controlled-candidate-lineage/1'
            or lineage.get('candidate_job_id') != candidate_job_id
            or lineage.get('task_id') != record.hard_stop_task_id
            or lineage.get('declaration_sha256') != record.hard_stop_declaration_sha256):
        raise ValueError('deferred_stage_candidate_lineage_invalid')
    permitted_workers = {_historical_worker_identity(str(lineage.get('historical_worker_tree') or '')),
                         current_worker_identity()}
    guard_root = Path(record.hard_stop_ledger_path).parent / '.metadata-provider-execution'
    for call in row.get('calls', []):
        attempt = str(call.get('attempt_id') or '')
        if not __import__('re').fullmatch(r'[a-f0-9]{32}', attempt) or metadata_provider_concurrency.get(attempt) is not None:
            raise ValueError('deferred_stage_source_worker_unverified')
        guard = guard_root / attempt
        receipt = read_host_execution_receipt(guard)
        if receipt is None or receipt.get('script_sha256') not in permitted_workers:
            raise ValueError('deferred_stage_source_worker_unverified')
        assert_host_execution_quiescent(guard, allow_missing=False)
        with AnchoredDirectory.open(guard) as directory:
            cleanup = json.loads(directory.read_bytes('worker-cleanup.json', max_bytes=4096))
            directory.assert_still_selected()
        if (cleanup.get('nonce') != receipt.get('nonce') or cleanup.get('worker') != receipt.get('worker')
                or cleanup.get('exitcode') != 0 or receipt.get('state') != 'quiesced'
                or receipt.get('reason') != 'worker_completed'):
            raise ValueError('deferred_stage_source_worker_unverified')
    authorities = [PlanReviewAuthority.model_validate(request.payload['plan_review_authority'])
                   for request in pending.requests]
    if not authorities or any(item != authorities[0] for item in authorities):
        raise ValueError('deferred_stage_review_authority_invalid')
    skeleton_sha = authorities[0].evidence_sha256.get('progressive_plan_skeleton')
    with AnchoredDirectory.open(Path(pending.run_dir)) as directory:
        raw_context = directory.read_bytes('research_context.json', max_bytes=16 * 1024 * 1024,
            expected_sha256=authorities[0].evidence_sha256.get('research_context'))
        directory.assert_still_selected()
    if not skeleton_sha or not authorities[0].evidence_sha256.get('research_context'):
        raise ValueError('deferred_stage_source_evidence_missing')
    stage = FirstObservedDeferredReviewStage(preparation_run_id, study_context_id, project_id, session_id, skeleton_sha)
    context = parse_research_context_json(raw_context)
    with stage.review_scope(context=context):
        pass  # Verify the signed host preparation; no cohort read.
    task = research_auto_resume._original_task(seed)
    return SimpleNamespace(stage=stage, task=task, candidate_job_id=candidate_job_id,
                           source_record=record, source_seed_sha256=seed.seed_sha256,
                           study=study)


def submit_deferred_review_stage(*, preparation_run_id: str, study_context_id: str,
                                 project_id: str, session_id: str) -> Any:
    """Host-only admission of the sole full-plan review stage; no approval is made."""
    from easyicu.webserver.durable_research import current_service
    from easyicu.webserver.research_run_submission import ResearchRunSubmissionRequest, submit_research_run
    service = current_service()
    if service is None:
        raise ValueError('deferred_reviewed_durable_service_required')
    binding = prepare_deferred_review_stage(service=service, preparation_run_id=preparation_run_id,
        study_context_id=study_context_id, project_id=project_id, session_id=session_id)
    source = binding.source_record
    request = ResearchRunSubmissionRequest(
        study_context_id=study_context_id, provider=str(source.provider_meta.get('provider') or ''),
        credential_source=source.credential_source, external_llm_opt_in=True,
        intent='candidate_plan', planner_start_mode='fresh',
    )
    return submit_research_run(request, no_automatic_retries=True, _deferred_binding=binding)
