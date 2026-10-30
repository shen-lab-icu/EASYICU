"""Reconstruct interrupted development research from its original authority.

The durable command carries references, never a replacement plan, review or
credential. Existing checkpoint owners still validate scientific artifacts and
execute work. Missing checkpoints or accounting are terminal recovery refusals.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from easyicu.webserver import study_contexts
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

SCHEMA = "easyicu.research-auto-resume/1"
_PROVIDER_IDENTITY = (
    "provider", "model", "credential_fingerprint", "endpoint_fingerprint",
    "credential_header", "base_url_endpoint", "session_binding_sha256",
)


def _refuse(reason: str) -> None:
    raise ResearchPipelineRunError(
        f"research_auto_resume_{reason}",
        "The interrupted research could not be continued under its original authority.",
    )


def _value(request: Any, name: str, default: Any = None) -> Any:
    return request.get(name, default) if isinstance(request, Mapping) else getattr(request, name, default)


def prepare_descriptor(
    request: Any = None, study: Mapping[str, Any] | None = None,
    runner_kwargs: Mapping[str, Any] | None = None, *,
    project_root: str | Path | None = None, credential_source: str | None = None,
    budget_mode: str | None = None, external_llm_opt_in: bool | None = None,
    execution_resume_source_run_id: str = "", literature_search_authorized: bool = False,
) -> dict[str, Any]:
    """Freeze only launch references after the normal submission gates pass."""
    kwargs = dict(runner_kwargs or {})
    study = study or kwargs.get("study_context") or {}
    opt_in = external_llm_opt_in if external_llm_opt_in is not None else _value(request, "external_llm_opt_in", False)
    if opt_in is not True:
        _refuse("opt_in_required")
    identity = str(study.get("id") or "")
    root = project_root or kwargs.get("project_root")
    mode = budget_mode or kwargs.get("budget_mode")
    source = credential_source or kwargs.get("credential_source")
    if not identity or not root or mode not in {"planner_canary", "full_reviewed"}:
        _refuse("launch_binding_invalid")
    if str(execution_resume_source_run_id).startswith("append-"):
        if mode != "full_reviewed":
            _refuse("reviewed_budget_required")
        from .research_append_dispatch import prepare_append_descriptor

        return prepare_append_descriptor(study=study, project_root=root, run_id=execution_resume_source_run_id)
    if source not in {"pi_verified", "codex_user_auth"}:
        _refuse("credential_source_invalid")
    return {
        "schema_version": SCHEMA,
        "study_context_id": identity,
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
        "project_root": str(Path(root).expanduser().resolve()),
        "credential_source": source,
        "budget_mode": mode,
        "external_llm_opt_in": True,
        "execution_resume_source_run_id": str(execution_resume_source_run_id or kwargs.get("execution_resume_source_run_id") or ""),
        "literature_search_authorized": bool(literature_search_authorized or kwargs.get("literature_search_authorized", False)),
    }


def _check_cancel(job: Any) -> None:
    checker = getattr(job, "is_cancel_requested", None)
    if (callable(checker) and checker()) or getattr(job, "cancel_requested", False):
        _refuse("cancelled")


def _owned_wrapper(descriptor: Mapping[str, Any], binding: Mapping[str, Any]) -> Path:
    root = Path(str(descriptor["project_root"])).expanduser().resolve()
    wrapper = Path(str(binding.get("wrapper_dir") or ""))
    study_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(descriptor["study_context_id"])).strip("-.")[:96] or "study"
    expected_parent = root / study_slug
    if (not wrapper.is_absolute() or wrapper.is_symlink() or wrapper.parent.is_symlink()
            or wrapper.resolve().parent != expected_parent.resolve()
            or not re.fullmatch(r"run_[A-Za-z0-9][A-Za-z0-9_-]{0,79}", wrapper.name)):
        _refuse("workspace_binding_invalid")
    return wrapper.resolve()


def _authority(descriptor: Mapping[str, Any], job: Any) -> tuple[dict, Any, Path, dict]:
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.profiles import is_paper_facing_profile
    from easyicu.webserver import agent_review_recovery, settings

    _check_cancel(job)
    if descriptor.get("schema_version") != SCHEMA or descriptor.get("external_llm_opt_in") is not True:
        _refuse("descriptor_invalid")
    if settings.load_settings().get("ai_enabled") is not True:
        _refuse("opt_in_revoked")
    current = study_contexts.get_context(str(descriptor.get("study_context_id") or ""))
    if current is None or study_contexts.scientific_configuration_sha256(current) != descriptor.get("scientific_configuration_sha256"):
        _refuse("configuration_superseded")
    binding = descriptor.get("provider_binding")
    if not isinstance(binding, dict):
        _refuse("provider_binding_missing")
    wrapper = _owned_wrapper(descriptor, binding)
    seed = agent_review_recovery.load_recovery_seed(wrapper)
    if seed is None or seed.schema_version != "easyicu.web-review-recovery-seed/4":
        _refuse("checkpoint_seed_missing")
    if (seed.scientific_configuration_sha256 != descriptor["scientific_configuration_sha256"]
            or seed.study.get("id") != descriptor["study_context_id"]
            or seed.credential_source != descriptor.get("credential_source")
            or seed.budget_mode != descriptor.get("budget_mode")):
        _refuse("checkpoint_binding_changed")
    config = PipelineConfig.from_recovery_payload(seed.pipeline_config, expected_digest=seed.pipeline_config_sha256)
    if is_paper_facing_profile(config.submission_profile_name):
        _refuse("formal_profile_requires_protocol")
    if Path(config.workdir).resolve() != wrapper / "pipeline":
        _refuse("checkpoint_binding_changed")
    for name in ("hard_stop_ledger_path", "hard_stop_task_id", "hard_stop_declaration_sha256"):
        if str(binding.get(name) or "") != str(getattr(seed, name)):
            _refuse("budget_binding_changed")
    ledger_path = Path(seed.hard_stop_ledger_path)
    if ledger_path.is_symlink() or ledger_path.parent.is_symlink() or not ledger_path.is_file():
        _refuse("budget_ledger_missing")
    from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger
    from easyicu.webserver import provider_adapter

    ledger = load_provider_hard_stop_ledger(ledger_path)
    if (ledger.get("task_order") != [seed.hard_stop_task_id]
            or ledger.get("batch_id") != seed.hard_stop_task_id
            or ledger.get("declaration_sha256") != seed.hard_stop_declaration_sha256
            or ledger.get("limits") != provider_adapter.web_research_agent_hard_stop_limits(seed.budget_mode).canonical_payload()):
        _refuse("budget_binding_changed")
    rows = ledger.get("tasks") or []
    if len(rows) != 1 or rows[0].get("status") not in {"running", "paused"}:
        _refuse("budget_task_terminal")
    return dict(current), seed, wrapper, rows[0]


def _provider_environment(seed: Any) -> Mapping[str, str] | None:
    from easyicu.webserver.provider_gate import resolve_provider_gate
    from easyicu.webserver.research_run_submission import provider_environment_for_agent_run

    # The account session is deliberately not serialized. Recovery cannot
    # substitute another logged-in session or a different credential route.
    if seed.credential_source == "codex_user_auth":
        _refuse("account_session_unavailable")
    provider = str(seed.provider_meta.get("provider") or "")
    resolve_provider_gate(run_type="full", llm_provider=provider, external_llm_opt_in=True, ai_enabled=True)
    return provider_environment_for_agent_run(
        credential_source=seed.credential_source, engine="research_agent_pipeline",
        run_type="full", external_llm_opt_in=True, llm_provider=provider,
    )


def _approved_checkpoint(seed: Any, wrapper: Path, run_id: str) -> None:
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
    from easyicu.webserver.agent_review_recovery import get_record

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", run_id):
        _refuse("review_run_invalid")
    record = get_record(run_id)
    if record is None or Path(record.wrapper_dir).resolve() != wrapper:
        _refuse("review_record_missing")
    for name in ("scientific_configuration_sha256", "pipeline_config_sha256", "hard_stop_ledger_path", "hard_stop_task_id", "hard_stop_declaration_sha256"):
        if getattr(record, name) != getattr(seed, name):
            _refuse("review_binding_changed")
    checkpoint = load_checkpoint(wrapper / "pipeline" / run_id / "human_review_checkpoint.json", require_pending=False)
    if checkpoint.state not in {"approved_pending_execution", "executing"}:
        _refuse("approved_checkpoint_required")
    if (checkpoint.run_id != run_id or checkpoint.pipeline_config_sha256 != seed.pipeline_config_sha256
            or not checkpoint.approved_decisions
            or any(item.get("decision") != "approved" for item in checkpoint.approved_decisions)
            or not checkpoint.approved_decision_records
            or any(item.get("reviewer_identity_source") != "authenticated" for item in checkpoint.approved_decision_records)):
        _refuse("recorded_approval_required")
    config = PipelineConfig.from_recovery_payload(seed.pipeline_config, expected_digest=seed.pipeline_config_sha256)
    if checkpoint.state == "executing":
        if config.runner_kind == "host":
            from easyicu.runtime.host_execution_lease import (
                HostExecutionLeaseError,
                assert_host_execution_quiescent,
            )

            try:
                assert_host_execution_quiescent(
                    wrapper / "pipeline" / run_id / ".host-execution",
                    allow_missing=False, wait_timeout=15.0,
                )
            except HostExecutionLeaseError:
                _refuse("child_process_state_unverified")
        elif config.runner_kind != "docker":
            _refuse("child_process_state_unverified")


def _original_task(seed: Any) -> Any:
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.webserver import provider_adapter

    limits = provider_adapter.web_research_agent_hard_stop_limits(seed.budget_mode)
    ledger = ProviderHardStopLedger(
        path=Path(seed.hard_stop_ledger_path), task_ids=(seed.hard_stop_task_id,),
        limits=limits, batch_id=seed.hard_stop_task_id,
        declaration_sha256=seed.hard_stop_declaration_sha256,
        resume_existing=True, require_existing=True,
    )
    task = ledger.start_task(seed.hard_stop_task_id)
    snapshot = ledger.snapshot()
    row = snapshot["tasks"][0]
    totals = snapshot["totals"]
    if (len(row["calls"]) >= limits.max_provider_attempts_per_run
            or totals["provider_attempts"] >= limits.max_provider_attempts_per_batch
            or sum(call["accounted_tokens"] for call in row["calls"]) >= limits.max_total_tokens_per_run
            or totals["accounted_tokens"] >= limits.max_total_tokens_per_batch
            or totals["accounted_estimated_cost_usd"] >= limits.max_estimated_cost_usd_per_batch):
        _refuse("budget_exhausted")
    if row["status"] == "running":
        task.assert_active()
    elif row["status"] != "paused" or row["elapsed_seconds"] >= limits.max_wall_clock_seconds_per_task:
        _refuse("budget_exhausted")
    return task


def _verify_guarded_planner_inflight(seed: Any, task_row: Mapping[str, Any]) -> None:
    """No planner replay overlaps an unverified local or uncounted request."""
    calls = task_row.get("calls")
    if not isinstance(calls, list):
        _refuse("budget_binding_changed")
    active = [call for call in calls if isinstance(call, dict) and call.get("state") == "in_progress"]
    if not active:
        return
    if (len(active) != 1 or calls[-1] is not active[0]
            or active[0].get("role") not in {"planner", "provider_recovery_probe"}):
        _refuse("planner_provider_worker_unverified")
    from easyicu.runtime.host_execution_lease import (
        HostExecutionLeaseError, assert_host_execution_quiescent, read_host_execution_receipt,
    )
    from easyicu.webserver import metadata_provider_concurrency as slots
    from easyicu.webserver.metadata_provider_worker import identity as worker_identity

    attempt_id = str(active[0].get("attempt_id") or "")
    held = slots.get(attempt_id)
    if not isinstance(held, dict):
        _refuse("planner_provider_worker_unverified")
    ledger_path = Path(seed.hard_stop_ledger_path)
    guard = ledger_path.parent / ".metadata-provider-execution" / attempt_id
    try:
        account = slots.account_identity(seed.provider_public)
    except ValueError:
        _refuse("provider_identity_changed")
    if (held.get("task_id") != seed.hard_stop_task_id
            or held.get("account") != account or held.get("ledger_path") != str(ledger_path)
            or held.get("guard_path") != str(guard) or held.get("worker_sha256") != worker_identity()
            or not held.get("guard_nonce")):
        _refuse("planner_provider_worker_unverified")
    try:
        assert_host_execution_quiescent(guard, allow_missing=False, wait_timeout=15.0)
        receipt = read_host_execution_receipt(guard)
    except (OSError, HostExecutionLeaseError):
        _refuse("planner_provider_worker_unverified")
    if (receipt is None or receipt.get("nonce") != held["guard_nonce"]
            or receipt.get("script_sha256") != held["worker_sha256"]):
        _refuse("planner_provider_worker_unverified")
    if not slots.has_capacity(account):
        _refuse("metadata_provider_concurrency_unavailable")


def _verify_checkpoint_inflight_retry(
    wrapper: Path, checkpoint: Mapping[str, Any], task_row: Mapping[str, Any],
) -> None:
    """A guarded request alone cannot restore a later phase's retry state.

    An accepted outline checkpoint does not contain foundation responses or
    parser feedback.  Only the authenticated journal for that exact pending
    request can restore it; other unjournaled planning phases remain stopped.
    """
    from easyicu.research_agent.planning.outline_retry_journal import FILE_NAME, load_retry_journal
    from easyicu.webserver import metadata_provider_concurrency as slots

    calls = task_row.get("calls")
    if not isinstance(calls, list):
        _refuse("budget_binding_changed")
    active = [c for c in calls if isinstance(c, dict) and c.get("state") == "in_progress"]
    if not active:
        return
    if len(active) != 1:
        _refuse("planner_provider_worker_unverified")
    if active[0].get("role") == "provider_recovery_probe":
        return
    if active[0].get("role") != "planner":
        _refuse("planner_provider_worker_unverified")
    relative = checkpoint.get("path")
    if not isinstance(relative, str) or not relative:
        _refuse("planner_retry_journal_missing")
    path = wrapper / relative
    try:
        path.resolve().relative_to(wrapper.resolve())
    except (OSError, ValueError):
        _refuse("planner_retry_journal_invalid")
    journal = path if path.name == FILE_NAME else path.parent / ".runtime" / FILE_NAME
    try:
        record = load_retry_journal(journal)
    except (OSError, ValueError):
        _refuse("planner_retry_journal_missing")
    binding = record.get("binding")
    pending = record.get("pending")
    stage = binding.get("stage") if isinstance(binding, dict) else None
    if path.name != FILE_NAME and stage != "foundation":
        _refuse("planner_retry_journal_missing")
    held = slots.get(str(active[0].get("attempt_id") or ""))
    request_sha = pending.get("request_sha256") if isinstance(pending, dict) else None
    if pending is None and record.get("expected_retry_guarded") is True:
        planner_calls = sum(isinstance(call, dict) and call.get("role") == "planner" for call in calls)
        if (type(record.get("abandoned")) is int and record["abandoned"] > 0
                and record.get("accounted_planner_calls") == planner_calls):
            request_sha = record.get("expected_retry_request_sha256")
    if (not isinstance(request_sha, str) or not isinstance(held, dict)
            or request_sha != held.get("request_sha256")):
        _refuse("planner_retry_journal_invalid")


def _recover_unstarted_metadata_planner(
    descriptor: Mapping[str, Any], job: Any, wrapper: Path, seed: Mapping[str, Any],
) -> Callable[[Any], Any]:
    """Recover a pre-call seed or one guarded, conservatively held request."""
    from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger
    from easyicu.webserver import agent_pipeline_runs, provider_adapter, settings
    from easyicu.webserver.provider_gate import resolve_provider_gate
    from easyicu.webserver.provider_job_recovery import ProviderJobRecovery, RecoveryJob
    from easyicu.webserver.research_early_resume import metadata_binding

    def validate(bound_job: Any) -> tuple[dict, Any]:
        _check_cancel(bound_job)
        if (descriptor.get("schema_version") != SCHEMA
                or descriptor.get("external_llm_opt_in") is not True
                or settings.load_settings().get("ai_enabled") is not True
                or descriptor.get("budget_mode") != "planner_canary"
                or descriptor.get("credential_source") != "pi_verified"
                or descriptor.get("execution_resume_source_run_id")
                or seed.get("budget_mode") != descriptor.get("budget_mode")
                or seed.get("credential_source") != descriptor.get("credential_source")
                or seed.get("study_id") != descriptor.get("study_context_id")
                or seed.get("scientific_configuration_sha256") != descriptor.get("scientific_configuration_sha256")):
            _refuse("early_seed_binding_changed")
        current = study_contexts.get_context(str(descriptor.get("study_context_id") or ""))
        if current is None or study_contexts.scientific_configuration_sha256(current) != descriptor.get("scientific_configuration_sha256"):
            _refuse("configuration_superseded")
        binding = descriptor.get("provider_binding")
        if not isinstance(binding, dict) or _owned_wrapper(descriptor, binding) != wrapper:
            _refuse("provider_binding_missing")
        ledger_path = Path(str(binding.get("hard_stop_ledger_path") or ""))
        task_id = str(binding.get("hard_stop_task_id") or "")
        ledger_runtime = ledger_path.parent
        ledger_wrapper = ledger_runtime.parent
        if (ledger_path.name != "provider_hard_stop_ledger.json"
                or ledger_runtime.name != ".runtime" or ledger_runtime.is_symlink()
                or _owned_wrapper(descriptor, {"wrapper_dir": str(ledger_wrapper)}) != ledger_wrapper
                or ledger_path.is_symlink() or not ledger_path.is_file()
                or str(ledger_path) != seed.get("hard_stop_ledger_path")
                or task_id != seed.get("hard_stop_task_id")
                or binding.get("hard_stop_declaration_sha256") != seed.get("hard_stop_declaration_sha256")
                or binding.get("hard_stop_declaration_sha256") != descriptor["scientific_configuration_sha256"]):
            _refuse("budget_binding_changed")
        ledger = load_provider_hard_stop_ledger(ledger_path)
        rows = ledger.get("tasks") or []
        if (ledger.get("task_order") != [task_id] or ledger.get("batch_id") != task_id
                or ledger.get("declaration_sha256") != descriptor["scientific_configuration_sha256"]
                or ledger.get("limits") != provider_adapter.web_research_agent_hard_stop_limits("planner_canary").canonical_payload()
                or len(rows) != 1 or rows[0].get("status") != "running"):
            _refuse("budget_binding_changed")
        calls = rows[0].get("calls")
        if not isinstance(calls, list):
            _refuse("early_provider_attempt_already_reserved")
        if calls:
            # A completed selection has no saved response here. Exactly one
            # still-unknown acquisition may continue only after its guarded
            # worker is quiescent and its remote slot remains occupied.
            active = [call for call in calls if call.get("state") == "in_progress"]
            if (len(active) != 1 or calls[-1] is not active[0]
                    or any(call.get("role") != "acquisition" for call in calls)
                    or any(call.get("state") != "failed_usage_unknown" for call in calls[:-1])):
                _refuse("early_provider_attempt_already_reserved")
            from easyicu.runtime.host_execution_lease import (
                HostExecutionLeaseError, assert_host_execution_quiescent, read_host_execution_receipt,
            )
            from easyicu.webserver import metadata_provider_concurrency as slots
            from easyicu.webserver.metadata_provider_worker import identity as worker_identity

            attempt = active[0].get("attempt_id")
            held = slots.get(str(attempt or ""))
            guard = ledger_path.parent / ".metadata-provider-execution" / str(attempt or "")
            if not isinstance(held, dict):
                _refuse("early_provider_attempt_already_reserved")
            public = seed.get("provider_public")
            if not isinstance(public, dict):
                _refuse("provider_identity_changed")
            try:
                expected_account = slots.account_identity(public)
            except ValueError:
                _refuse("provider_identity_changed")
            if (held.get("task_id") != task_id
                    or held.get("account") != expected_account
                    or held.get("ledger_path") != str(ledger_path)
                    or held.get("guard_path") != str(guard)
                    or held.get("worker_sha256") != worker_identity()
                    or not held.get("guard_nonce")):
                _refuse("early_provider_attempt_already_reserved")
            try:
                assert_host_execution_quiescent(guard, allow_missing=False, wait_timeout=15.0)
                receipt = read_host_execution_receipt(guard)
            except (OSError, HostExecutionLeaseError):
                _refuse("early_provider_worker_unverified")
            if (receipt is None or receipt.get("nonce") != held["guard_nonce"]
                    or receipt.get("script_sha256") != held["worker_sha256"]):
                _refuse("early_provider_worker_unverified")
            if not slots.has_capacity(expected_account):
                _refuse("metadata_provider_concurrency_unavailable")
        source = seed.get("source")
        if not isinstance(source, dict):
            _refuse("early_source_invalid")
        source_path = str((current.get("data_source") or {}).get("path") or "")
        try:
            current_source = metadata_binding(str(source.get("database") or ""), source_path)
        except (OSError, ValueError):
            _refuse("early_source_unavailable")
        if current_source != source:
            _refuse("early_source_changed")
        original_task = SimpleNamespace(
            budget_mode="planner_canary", hard_stop_ledger_path=str(ledger_path),
            hard_stop_task_id=task_id,
            hard_stop_declaration_sha256=descriptor["scientific_configuration_sha256"],
        )
        return dict(current), original_task

    current, original_task = validate(job)
    public = seed.get("provider_public")
    if not isinstance(public, dict) or not public.get("provider"):
        _refuse("provider_identity_changed")
    provider = resolve_provider_gate(
        run_type="full", llm_provider=str(public["provider"]),
        external_llm_opt_in=True, ai_enabled=True,
    )
    original_task.credential_source = "pi_verified"
    original_task.provider_meta = provider
    environment = _provider_environment(original_task)
    client, current_public = provider_adapter.build_research_agent_provider_client(provider, environ=environment)
    if any(current_public.get(key) != public.get(key) for key in _PROVIDER_IDENTITY):
        _refuse("provider_identity_changed")
    task = _original_task(original_task)
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(seed["source"]["export_path"]), study_context=current,
        project_root=descriptor["project_root"], provider=provider,
        provider_environment=environment, credential_source="pi_verified",
        budget_mode="planner_canary", literature_search_authorized=descriptor.get("literature_search_authorized") is True,
        allow_environment_resume=False, runner_image=seed.get("runner_image"),
    )

    def resume(bound_job: Any) -> Any:
        validate(bound_job)
        task.assert_active()
        attempt = descriptor.get("_resume_attempt")
        if type(attempt) is not int or not 2 <= attempt <= 4:
            _refuse("attempt_binding_invalid")
        recovery = ProviderJobRecovery(bound_job)
        recovery.bind(wrapper_dir=wrapper, task=task, client=client)
        return runner(RecoveryJob(recovery, f"{str(bound_job.id)[:60]}-restart-{attempt}"))

    return resume


def recover_runner(descriptor: Mapping[str, Any], job: Any) -> Callable[[Any], Any]:
    """Return an exact continuation; no branch may silently launch fresh work."""
    from .research_multi_actions import SCHEMA as MULTI_SCHEMA, recover_multi_runner

    if descriptor.get("schema_version") == MULTI_SCHEMA:
        return recover_multi_runner(descriptor, job)
    from .research_append_dispatch import SCHEMA as APPEND_SCHEMA, recover_append_runner

    if descriptor.get("schema_version") == APPEND_SCHEMA:
        return recover_append_runner(descriptor, job)
    from easyicu.webserver import agent_pipeline_runs, provider_adapter, research_launch_resume
    from easyicu.webserver.provider_job_recovery import ProviderJobRecovery, RecoveryJob

    binding = descriptor.get("provider_binding")
    if isinstance(binding, dict):
        early_wrapper = _owned_wrapper(descriptor, binding)
        from easyicu.webserver.agent_review_recovery import load_recovery_seed
        from easyicu.webserver.research_early_resume import load_seed

        if load_recovery_seed(early_wrapper) is None:
            try:
                early_seed = load_seed(early_wrapper)
            except (OSError, ValueError):
                _refuse("early_seed_invalid")
            if early_seed is not None:
                return _recover_unstarted_metadata_planner(descriptor, job, early_wrapper, early_seed)
    current, seed, wrapper, task_row = _authority(descriptor, job)
    run_id = str(descriptor.get("execution_resume_source_run_id") or "")
    if run_id:
        if seed.budget_mode != "full_reviewed":
            _refuse("reviewed_budget_required")
        _approved_checkpoint(seed, wrapper, run_id)
        _original_task(seed)
        environment = _provider_environment(seed)

        def resume_approved(bound_job: Any) -> Any:
            current_now, seed_now, wrapper_now, _ = _authority(descriptor, bound_job)
            _approved_checkpoint(seed_now, wrapper_now, run_id)
            _original_task(seed_now)
            recovery = ProviderJobRecovery(bound_job)
            # The review owner reconstructs and binds the original ledger's
            # task. An existing recovery envelope suppresses fresh outage loops.
            return agent_pipeline_runs.resume_research_pipeline(
                run_id=run_id, study_context_id=descriptor["study_context_id"],
                decision="approved", reviewer="", note="", job=RecoveryJob(recovery, str(bound_job.id)),
                current_study_context=current_now, provider_environment=environment,
                require_recorded_decision=True,
            )

        return resume_approved
    if task_row["status"] != "running":
        _refuse("human_review_required")
    checkpoint = research_launch_resume.development_resume_checkpoint(
        wrapper_dir=wrapper, study_id=descriptor["study_context_id"],
        scientific_configuration_sha256=descriptor["scientific_configuration_sha256"],
    )
    if checkpoint is None:
        _refuse("planner_checkpoint_missing")
    _verify_guarded_planner_inflight(seed, task_row)
    _verify_checkpoint_inflight_retry(wrapper, checkpoint, task_row)
    environment = _provider_environment(seed)
    client, public = provider_adapter.build_research_agent_provider_client(dict(seed.provider_meta), environ=environment)
    if any(public.get(key) != seed.provider_public.get(key) for key in _PROVIDER_IDENTITY):
        _refuse("provider_identity_changed")
    task = _original_task(seed)
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str((current.get("data_source") or {}).get("path") or ""),
        study_context=current, project_root=descriptor["project_root"],
        provider=dict(seed.provider_meta), provider_environment=environment,
        credential_source=seed.credential_source, budget_mode=seed.budget_mode,
        literature_search_authorized=descriptor.get("literature_search_authorized") is True,
        development_resume_source_job_id=wrapper.name.removeprefix("run_"),
    )

    def resume_planner(bound_job: Any) -> Any:
        _authority(descriptor, bound_job)
        task.assert_active()
        recovery = ProviderJobRecovery(bound_job)
        recovery.bind(wrapper_dir=wrapper, task=task, client=client)
        # DBOS owns restart attempts. Do not reset the same-process transport
        # retry counter by starting another ProviderJobRecovery.run loop.
        attempt = descriptor.get("_resume_attempt")
        if type(attempt) is not int or not 2 <= attempt <= 4:
            _refuse("attempt_binding_invalid")
        continuation_id = f"{str(bound_job.id)[:60]}-restart-{attempt}"
        if continuation_id == wrapper.name.removeprefix("run_"):
            _refuse("workspace_binding_invalid")
        return runner(RecoveryJob(recovery, continuation_id))

    return resume_planner
