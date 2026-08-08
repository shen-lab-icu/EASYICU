"""Capability-scoped EasyICU tools exposed to Pi AgentSession."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from fastapi import HTTPException

from easyicu.webserver import agent_runs, capabilities, jobs, sources, study_contexts

from .contracts import (
    AuthorityBinding,
    PiCopilotError,
    PiToolResult,
    ToolExecutionContext,
)
from .projections import (
    bounded_json_projection,
    path_digest,
    project_artifacts,
    project_capabilities,
    project_job,
    project_run_row,
    project_study_context,
    reject_sensitive_message,
)

READ_TOOLS = frozenset(
    {
        "easyicu_workspace_status",
        "easyicu_inspect_context",
        "easyicu_inspect_plan",
        "easyicu_inspect_capability",
        "easyicu_inspect_run",
        "easyicu_inspect_step",
        "easyicu_inspect_validation",
        "easyicu_list_artifacts",
        "easyicu_inspect_evidence",
        "easyicu_explain_blocker",
        "easyicu_resume",
    }
)
CONTROL_TOOLS = frozenset(
    {
        "easyicu_update_study_context",
        "easyicu_run",
        "easyicu_cancel",
        "easyicu_request_replan",
    }
)
ALLOWED_TOOLS = READ_TOOLS | CONTROL_TOOLS


def _result(
    context: ToolExecutionContext,
    *,
    status: str,
    code: str,
    summary: str,
    owner: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return PiToolResult(
        status=status,
        code=code,
        summary=summary[:2000],
        owner=owner,
        details=bounded_json_projection(details or {}),
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")


def _require_args(
    params: Mapping[str, Any],
    *,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(params) - allowed_set)
    if unknown:
        raise PiCopilotError(
            "pi_tool_unknown_arguments",
            "The EasyICU tool rejected unknown arguments.",
            details={"fields": unknown},
        )
    missing = sorted(
        key
        for key in required
        if key not in params or not str(params.get(key) or "").strip()
    )
    if missing:
        raise PiCopilotError(
            "pi_tool_arguments_required",
            "The EasyICU tool is missing required arguments.",
            details={"fields": missing},
        )


def _bound_context(binding: AuthorityBinding) -> Optional[Dict[str, Any]]:
    if binding.study_context_id:
        try:
            return study_contexts.get_context(binding.study_context_id)
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "study_context_invalid"),
                "The bound StudyContext could not be loaded.",
                details=exc.detail,
            ) from exc
    return study_contexts.get_active_context()


def _run_rows(context: ToolExecutionContext) -> Sequence[Dict[str, Any]]:
    history = agent_runs.list_run_history(
        study_id=context.session.binding.study_context_id,
        limit=50,
    )
    return [
        row
        for row in (history.get("runs") or [])
        if isinstance(row, dict)
    ]


def _select_run(
    context: ToolExecutionContext, requested_run_id: Any = None
) -> Optional[Dict[str, Any]]:
    requested = str(requested_run_id or "").strip()
    preferred = requested or str(context.session.binding.run_id or "").strip()
    rows = _run_rows(context)
    if preferred:
        return next((row for row in rows if row.get("run_id") == preferred), None)
    return rows[0] if rows else None


def _run_review(row: Mapping[str, Any]) -> Dict[str, Any]:
    # project_dir remains inside this host process. It is never put in a
    # PiToolResult or returned to the model/browser.
    review = agent_runs.read_run_review(str(row.get("project_dir") or ""))
    if not review.get("ok"):
        raise PiCopilotError(
            str(review.get("error") or "pi_run_review_unavailable"),
            "The selected EasyICU run review is unavailable.",
            details={
                key: review.get(key)
                for key in ("error", "artifact")
                if review.get(key) is not None
            },
        )
    return review


def _plan_projection(payload: Mapping[str, Any]) -> Dict[str, Any]:
    steps = payload.get("steps")
    steps = steps if isinstance(steps, list) else []
    projected_steps = []
    for index, raw in enumerate(steps[:50]):
        row = raw if isinstance(raw, Mapping) else {}
        projected_steps.append(
            {
                key: row.get(key)
                for key in (
                    "id",
                    "step_id",
                    "title",
                    "intent",
                    "method",
                    "status",
                    "depends_on",
                    "evidence_ids",
                    "output_type",
                )
                if row.get(key) is not None
            }
            | {"position": index + 1}
        )
    return bounded_json_projection(
        {
            key: payload.get(key)
            for key in (
                "run_id",
                "study_id",
                "research_question",
                "provider",
                "execution",
                "status",
            )
            if payload.get(key) is not None
        }
        | {"step_count": len(steps), "steps": projected_steps}
    )


def _workspace_status(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    study = _bound_context(context.session.binding)
    registry = sources.load_registry()
    active_path = registry.get("active_path")
    active_job = None
    if study and study.get("active_job_id"):
        job = jobs.MANAGER.get(str(study["active_job_id"]))
        active_job = project_job(job.snapshot() if job else None)
    rows = _run_rows(context)
    details = {
        "study": project_study_context(study),
        "active_export": {
            "present": bool(active_path),
            "path_digest": path_digest(active_path),
            "source_count": len(registry.get("sources") or []),
        },
        "active_job": active_job or {"present": False},
        "latest_run": project_run_row(rows[0]) if rows else {"present": False},
        "scientific_authority": "EasyICU",
        "pi_session_role": "ux_state_only",
    }
    return _result(
        context,
        status="ok",
        code="easyicu_workspace_status_ready",
        summary="Loaded the current EasyICU study, job, and run status without patient rows or filesystem paths.",
        owner="easyicu.webserver.study_contexts",
        details=details,
    )


def _inspect_context(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    study = _bound_context(context.session.binding)
    if not study:
        return _result(
            context,
            status="not_found",
            code="study_context_not_found",
            summary="No typed StudyContext is currently bound to this Pi session.",
            owner="easyicu.webserver.study_contexts",
        )
    return _result(
        context,
        status="ok",
        code="study_context_projected",
        summary=f"Loaded StudyContext revision {int(study.get('revision') or 0)} through the PHI-safe projection.",
        owner="easyicu.webserver.study_contexts",
        details={"study": project_study_context(study)},
    )


def _inspect_capability(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    return _result(
        context,
        status="ok",
        code="capability_policy_projected",
        summary="Loaded the current EasyICU capability policy without credential values or private paths.",
        owner="easyicu.webserver.capabilities",
        details=project_capabilities(capabilities.capability_status()),
    )


def _inspect_run(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id", "job_id"))
    job_id = str(
        params.get("job_id")
        or context.session.binding.active_job_id
        or ""
    ).strip()
    if job_id:
        job = jobs.MANAGER.get(job_id)
        if job:
            return _result(
                context,
                status="ok",
                code="easyicu_job_status_projected",
                summary=f"EasyICU job {job_id} is {job.status}.",
                owner="easyicu.webserver.jobs",
                details={"job": project_job(job.snapshot())},
            )
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_run_not_found",
            summary="No matching persisted EasyICU run was found.",
            owner="easyicu.webserver.agent_runs",
        )
    return _result(
        context,
        status="ok",
        code="easyicu_run_status_projected",
        summary=f"Loaded the bounded status for EasyICU run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details={"run": project_run_row(row)},
    )


def _inspect_plan(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_plan_not_found",
            summary="No persisted EasyICU run with a plan artefact was found.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    plan = (review.get("artifact_payloads") or {}).get("agent_plan.json")
    if not isinstance(plan, Mapping):
        return _result(
            context,
            status="not_found",
            code="easyicu_plan_artifact_missing",
            summary=f"Run {row.get('run_id')} does not have an inspectable plan artefact.",
            owner="easyicu.webserver.agent_runs",
        )
    projected = _plan_projection(plan)
    return _result(
        context,
        status="ok",
        code="easyicu_plan_projected",
        summary=f"Loaded {projected.get('step_count', 0)} bounded plan steps from run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details={"plan": projected},
    )


def _inspect_step(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id", "step_id"), required=("step_id",))
    plan_result = _inspect_plan(
        context, {"run_id": params.get("run_id")} if params.get("run_id") else {}
    )
    if plan_result.get("status") != "ok":
        return plan_result
    plan = (plan_result.get("details") or {}).get("plan") or {}
    step_id = str(params["step_id"])
    step = next(
        (
            row
            for row in (plan.get("steps") or [])
            if str(row.get("step_id") or row.get("id") or "") == step_id
        ),
        None,
    )
    if not step:
        return _result(
            context,
            status="not_found",
            code="easyicu_plan_step_not_found",
            summary=f"No plan step named {step_id!r} exists in the selected bounded plan.",
            owner="easyicu.webserver.agent_runs",
        )
    return _result(
        context,
        status="ok",
        code="easyicu_plan_step_projected",
        summary=f"Loaded the bounded contract for plan step {step_id}.",
        owner="easyicu.webserver.agent_runs",
        details={"step": step},
    )


def _inspect_validation(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_validation_not_found",
            summary="No matching EasyICU run was found for validation inspection.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    details = bounded_json_projection(
        {
            "run_id": row.get("run_id"),
            "gate": review.get("gate") or {},
            "readiness": review.get("readiness") or {},
            "signed": bool(review.get("signed")),
            "signoff_stale": bool(review.get("signoff_stale")),
        }
    )
    return _result(
        context,
        status="ok",
        code="easyicu_validation_projected",
        summary=f"Loaded gate and readiness status for run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details=details,
    )


def _list_artifacts(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_artifacts_not_found",
            summary="No matching EasyICU run was found for artefact inspection.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    artifacts = project_artifacts(review.get("artifacts") or [])
    return _result(
        context,
        status="ok",
        code="easyicu_artifacts_projected",
        summary=f"Listed {len(artifacts)} whitelisted artefacts for run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details={"run_id": row.get("run_id"), "artifacts": artifacts},
    )


def _inspect_evidence(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_evidence_not_found",
            summary="No matching EasyICU run was found for evidence inspection.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    ledger = (review.get("artifact_payloads") or {}).get("evidence_ledger.json")
    if not isinstance(ledger, Mapping):
        return _result(
            context,
            status="not_found",
            code="easyicu_evidence_ledger_missing",
            summary=f"Run {row.get('run_id')} does not have an evidence ledger.",
            owner="easyicu.webserver.agent_runs",
        )
    provider = ledger.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    strict = ledger.get("strict_evidence_audit")
    strict = strict if isinstance(strict, Mapping) else {}
    numeric = ledger.get("numeric_evidence_audit")
    numeric = numeric if isinstance(numeric, Mapping) else {}
    privacy = ledger.get("privacy")
    privacy = privacy if isinstance(privacy, Mapping) else {}
    artifacts = ledger.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    details = {
        "run_id": ledger.get("run_id"),
        "run_type": ledger.get("run_type"),
        "status": ledger.get("status"),
        "artifacts": project_artifacts(
            row for row in artifacts if isinstance(row, Mapping)
        ),
        "provider": {
            key: provider.get(key)
            for key in (
                "provider",
                "external",
                "provider_gate",
                "canonical_opt_in_passed",
                "per_run_opt_in",
                "external_calls",
                "mock_calls",
            )
            if provider.get(key) is not None
        },
        "strict_evidence_audit": {
            key: strict.get(key)
            for key in (
                "passed",
                "claims_passed",
                "sentences_passed",
                "claim_count",
                "sentence_count",
            )
            if strict.get(key) is not None
        },
        "numeric_evidence_audit": {
            key: numeric.get(key)
            for key in ("passed", "status", "checked_values", "issue_count")
            if numeric.get(key) is not None
        },
        "privacy": {
            key: privacy.get(key)
            for key in ("passed", "status", "uploads", "row_level_data")
            if privacy.get(key) is not None
        },
    }
    return _result(
        context,
        status="ok",
        code="easyicu_evidence_projected",
        summary=f"Loaded bounded evidence and audit status for run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details=details,
    )


def _explain_blocker(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id", "job_id"))
    run_or_job = _inspect_run(context, params)
    details = run_or_job.get("details") or {}
    job = details.get("job") if isinstance(details.get("job"), Mapping) else {}
    if job and job.get("status") in {"failed", "cancelled"}:
        code = str(job.get("error_code") or job.get("cancel_reason") or job["status"])
        return _result(
            context,
            status="blocked",
            code=code,
            summary=f"The active EasyICU job is {job.get('status')}; the JobManager owner reported {code}.",
            owner="easyicu.webserver.jobs",
            details={"job": job},
        )
    row = _select_run(context, params.get("run_id"))
    if row:
        review = _run_review(row)
        gate = review.get("gate") or {}
        readiness = review.get("readiness") or {}
        if gate.get("status") == "blocked" or readiness.get("status") not in {
            "ready",
            "signed",
        }:
            code = str(
                gate.get("reason")
                or readiness.get("reason")
                or "easyicu_readiness_blocked"
            )
            return _result(
                context,
                status="blocked",
                code=code,
                summary=f"EasyICU run {row.get('run_id')} is held by its gate/readiness owner: {code}.",
                owner="easyicu.webserver.agent_runs",
                details={
                    "run_id": row.get("run_id"),
                    "gate_status": gate.get("status"),
                    "gate_reason": gate.get("reason"),
                    "readiness_status": readiness.get("status"),
                },
            )
    return _result(
        context,
        status="ok",
        code="easyicu_no_active_blocker",
        summary="No active job failure or persisted run gate blocker was found.",
        owner="easyicu.webserver.pi_copilot",
    )


_STUDY_SETUP_FIELDS = frozenset(
    {
        "title",
        "question",
        "purpose",
        "cohort",
        "modules",
        "outcome",
        "time_window",
        "comparator",
        "export_format",
        "analysis_goal",
        "confirmations",
        "bind_active_export",
    }
)


def _update_study_context(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Persist conversational setup through the existing typed owner."""

    _require_args(params, allowed=_STUDY_SETUP_FIELDS)
    if "configure" not in context.allowed_actions:
        return _result(
            context,
            status="blocked",
            code="pi_action_authorization_required",
            summary=(
                "Saving study setup requires the user to grant Configure for "
                "this message. The proposed values were not persisted."
            ),
            owner="easyicu.webserver.pi_copilot",
        )
    binding = context.session.binding
    current = _bound_context(binding)
    if binding.study_context_id and current is None:
        return _result(
            context,
            status="not_found",
            code="study_context_not_found",
            summary="The bound StudyContext no longer exists; no replacement context was created.",
            owner="easyicu.webserver.study_contexts",
        )
    if current and current.get("active_job_id"):
        return _result(
            context,
            status="blocked",
            code="study_context_active_job_conflict",
            summary="Study setup cannot change while its authoritative EasyICU job is active.",
            owner="easyicu.webserver.study_contexts",
        )

    patch = {
        key: params[key]
        for key in _STUDY_SETUP_FIELDS - {"bind_active_export"}
        if key in params
    }
    if current and current.get("id"):
        patch["id"] = current["id"]
    for field in (
        "title",
        "question",
        "purpose",
        "outcome",
        "comparator",
        "analysis_goal",
    ):
        if patch.get(field):
            reject_sensitive_message(str(patch[field]))

    if params.get("bind_active_export"):
        registry = sources.load_registry()
        active_path = str(registry.get("active_path") or "").strip()
        if not active_path:
            return _result(
                context,
                status="blocked",
                code="no_active_export",
                summary="No validated active EasyICU export is available to bind.",
                owner="easyicu.webserver.sources",
            )
        source = next(
            (
                row
                for row in (registry.get("sources") or [])
                if isinstance(row, Mapping) and row.get("path") == active_path
            ),
            {},
        )
        patch["data_source"] = {
            "path": active_path,
            "label": source.get("label") or "active EasyICU export",
            "database": source.get("database") or "",
        }
    if not patch:
        raise PiCopilotError(
            "pi_tool_arguments_required",
            "At least one typed study-setup field is required.",
        )

    try:
        updated = study_contexts.upsert_context(
            patch,
            active=True,
            expected_revision=(
                int(current.get("revision") or 0) if current else None
            ),
            require_revision=bool(current),
            lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        return _result(
            context,
            status="blocked",
            code=str(exc.detail.get("error") or "study_context_update_blocked"),
            summary="The typed StudyContext owner rejected the proposed setup update.",
            owner="easyicu.webserver.study_contexts",
            details={
                key: exc.detail.get(key)
                for key in (
                    "error",
                    "field",
                    "fields",
                    "expected_revision",
                    "current_revision",
                )
                if exc.detail.get(key) is not None
            },
        )
    return _result(
        context,
        status="ok",
        code="study_context_updated",
        summary=(
            f"Saved typed StudyContext revision {int(updated.get('revision') or 0)}. "
            "Rebind this Pi session before the next message."
        ),
        owner="easyicu.webserver.study_contexts",
        details={
            "study": project_study_context(updated),
            "rebind_required": True,
        },
    )


def _run(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_type", "llm_provider"))
    if "run" not in context.allowed_actions:
        return _result(
            context,
            status="blocked",
            code="pi_action_authorization_required",
            summary="Starting an EasyICU run requires the user to grant Run for this message.",
            owner="easyicu.webserver.pi_copilot",
        )
    run_type = str(params.get("run_type") or "preflight").strip().lower()
    if run_type != "preflight":
        return _result(
            context,
            status="blocked",
            code="pi_full_run_requires_dedicated_confirmation",
            summary=(
                "The first Pi shell slice can start only the deterministic local "
                "preflight. A scientific provider run needs its existing dedicated "
                "provider and per-run confirmation flow."
            ),
            owner="easyicu.webserver.provider_gate",
        )
    study = _bound_context(context.session.binding)
    if not study or not study.get("id"):
        return _result(
            context,
            status="blocked",
            code="study_context_required",
            summary="Create and bind a typed StudyContext before starting an EasyICU run.",
            owner="easyicu.webserver.study_contexts",
        )
    # Import lazily to keep the route-composition module out of this package's
    # import graph. The function remains the one existing run submission path;
    # this adapter does not reconstruct its validation or JobManager behavior.
    from easyicu.webserver.routes.agent import jobs_agent_run

    try:
        submitted = jobs_agent_run(
            {
                "study_context_id": study["id"],
                "question": study.get("question"),
                "run_type": "preflight",
                "llm_provider": "mock",
                "external_llm_opt_in": False,
            }
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "easyicu_run_submission_blocked"),
            summary="The existing EasyICU run submission boundary rejected the request.",
            owner="easyicu.webserver.routes.agent",
            details={
                key: detail.get(key)
                for key in ("error", "blockers", "blocker_codes", "job_id")
                if detail.get(key) is not None
            },
        )
    return _result(
        context,
        status="ok",
        code="easyicu_run_submitted",
        summary=f"Submitted deterministic EasyICU preflight job {submitted.get('job_id')}.",
        owner="easyicu.webserver.routes.agent",
        details={
            key: submitted.get(key)
            for key in (
                "job_id",
                "kind",
                "status",
                "study_context_id",
                "study_context_revision",
            )
            if submitted.get(key) is not None
        },
    )


def _resume(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("job_id", "run_id"))
    job_id = str(
        params.get("job_id")
        or context.session.binding.active_job_id
        or ""
    ).strip()
    if job_id:
        job = jobs.MANAGER.get(job_id)
        if job:
            return _result(
                context,
                status="ok",
                code="easyicu_job_reattached",
                summary=f"Reattached to EasyICU job {job_id}; its authoritative status is {job.status}.",
                owner="easyicu.webserver.jobs",
                details={"job": project_job(job.snapshot())},
            )
    if params.get("run_id") or context.session.binding.run_id:
        return _result(
            context,
            status="blocked",
            code="scientific_resume_not_supported",
            summary=(
                "The Pi conversation can resume, but the current EasyICU Web "
                "scientific pipeline has no public crash-resume owner contract. "
                "No replacement resume path was invented."
            ),
            owner="easyicu.research_agent.pipeline",
        )
    return _result(
        context,
        status="not_found",
        code="easyicu_resume_target_not_found",
        summary="No active EasyICU job or persisted run was bound to this Pi session.",
        owner="easyicu.webserver.jobs",
    )


def _cancel(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("job_id",))
    if "cancel" not in context.allowed_actions:
        return _result(
            context,
            status="blocked",
            code="pi_action_authorization_required",
            summary="Cancelling an EasyICU job requires the user to grant Cancel for this message.",
            owner="easyicu.webserver.pi_copilot",
        )
    job_id = str(
        params.get("job_id")
        or context.session.binding.active_job_id
        or ""
    ).strip()
    job = jobs.MANAGER.get(job_id) if job_id else None
    if not job:
        return _result(
            context,
            status="not_found",
            code="easyicu_job_not_found",
            summary="The specifically bound EasyICU job no longer exists in JobManager.",
            owner="easyicu.webserver.jobs",
        )
    accepted = job.request_cancel("pi_copilot_user_authorized")
    return _result(
        context,
        status="ok" if accepted else "blocked",
        code=(
            "easyicu_job_cancel_requested"
            if accepted
            else "easyicu_job_already_terminal"
        ),
        summary=(
            f"Cooperative cancellation was requested for EasyICU job {job_id}."
            if accepted
            else f"EasyICU job {job_id} was already terminal."
        ),
        owner="easyicu.webserver.jobs",
        details={"job": project_job(job.snapshot())},
    )


def _request_replan(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("reason",), required=("reason",))
    return _result(
        context,
        status="blocked",
        code="scientific_replan_not_supported",
        summary=(
            "EasyICU does not yet expose a public replan owner to the Web shell. "
            "The request was not converted into direct plan mutation."
        ),
        owner="easyicu.research_agent.current_plan_authority",
        details={"reason_digest": path_digest(params.get("reason"))},
    )


_DISPATCH = {
    "easyicu_workspace_status": _workspace_status,
    "easyicu_inspect_context": _inspect_context,
    "easyicu_inspect_plan": _inspect_plan,
    "easyicu_inspect_capability": _inspect_capability,
    "easyicu_inspect_run": _inspect_run,
    "easyicu_inspect_step": _inspect_step,
    "easyicu_inspect_validation": _inspect_validation,
    "easyicu_list_artifacts": _list_artifacts,
    "easyicu_inspect_evidence": _inspect_evidence,
    "easyicu_explain_blocker": _explain_blocker,
    "easyicu_update_study_context": _update_study_context,
    "easyicu_run": _run,
    "easyicu_resume": _resume,
    "easyicu_cancel": _cancel,
    "easyicu_request_replan": _request_replan,
}


def execute_tool(
    name: str,
    arguments: Mapping[str, Any],
    context: ToolExecutionContext,
) -> Dict[str, Any]:
    """Execute exactly one registered tool through its existing EasyICU owner."""

    tool_name = str(name or "").strip()
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        raise PiCopilotError(
            "pi_tool_unknown",
            "The Pi sidecar requested an unregistered EasyICU tool.",
            details={"tool": tool_name},
        )
    if not isinstance(arguments, Mapping):
        raise PiCopilotError(
            "pi_tool_arguments_invalid",
            "EasyICU tool arguments must be an object.",
        )
    return handler(context, arguments)


__all__ = [
    "ALLOWED_TOOLS",
    "CONTROL_TOOLS",
    "READ_TOOLS",
    "execute_tool",
]
