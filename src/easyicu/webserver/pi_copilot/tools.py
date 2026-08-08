"""Capability-scoped EasyICU tools exposed to Pi AgentSession."""

from __future__ import annotations

import json
from pathlib import Path
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
    ensure_safe_projection,
    path_digest,
    project_artifacts,
    project_capabilities,
    project_job,
    project_run_row,
    project_study_context,
    reject_sensitive_message,
    stable_code,
)
from .workspace import ProjectWorkspace

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
WORKSPACE_TOOLS = frozenset(
    {
        "easyicu_load_skill",
        "easyicu_list_project_files",
        "easyicu_read_project_file",
        "easyicu_write_project_file",
        "easyicu_edit_project_file",
        "easyicu_check_project_file",
        "easyicu_preview_project_file",
    }
)
ALLOWED_TOOLS = READ_TOOLS | CONTROL_TOOLS | WORKSPACE_TOOLS


def _result(
    context: ToolExecutionContext,
    *,
    status: str,
    code: str,
    summary: str,
    owner: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = PiToolResult(
        status=status,
        code=code,
        summary=summary[:2000],
        owner=owner,
        details=bounded_json_projection(details or {}),
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")
    return ensure_safe_projection(payload)


def _workspace_result(
    context: ToolExecutionContext,
    *,
    status: str,
    code: str,
    summary: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return bounded project artifacts without applying patient projections.

    Workspace text is either model-authored or explicitly placed in the
    isolated project artifact directory.  It never reads EasyICU source-data
    paths.  Keep the JSON-line envelope bounded while preserving useful code.
    """

    safe_details = dict(details or {})
    encoded = json.dumps(safe_details, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > 30_000:
        raise PiCopilotError(
            "pi_workspace_projection_too_large",
            "The project artifact result exceeds the bounded Pi tool contract.",
            status_code=500,
            details={"bytes": len(encoded), "max_bytes": 30_000},
        )
    return PiToolResult(
        status=status,
        code=code,
        summary=summary[:2000],
        owner="easyicu.webserver.pi_copilot.workspace",
        details=safe_details,
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")


def _consume_action(
    context: ToolExecutionContext, action: str
) -> Optional[Dict[str, Any]]:
    outcome = context.grant.consume(action)
    if outcome == "granted":
        return None
    if outcome == "consumed":
        return _result(
            context,
            status="blocked",
            code="pi_action_grant_consumed",
            summary=f"The one-use {action} grant for this message was already consumed.",
            owner="easyicu.webserver.pi_copilot",
        )
    return _result(
        context,
        status="blocked",
        code="pi_action_authorization_required",
        summary=f"This action requires a one-use {action} grant for the current message.",
        owner="easyicu.webserver.pi_copilot",
    )


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


def _artifact_resource(run_id: Any, artifact_name: Any) -> Optional[Dict[str, Any]]:
    """Build a path-free browser reference to one whitelisted run artefact."""

    clean_run = stable_code(run_id)
    clean_name = str(artifact_name or "").strip()
    if (
        clean_run is None
        or not clean_name
        or len(clean_name) > 160
        or Path(clean_name).name != clean_name
        or not clean_name.endswith(".json")
    ):
        return None
    return {
        "kind": "research_artifact",
        "run_id": clean_run,
        "artifact": clean_name,
        "label": clean_name,
        "media_type": "application/json",
    }


def _artifact_resources(
    run_id: Any, artifacts: Iterable[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    resources = []
    for artifact in list(artifacts)[:80]:
        resource = _artifact_resource(run_id, artifact.get("name"))
        if resource is not None:
            resources.append(resource)
    return resources


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
        details={
            "plan": projected,
            "resource": _artifact_resource(row.get("run_id"), "agent_plan.json"),
        },
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
    gate = review.get("gate")
    gate = gate if isinstance(gate, Mapping) else {}
    readiness = review.get("readiness")
    readiness = readiness if isinstance(readiness, Mapping) else {}
    checks = gate.get("checks")
    checks = checks if isinstance(checks, list) else []
    failed_requirement_codes = [
        stable_code(check.get("id"))
        for check in checks
        if isinstance(check, Mapping)
        and isinstance(check.get("id"), str)
        and not check.get("passed")
    ][:50]
    failed_requirement_codes = [
        code for code in failed_requirement_codes if code is not None
    ]
    missing_requirement_codes = [
        stable_code(item)
        for item in (readiness.get("non_human_failures") or [])
        if isinstance(item, str)
    ][:50]
    missing_requirement_codes = [
        code for code in missing_requirement_codes if code is not None
    ]
    details = bounded_json_projection(
        {
            "run_id": row.get("run_id"),
            "gate": {
                "status": gate.get("status"),
                "gate_code": stable_code(gate.get("reason")),
                "reportable": bool(gate.get("reportable")),
                "draft_unlocked": bool(gate.get("draft_unlocked")),
                "checks_total": len(checks),
                "checks_passed": sum(
                    1
                    for check in checks
                    if isinstance(check, Mapping) and check.get("passed")
                ),
                "failed_requirement_codes": failed_requirement_codes,
            },
            "readiness": {
                key: readiness.get(key)
                for key in (
                    "status",
                    "signable",
                    "signed",
                    "signoff_stale",
                    "reportable",
                    "draft_unlocked",
                    "gate_status",
                    "checks_total",
                    "checks_passed",
                    "human_signoff_passed_in_gate",
                )
                if readiness.get(key) is not None
            }
            | {"missing_requirement_codes": missing_requirement_codes},
            "signed": bool(review.get("signed")),
            "signoff_stale": bool(review.get("signoff_stale")),
            "resource": _artifact_resource(
                row.get("run_id"), "quality_gate.json"
            ),
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
        details={
            "run_id": row.get("run_id"),
            "artifacts": artifacts,
            "resources": _artifact_resources(row.get("run_id"), artifacts),
        },
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
        "resource": _artifact_resource(
            row.get("run_id"), "evidence_ledger.json"
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
        code = str(
            job.get("error_code")
            or job.get("cancel_reason_code")
            or job["status"]
        )
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
            code = (
                stable_code(gate.get("reason"))
                or stable_code(readiness.get("reason"))
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
                    "gate_code": stable_code(gate.get("reason")),
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
    grant_block = _consume_action(context, "configure")
    if grant_block is not None:
        return grant_block

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
    result = _result(
        context,
        status="ok",
        code="study_context_updated",
        summary=(
            f"Saved typed StudyContext revision {int(updated.get('revision') or 0)}. "
            "The conversation host will rebind this Pi session after the turn settles."
        ),
        owner="easyicu.webserver.study_contexts",
        details={
            "study": project_study_context(updated),
            "rebind_required": True,
            "host_rebind_after_turn": True,
        },
    )
    context.invalidate_authority("study_context_updated")
    return result


def _run(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_type", "llm_provider"))
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
    grant_block = _consume_action(context, "run")
    if grant_block is not None:
        return grant_block
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
    result = _result(
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
    context.invalidate_authority("easyicu_run_submitted")
    return result


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
    grant_block = _consume_action(context, "cancel")
    if grant_block is not None:
        return grant_block
    accepted = job.request_cancel("pi_copilot_user_authorized")
    result = _result(
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
    if accepted:
        context.invalidate_authority("easyicu_job_cancel_requested")
    return result


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


def _workspace_access(
    context: ToolExecutionContext,
    *,
    require_write: bool = False,
) -> tuple[Optional[ProjectWorkspace], Optional[Dict[str, Any]]]:
    if context.session.agent_mode != "workspace":
        return None, _result(
            context,
            status="blocked",
            code="pi_workspace_mode_required",
            summary="Open a Pi workspace conversation before using project artifact tools.",
            owner="easyicu.webserver.pi_copilot.workspace",
        )
    if context.workspace_root is None or not context.session.project_id:
        return None, _result(
            context,
            status="blocked",
            code="pi_workspace_unavailable",
            summary="The isolated Pi project workspace is unavailable for this session.",
            owner="easyicu.webserver.pi_copilot.workspace",
        )
    if require_write and "workspace_write" not in context.grant.provided_actions:
        return None, _result(
            context,
            status="blocked",
            code="pi_workspace_write_authorization_required",
            summary="Project file changes require the workspace-write grant for this message.",
            owner="easyicu.webserver.pi_copilot.workspace",
        )
    return ProjectWorkspace(context.workspace_root), None


def _workspace_resource(payload: Mapping[str, Any], *, kind: str = "file") -> Dict[str, Any]:
    return {
        "kind": kind,
        "file": str(payload.get("file") or ""),
        "label": Path(str(payload.get("file") or "artifact")).name,
        "media_type": str(payload.get("media_type") or "text/plain"),
    }


def _load_skill(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("name",), required=("name",))
    _, blocked = _workspace_access(context)
    if blocked:
        return blocked
    name = str(params.get("name") or "").strip()
    if name != "web-prototype":
        return _result(
            context,
            status="not_found",
            code="pi_workspace_skill_not_found",
            summary="The requested governed Pi workspace skill is not installed.",
            owner="easyicu.webserver.pi_copilot.workspace",
            details={"available_skills": ["web-prototype"]},
        )
    skill_file = (
        Path(__file__).resolve().with_name("node_app")
        / "src"
        / "skills"
        / "web-prototype"
        / "SKILL.md"
    )
    instructions = skill_file.read_text(encoding="utf-8")[:12_000]
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_skill_loaded",
        summary="Loaded the governed web-prototype workspace skill.",
        details={"skill": name, "instructions": instructions},
    )


def _list_project_files(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    rows = workspace.list_files(context.session.project_id)
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_files_listed",
        summary=f"Listed {len(rows)} project workspace files.",
        details={"files": rows, "count": len(rows)},
    )


def _read_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("file", "start_line", "end_line"),
        required=("file",),
    )
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.read_file(
        context.session.project_id,
        params["file"],
        start_line=int(params.get("start_line") or 1),
        end_line=(int(params["end_line"]) if params.get("end_line") else None),
    )
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_read",
        summary=f"Read {payload['file']}.",
        details={**payload, "resource": _workspace_resource(payload)},
    )


def _write_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("file", "content"), required=("file", "content"))
    workspace, blocked = _workspace_access(context, require_write=True)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.write_file(
        context.session.project_id,
        params["file"],
        params["content"],
    )
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_written",
        summary=f"{'Created' if payload['created'] else 'Updated'} {payload['file']}.",
        details={**payload, "resource": _workspace_resource(payload)},
    )


def _edit_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("file", "old_text", "new_text"),
        required=("file", "old_text"),
    )
    workspace, blocked = _workspace_access(context, require_write=True)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.edit_file(
        context.session.project_id,
        params["file"],
        old_text=params["old_text"],
        new_text=params.get("new_text") or "",
    )
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_edited",
        summary=f"Edited {payload['file']} with one exact replacement.",
        details={**payload, "resource": _workspace_resource(payload)},
    )


def _check_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("file",), required=("file",))
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.check_file(context.session.project_id, params["file"])
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_checked",
        summary=f"Checked {payload['file']} with {payload['checker']}.",
        details={**payload, "resource": _workspace_resource(payload)},
    )


def _preview_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("file",), required=("file",))
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.preview_file(context.session.project_id, params["file"])
    resource = _workspace_resource(payload, kind="webpage")
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_preview_ready",
        summary=f"Prepared the live preview for {payload['file']}.",
        details={
            "file": payload["file"],
            "media_type": payload["media_type"],
            "resource": resource,
        },
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
    "easyicu_load_skill": _load_skill,
    "easyicu_list_project_files": _list_project_files,
    "easyicu_read_project_file": _read_project_file,
    "easyicu_write_project_file": _write_project_file,
    "easyicu_edit_project_file": _edit_project_file,
    "easyicu_check_project_file": _check_project_file,
    "easyicu_preview_project_file": _preview_project_file,
}


def execute_tool(
    name: str,
    arguments: Mapping[str, Any],
    context: ToolExecutionContext,
) -> Dict[str, Any]:
    """Execute exactly one registered tool through its existing EasyICU owner."""

    context.assert_authority_fresh()
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
    "WORKSPACE_TOOLS",
    "execute_tool",
]
