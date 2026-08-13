"""HTTP routes for the metadata-only StudyContext store."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import jobs as job_store
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.routes.request_parsing import body_bool

router = APIRouter()


def _product_context(context: Dict[str, Any] | None) -> bool:
    """Return whether a context belongs on ordinary product project rails.

    Evaluation contexts remain addressable by exact id for reproducibility and
    run provenance.  Only the aggregate product list filters them; no evidence
    or context record is deleted.
    """
    confirmations = (
        context.get("confirmations") if isinstance(context, dict) else None
    )
    if not isinstance(confirmations, dict):
        return True
    return not (
        confirmations.get("development_only") is True
        or confirmations.get("internal_evaluation") is True
        or confirmations.get("not_for_manuscript") is True
    )


def _reconcile_active_job(context: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not context or not context.get("active_job_id"):
        return context
    job_id = str(context["active_job_id"])
    job = job_store.MANAGER.get(job_id)
    if job is not None and job.status == "running":
        return context
    done_stage = (
        "review_blocked"
        if job is not None
        and isinstance(job.result, dict)
        and isinstance(job.result.get("gate"), dict)
        and job.result["gate"].get("status") == "blocked"
        else "review"
    )
    stage = {
        "done": done_stage,
        "failed": "agent_failed",
        "cancelled": "agent_cancelled",
    }.get(getattr(job, "status", None), "agent_interrupted")
    result = context_store.clear_active_job_if(
        str(context["id"]),
        job_id,
        current_stage=stage,
        last_route="agent",
    )
    return result["context"]


def _raise_context_error(exc: context_store.StudyContextError) -> None:
    error = exc.detail.get("error")
    status_code = (
        404
        if error == "study_context_not_found"
        else 409
        if error
        in {"study_context_revision_required", "study_context_revision_conflict"}
        else 400
    )
    raise HTTPException(status_code=status_code, detail=exc.detail) from exc


@router.get("/api/study-contexts/active")
def get_active_study_context() -> dict:
    try:
        context = _reconcile_active_job(context_store.get_active_context())
    except context_store.StudyContextError as exc:
        _raise_context_error(exc)
    return {"ok": True, "context": context}


@router.get("/api/study-contexts")
def get_study_contexts() -> dict:
    try:
        result = context_store.list_contexts()
        result["contexts"] = [
            _reconcile_active_job(context)
            for context in result["contexts"]
            if _product_context(context)
        ]
    except context_store.StudyContextError as exc:
        _raise_context_error(exc)
    return {"ok": True, **result}


@router.post("/api/study-contexts")
def post_study_context(body: Dict[str, Any]) -> dict:
    patch = dict(body)
    active = body_bool(patch, "active", True)
    patch.pop("active", None)
    expected_revision = patch.pop("expected_revision", None)
    content_fields = set(patch) - {"id"}
    try:
        context = context_store.upsert_context(
            patch,
            active=active,
            expected_revision=expected_revision,
            require_revision=bool(content_fields),
            lifecycle_write=False,
        )
        result = context_store.list_contexts()
    except context_store.StudyContextError as exc:
        _raise_context_error(exc)
    return {"ok": True, "context": context, "active_id": result.get("active_id")}


@router.post("/api/study-contexts/handoff")
def post_study_context_handoff(body: Dict[str, Any]) -> dict:
    context_id = str(body.get("study_context_id") or body.get("id") or "").strip()
    if not context_id:
        raise HTTPException(
            status_code=400, detail={"error": "study_context_id_required"}
        )
    current_stage = body.get("current_stage")
    target_route = body.get("target_route") or body.get("last_route")
    from_route = body.get("from_route") or body.get("last_route")
    has_active_job = "active_job_id" in body
    if current_stage is None and target_route is None and not has_active_job:
        raise HTTPException(
            status_code=400, detail={"error": "study_context_handoff_required"}
        )
    try:
        previous = context_store.get_context(context_id)
        if previous is None:
            raise context_store.StudyContextError(
                {
                    "error": "study_context_not_found",
                    "study_context_id": context_id,
                }
            )
        if (has_active_job or previous.get("active_job_id")) and (
            "expected_revision" not in body
        ):
            raise context_store.StudyContextError(
                {
                    "error": "study_context_revision_required",
                    "study_context_id": context_id,
                    "current_revision": previous.get("revision"),
                }
            )
        kwargs: Dict[str, Any] = {
            "current_stage": current_stage,
            "last_route": target_route,
        }
        if has_active_job:
            kwargs["active_job_id"] = body.get("active_job_id")
        if "expected_revision" in body:
            kwargs["expected_revision"] = body.get("expected_revision")
        context = context_store.handoff_context(context_id, **kwargs)
    except context_store.StudyContextError as exc:
        _raise_context_error(exc)
    return {
        "ok": True,
        "context": context,
        "handoff": {
            "from_stage": previous.get("current_stage"),
            "to_stage": context.get("current_stage"),
            "from_route": from_route or previous.get("last_route"),
            "target_route": context.get("last_route"),
        },
    }


@router.get("/api/study-contexts/{context_id}")
def get_study_context(context_id: str) -> dict:
    try:
        context = _reconcile_active_job(context_store.get_context(context_id))
    except context_store.StudyContextError as exc:
        _raise_context_error(exc)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "study_context_not_found",
                "study_context_id": context_id,
            },
        )
    return {"ok": True, "context": context}


__all__ = ["router"]
