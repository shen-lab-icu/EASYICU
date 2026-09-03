"""Guided Copilot HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import guided_sessions, study_contexts
from easyicu.webserver.pi_copilot.project_authority import ProjectAuthorityStore
from easyicu.webserver.routes.request_parsing import body_int

router = APIRouter()


@router.post("/api/guided/drafts")
def post_guided_draft(body: Dict[str, Any]) -> dict:
    """Persist a metadata-only guided Copilot draft.

    This is intentionally separate from Agent run creation: a guided draft does
    not create an Agent run, does not read patient rows, and never unlocks a
    manuscript draft. It does create a local metadata-only project folder so
    users can manage drafts on disk.
    """
    return guided_sessions.create_guided_draft(body)


@router.post("/api/guided/drafts/list")
def post_guided_drafts_list(body: Dict[str, Any] | None = None) -> dict:
    """List drafts together with their authoritative configuration health."""
    result = guided_sessions.list_guided_drafts(
        limit=body_int(body or {}, "limit", 20, min_value=1, max_value=100)
    )
    drafts = result.get("drafts")
    if not isinstance(drafts, list) or not drafts:
        return result
    bindings = {
        binding.project_id: binding.study_context_id
        for binding in ProjectAuthorityStore().bindings()
    }
    relevant_context_ids = [
        bindings[str(row.get("id") or "")]
        for row in drafts
        if isinstance(row, dict) and str(row.get("id") or "") in bindings
    ]
    existing_ids = study_contexts.existing_context_ids(relevant_context_ids)
    for row in drafts:
        if not isinstance(row, dict):
            continue
        context_id = bindings.get(str(row.get("id") or ""))
        if context_id is None:
            status = "unbound"
        elif context_id in existing_ids:
            status = "ready"
        else:
            status = "configuration_missing"
        row["configuration_health"] = {
            "status": status,
            "can_continue": status != "configuration_missing",
        }
    return result


def _guided_draft_remove_response(body: Dict[str, Any] | None) -> dict:
    result = guided_sessions.remove_guided_draft(body or {})
    if not result.get("ok") and not result.get("blocked"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/guided/drafts/remove")
def post_guided_draft_remove(body: Dict[str, Any] | None = None) -> dict:
    """Remove a Guided draft from the registry, optionally using system trash.

    The default remains metadata-only.  Moving the matching local project folder
    to the system trash requires a separate explicit flag and exact draft-id
    confirmation; permanent deletion is not exposed by this endpoint.
    """
    return _guided_draft_remove_response(body)


@router.delete("/api/guided/drafts/remove")
def delete_guided_draft_remove(body: Dict[str, Any] | None = None) -> dict:
    """Compatibility path for cached clients that used DELETE for draft removal."""
    return _guided_draft_remove_response(body)


@router.post("/api/guided/session")
def post_guided_session(body: Dict[str, Any]) -> dict:
    """Create a local metadata-only front-door Guided Copilot session."""
    return guided_sessions.create_guided_session(body)


@router.post("/api/guided/project/open")
def post_guided_project_open(body: Dict[str, Any]) -> dict:
    """Open or create Guided Copilot memory scoped to one local project folder."""
    result = guided_sessions.open_guided_project(body or {})
    if not result.get("ok") and not result.get("blocked"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/guided/message")
def post_guided_message(body: Dict[str, Any]) -> dict:
    """Route one Guided Copilot local-mode message through the backend state machine."""
    result = guided_sessions.post_guided_message(body)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/guided/action")
def post_guided_action(body: Dict[str, Any]) -> dict:
    """Execute a whitelisted Guided Copilot routing action or fail closed."""
    result = guided_sessions.execute_guided_action(body or {})
    if not result.get("ok") and not result.get("blocked"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/guided/sessions/list")
def post_guided_sessions_list(body: Dict[str, Any] | None = None) -> dict:
    """List local metadata-only Guided Copilot session folders."""
    return guided_sessions.list_guided_sessions(
        limit=body_int(body or {}, "limit", 20, min_value=1, max_value=100)
    )


__all__ = ["router"]
