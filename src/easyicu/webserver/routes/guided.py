"""Guided Copilot HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import guided_sessions

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
    """List metadata-only guided Copilot drafts from local settings storage."""
    return guided_sessions.list_guided_drafts(
        limit=int((body or {}).get("limit") or 20)
    )


def _guided_draft_remove_response(body: Dict[str, Any] | None) -> dict:
    result = guided_sessions.remove_guided_draft(body or {})
    if not result.get("ok") and not result.get("blocked"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/guided/drafts/remove")
def post_guided_draft_remove(body: Dict[str, Any] | None = None) -> dict:
    """Remove a metadata-only guided draft from the local registry.

    This deliberately does not delete the local project folder. Project folders
    may contain Idea Mining or Agent artifacts and require separate explicit
    file-system management.
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
        limit=int((body or {}).get("limit") or 20)
    )


__all__ = ["router"]
