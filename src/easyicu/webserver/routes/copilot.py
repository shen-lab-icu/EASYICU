"""Copilot compatibility HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import copilot_sessions

router = APIRouter()


@router.post("/api/copilot/sessions")
def post_copilot_session(body: Dict[str, Any]) -> dict:
    """Compatibility endpoint for local metadata-only Copilot/Page guide sessions."""
    return copilot_sessions.create_session(body)


@router.post("/api/copilot/message")
def post_copilot_message(body: Dict[str, Any]) -> dict:
    """Compatibility endpoint for one bounded local Copilot/Page guide shortcut."""
    result = copilot_sessions.post_message(body)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/copilot/action")
def post_copilot_action(body: Dict[str, Any]) -> dict:
    """Compatibility endpoint for a whitelisted local Copilot/Page guide action."""
    result = copilot_sessions.execute_action(body)
    if not result.get("ok") and not result.get("blocked"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/copilot/sessions/list")
def post_copilot_sessions_list(body: Dict[str, Any] | None = None) -> dict:
    """List local metadata-only Copilot/Page guide session folders."""
    return copilot_sessions.list_sessions(limit=int((body or {}).get("limit") or 20))


__all__ = ["router"]
