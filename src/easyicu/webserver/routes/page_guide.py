"""Page Guide HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import copilot_sessions
from easyicu.webserver.routes.request_parsing import body_int

router = APIRouter()


@router.post("/api/page-guide/sessions")
def post_page_guide_session(body: Dict[str, Any]) -> dict:
    """Create a local metadata-only Page guide session."""
    payload = dict(body or {})
    payload["scope"] = "page_guide"
    return copilot_sessions.create_session(payload)


@router.post("/api/page-guide/message")
def post_page_guide_message(body: Dict[str, Any]) -> dict:
    """Classify one Page guide shortcut and return bounded local UI actions."""
    payload = dict(body or {})
    payload["scope"] = "page_guide"
    result = copilot_sessions.post_message(payload)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/page-guide/action")
def post_page_guide_action(body: Dict[str, Any]) -> dict:
    """Execute a whitelisted local Page guide action or fail closed."""
    result = copilot_sessions.execute_action(body or {})
    if not result.get("ok") and not result.get("blocked"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/page-guide/sessions/list")
def post_page_guide_sessions_list(body: Dict[str, Any] | None = None) -> dict:
    """List local metadata-only Page guide session folders."""
    return copilot_sessions.list_sessions(
        limit=body_int(body or {}, "limit", 20, min_value=1, max_value=100)
    )


__all__ = ["router"]
