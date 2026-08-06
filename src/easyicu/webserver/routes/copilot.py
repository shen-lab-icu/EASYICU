"""Copilot compatibility HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import copilot_sessions, study_intent

router = APIRouter()


@router.post("/api/copilot/study-intent")
def post_copilot_study_intent(body: Dict[str, Any]) -> dict:
    """Read a typed study-contract proposal from the user's own question.

    Never rewrites the question and never fills a slot it could not read: the
    response names every unread slot so the caller has to ask rather than
    substitute a default.
    """
    payload = body or {}
    try:
        return study_intent.extract_study_intent(
            payload.get("question"),
            llm_provider=str(payload.get("llm_provider") or "offline"),
            external_llm_opt_in=bool(payload.get("external_llm_opt_in")),
            ai_enabled=bool(payload.get("ai_enabled")),
            language=str(payload.get("language") or "en"),
        )
    except study_intent.StudyIntentError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


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
