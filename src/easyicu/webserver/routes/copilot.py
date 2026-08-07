"""Copilot compatibility HTTP route adapters."""

from __future__ import annotations

from typing import Annotated, Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool, StringConstraints

from easyicu.webserver import copilot_sessions, study_intent
from easyicu.webserver import settings as settings_store

router = APIRouter()


class StudyIntentRequest(BaseModel):
    """Bounded wire contract for optional Copilot intent extraction."""

    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=1200),
    ]
    llm_provider: str = "offline"
    external_llm_opt_in: StrictBool = False
    language: str = "en"


@router.post("/api/copilot/study-intent")
def post_copilot_study_intent(body: StudyIntentRequest) -> dict:
    """Read a typed study-contract proposal from the user's own question.

    Never rewrites the question and never fills a slot it could not read: the
    response names every unread slot so the caller has to ask rather than
    substitute a default.
    """
    try:
        return study_intent.extract_study_intent(
            body.question,
            llm_provider=body.llm_provider,
            external_llm_opt_in=body.external_llm_opt_in,
            # Browser request data cannot grant the server-wide external-call
            # permission.  That authority belongs to the local settings store.
            ai_enabled=bool(settings_store.load_settings().get("ai_enabled")),
            language=body.language,
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
