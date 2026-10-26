"""Bounded conversation mention APIs; display wording belongs to the client."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from easyicu.extensions.skill_attachments import ConceptRole, Digest, MentionKind, Mentions, ObjectId
from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError

router = APIRouter(prefix="/api/copilot/pi")


class ResolveMentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(min_length=1, max_length=160)
    kind: MentionKind
    id: ObjectId
    source: str | None = Field(default=None, max_length=32)
    role: ConceptRole = "candidate"
    digest: Digest | None = None


class DraftMentionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(min_length=1, max_length=160)
    mentions: Mentions


def _call(method, *args, **kwargs):
    try:
        return getattr(get_pi_copilot_service(), method)(*args, **kwargs)
    except PiCopilotError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code}) from exc


@router.get("/sessions/{session_id}/mentions")
def search_mentions(session_id: str, project_id: str, kind: MentionKind,
                    query: str = Query(default="", max_length=120), limit: int = Query(default=20, ge=1, le=40)):
    return _call("search_mentions", session_id, project_id=project_id, kind=kind, query=query, limit=limit)


@router.post("/sessions/{session_id}/mentions/resolve")
def resolve_mention(session_id: str, body: ResolveMentionRequest):
    return _call("resolve_mention", session_id, **body.model_dump())


@router.put("/sessions/{session_id}/draft-mentions")
def put_draft_mentions(session_id: str, body: DraftMentionsRequest):
    return _call("set_draft_mentions", session_id, project_id=body.project_id, mentions=body.mentions)
