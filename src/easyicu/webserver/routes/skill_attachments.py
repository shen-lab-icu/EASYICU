"""Skill composer and project-default APIs; UI copy remains client owned."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from easyicu.extensions.skill_attachments import AttachedSkills
from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError

router = APIRouter(prefix="/api/copilot/pi")


class DraftSkillsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(min_length=1, max_length=160)
    attached_skills: AttachedSkills


class DefaultSkillsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attached_skills: AttachedSkills
    expected_attached_skills: AttachedSkills | None = None


def _call(method: str, *args, **kwargs):
    try:
        return getattr(get_pi_copilot_service(), method)(*args, **kwargs)
    except PiCopilotError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code}) from exc


@router.get("/sessions/{session_id}/skills")
def get_attachable_skills(session_id: str, project_id: str):
    return _call("attachable_skills", session_id, project_id=project_id)


@router.put("/sessions/{session_id}/draft-skills")
def put_draft_skills(session_id: str, body: DraftSkillsRequest):
    return _call("set_draft_skills", session_id, project_id=body.project_id, attached_skills=body.attached_skills)


@router.get("/projects/{project_id}/default-skills")
def get_default_skills(project_id: str):
    return _call("project_default_skills", project_id=project_id)


@router.put("/projects/{project_id}/default-skills")
def put_default_skills(project_id: str, body: DefaultSkillsRequest):
    kwargs = {"project_id": project_id, "attached_skills": body.attached_skills}
    if body.expected_attached_skills is not None:
        kwargs["expected_attached_skills"] = body.expected_attached_skills
    return _call("project_default_skills", **kwargs)
