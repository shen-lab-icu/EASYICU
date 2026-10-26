"""Read/update personal preferences and bounded, project-scoped model usage."""
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from pydantic import StringConstraints

from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.personal_preferences import PreferencesError
from easyicu.webserver.pi_copilot.personal_usage import conversation_usage, run_usage

router = APIRouter()
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


def _call(operation):
    try:
        return operation(get_pi_copilot_service())
    except (PreferencesError, PiCopilotError) as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code}) from exc


@router.get("/api/copilot/pi/preferences")
def get_preferences() -> dict:
    return _call(lambda service: service.preferences.projection())


@router.patch("/api/copilot/pi/preferences")
def update_preferences(body: Annotated[dict, Body()]) -> dict:
    def update(service):
        service.preferences.update(body)
        return service.preferences.projection()
    return _call(update)


@router.get("/api/copilot/pi/projects/{project_id}/runs/{run_id}/usage")
def get_run_usage(project_id: Identifier, run_id: Identifier) -> dict:
    return _call(lambda service: run_usage(service, project_id=project_id, run_id=run_id))


@router.get("/api/copilot/pi/sessions/{session_id}/usage")
def get_conversation_usage(session_id: Identifier, project_id: Identifier) -> dict:
    return _call(lambda service: conversation_usage(service, project_id=project_id, session_id=session_id))
