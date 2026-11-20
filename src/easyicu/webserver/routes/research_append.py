"""HTTP adapter for additions to completed results; approval uses agent/control."""
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.research_append_actions import append_options, propose_for_conversation
from easyicu.webserver.research_followups import result_followups

router = APIRouter()
_LOG = logging.getLogger(__name__)

from .research_multi import router as multi_router  # noqa: E402

router.include_router(multi_router)


class AppendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
    kind: Literal["sensitivity", "exposure", "subgroup", "article"]
    expected_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _call(fn, **kwargs):
    try:
        return fn(get_pi_copilot_service(), **kwargs)
    except PiCopilotError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code}) from exc
    except (ValueError, OSError, KeyError) as exc:
        # The card keeps one stable code; the cause is logged so a refused
        # append is diagnosable (B34b: a typed-context parse hid behind it).
        _LOG.warning("result append unavailable in %s: %s: %s", fn.__name__, type(exc).__name__, " | ".join(str(exc).splitlines()[:2])[:300])
        raise HTTPException(
            status_code=409, detail={"error": "result_append_unavailable", "cause": type(exc).__name__},
        ) from exc


@router.get("/api/copilot/pi/projects/{project_id}/runs/{run_id}/append-options")
def get_append_options(project_id: str, run_id: str):
    return _call(append_options, project_id=project_id, run_id=run_id)


@router.get("/api/copilot/pi/projects/{project_id}/runs/{run_id}/follow-ups")
def get_result_followups(project_id: str, run_id: str):
    return _call(result_followups, project_id=project_id, run_id=run_id)


@router.post("/api/copilot/pi/sessions/{session_id}/append-run")
def post_append_run(session_id: str, body: AppendRequest):
    return _call(propose_for_conversation, session_id=session_id, **body.model_dump())
