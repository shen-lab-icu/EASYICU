"""Project-scoped read of a run's scientific review (owner: webserver.scientific_review)."""
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import StringConstraints

from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.run_scientific_review import run_scientific_review

router = APIRouter()
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


@router.get("/api/copilot/pi/projects/{project_id}/runs/{run_id}/scientific-review")
def get_run_scientific_review(
    project_id: Identifier,
    run_id: Identifier,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=50)] = 50,
) -> dict:
    try:
        return run_scientific_review(get_pi_copilot_service(), project_id=project_id, run_id=run_id,
                                     offset=offset, limit=limit)
    except PiCopilotError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code}) from exc
