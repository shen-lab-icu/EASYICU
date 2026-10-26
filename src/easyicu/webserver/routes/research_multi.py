"""Path-free host actions; the lead supplies the combined-card presentation."""
from fastapi import APIRouter, HTTPException
from pydantic import Field, ValidationError

from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver import research_multi_actions as actions
from easyicu.webserver.research_multi_contracts import Code, Contract, Database, DatabaseTarget, Digest, MultiDatabaseRequest

router = APIRouter()


class BatchAction(Contract):
    project_id: Code
    session_id: Code
    batch_id: Digest


class Approval(BatchAction):
    card_sha256: Digest


class Cancellation(BatchAction):
    database: Database


class Followup(Contract):
    project_id: Code
    session_id: Code
    run_id: Code
    suggestion_id: Digest
    expected_source_sha256: Digest
    targets: tuple[DatabaseTarget, ...] = Field(default=(), max_length=6)


def call(fn, **kwargs):
    try:
        return fn(get_pi_copilot_service(), **kwargs)
    except PiCopilotError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code}) from exc
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise HTTPException(status_code=409, detail={"error": getattr(exc, "code", "multi_database_unavailable")}) from exc


@router.post("/api/copilot/pi/multi-database")
def propose(body: MultiDatabaseRequest):
    return call(actions.propose, request=body)


@router.post("/api/copilot/pi/multi-database/follow-up")
def followup(body: Followup):
    return call(actions.dispatch_followup, **body.model_dump())


@router.post("/api/copilot/pi/multi-database/approve")
def approve(body: Approval):
    return call(actions.approve, **body.model_dump())


@router.post("/api/copilot/pi/multi-database/cancel")
def cancel(body: Cancellation):
    return call(actions.cancel, **body.model_dump())


@router.get("/api/copilot/pi/multi-database/{batch_id}")
def status(batch_id: str, project_id: str, session_id: str):
    try:
        action = BatchAction(batch_id=batch_id, project_id=project_id, session_id=session_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"error": "multi_database_request_invalid"}) from exc
    return call(actions.status, **action.model_dump())
