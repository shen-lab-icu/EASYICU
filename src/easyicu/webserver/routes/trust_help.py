"""Read-only trust/help APIs. The desktop owns presentation and last-seen state."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

from easyicu.webserver import run_refs
from easyicu.webserver.hosted import require_instance_path
from easyicu.webserver.model_request_records import conversation_egress, run_egress
from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.problem_bundle import build_problem_bundle
from easyicu.webserver.release_notes import whats_new

router = APIRouter()
_PRIVATE = {"Cache-Control": "no-store"}


def _run_path(run_ref: str) -> Path:
    path = run_refs.resolve(run_ref)
    if path is None:
        raise HTTPException(status_code=404, detail="run_ref_unknown")
    require_instance_path(path)
    return Path(path)


@router.get("/api/trust/runs/{run_ref}/egress")
def get_run_egress(run_ref: str) -> JSONResponse:
    return JSONResponse(run_egress(_run_path(run_ref)), headers=_PRIVATE)


@router.get("/api/trust/conversations/{session_id}/egress")
def get_conversation_egress(session_id: str) -> JSONResponse:
    try:
        payload = conversation_egress(get_pi_copilot_service(), session_id)
    except PiCopilotError as exc:
        raise HTTPException(status_code=exc.status_code, detail="conversation_unavailable") from exc
    return JSONResponse(payload, headers=_PRIVATE)


@router.get("/api/help/whats-new")
def get_whats_new(last_seen_version: str | None = None) -> JSONResponse:
    try:
        payload = whats_new(last_seen_version)
    except ValueError as exc:
        code = "last_seen_version_invalid" if str(exc) == "last_seen_version_invalid" else "release_notes_invalid"
        raise HTTPException(status_code=400 if code == "last_seen_version_invalid" else 503, detail=code) from exc
    return JSONResponse(payload, headers=_PRIVATE)


@router.get("/api/help/runs/{run_ref}/problem-bundle")
def get_problem_bundle(run_ref: str) -> Response:
    return Response(build_problem_bundle(_run_path(run_ref)), media_type="application/zip",
                    headers={**_PRIVATE, "Content-Disposition": 'attachment; filename="easyicu-problem.zip"'})
