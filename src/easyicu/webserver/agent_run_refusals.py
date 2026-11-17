"""One safe, typed web-log receipt for every refused agent-run request."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Mapping

from fastapi import HTTPException

from easyicu.webserver.host_action_contracts import ACTION_DECISION_FAMILIES

logger = logging.getLogger(__name__)


def logged_refusal(error: HTTPException, body: Mapping[str, Any]) -> HTTPException:
    """Never log exception text, request values, identifiers or credentials."""
    if error.status_code not in {400, 409}:
        return error
    detail = dict(error.detail) if isinstance(error.detail, Mapping) else {}
    code = detail.get("error")
    if not isinstance(code, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,127}", code) is None:
        code = "agent_run_request_refused"
    detail["error"] = code
    action = body.get("host_action")
    kind = action.get("action_code") if isinstance(action, Mapping) else None
    if not isinstance(kind, str) or kind not in ACTION_DECISION_FAMILIES:
        kind = "invalid" if action is not None else "none"
    study_hash = hashlib.sha256(str(body.get("study_context_id") or "").encode()).hexdigest()[:12]
    receipt = {"event": "agent_run_refused", "status": error.status_code,
               "error": code, "host_action": kind, "study_context_hash": study_hash}
    logger.warning("%s", json.dumps(receipt, sort_keys=True), extra=receipt)
    return HTTPException(status_code=error.status_code, detail=detail, headers=error.headers)
