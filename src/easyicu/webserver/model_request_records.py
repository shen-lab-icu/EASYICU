"""Read request metadata through existing run and conversation ownership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from easyicu.model_request_metadata import project_request
from easyicu.webserver.research_run_usage import research_run_usage

MAX_RECEIPT_BYTES = 8 * 1024 * 1024


def _view(status: str, requests: list, *, state: str = "unknown") -> dict[str, Any]:
    return {"schema_version": "easyicu.egress-view/1", "status": status,
            "reason_code": {"available": "request_metadata_available",
                            "not_recorded": "request_metadata_not_recorded",
                            "unavailable": "request_metadata_invalid"}[status],
            "scope": "recorded_provider_attempts", "run_state": state,
            "payload_status": "withheld", "requests": requests}


def run_egress(wrapper_dir: Path) -> dict[str, Any]:
    usage = research_run_usage(wrapper_dir, include_requests=True)
    if usage is None:
        return _view("not_recorded", [])
    if not usage["accounting_complete"]:
        return _view("unavailable", [])
    return _view("available", usage["request_records"], state=usage["status"])


def conversation_egress(service: Any, session_id: str) -> dict[str, Any]:
    """Resolve only host-owned metadata; never open Pi's transcript or a provider.

    The API and Codex gateway share the template gateway's session directory.
    Resolving/starting a gateway here would refresh credentials on a read API.
    """
    record = service._get_record(session_id)
    root = Path(service.gateway.session_dir)
    identity = hashlib.sha256(record.session_id.encode("utf-8")).hexdigest()
    path = root / ".egress" / f"{identity}.jsonl"
    try:
        if any(p.is_symlink() for p in (path, path.parent, root)):
            raise ValueError("symlinked_receipts")
        if not path.exists():
            return _view("not_recorded", [])
        if not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
            raise ValueError("invalid_receipt_file")
        with path.open("rb") as stream:
            raw = stream.read(MAX_RECEIPT_BYTES + 1)
        if len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError("receipt_file_too_large")
        rows = [json.loads(line) for line in raw.splitlines()]
        if any(not isinstance(row, dict) or row.get("schema_version") != "easyicu.model-request/1"
               for row in rows):
            raise ValueError("invalid_receipt")
        return _view("available", [project_request(row, sequence=i + 1)
                                    for i, row in enumerate(rows)])
    except (OSError, ValueError, TypeError):
        return _view("unavailable", [])
