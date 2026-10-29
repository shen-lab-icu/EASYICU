"""A bounded file-text channel for already isolated workspace reads.

Generic study/tool projections retain their 2,000-character string ceiling.
Only this channel accepts the workspace reader's existing 24,000 characters,
after the same complete-string privacy scan and ordinary metadata guards.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .contracts import PiCopilotError, PiToolResult, ToolExecutionContext
from .projections import ensure_safe_projection, reject_sensitive_message
from .workspace import MAX_READ_CHARS

MAX_READ_RESULT_BYTES = 160 * 1024
MAX_READ_METADATA_BYTES = 8 * 1024


def project_workspace_read_result(
    context: ToolExecutionContext, payload: Mapping[str, Any], resource: Mapping[str, Any],
) -> dict[str, Any]:
    text = payload.get("text")
    if not isinstance(text, str):
        raise PiCopilotError("pi_workspace_read_projection_invalid", "The workspace reader returned invalid text.", status_code=500)
    # Scan the entire selected text, including both sides of any would-be
    # chunk boundary. Never hide sensitive tails through clipping or encoding.
    try:
        reject_sensitive_message(text)
    except PiCopilotError as exc:
        raise PiCopilotError("pi_projection_blocked", "The host withheld a sensitive model-visible string value.",
                             status_code=500, details={"cause_code": exc.code}) from exc
    if len(text) > MAX_READ_CHARS:
        raise PiCopilotError("pi_workspace_read_projection_too_large", "The workspace read exceeds its bounded file-text channel.",
                             status_code=500, details={"max_chars": MAX_READ_CHARS})
    metadata = {key: value for key, value in payload.items() if key != "text"}
    details = {**metadata, "resource": dict(resource)}
    # The exemption applies only to details.text. Metadata, nested descriptors
    # and the authority envelope keep the shared privacy/list/string limits.
    ensure_safe_projection(details, max_bytes=MAX_READ_METADATA_BYTES)
    envelope = PiToolResult(status="ok", code="pi_workspace_file_read", summary=f"Read {payload['file']}.",
        owner="easyicu.webserver.pi_copilot.workspace", details=details,
        authority=context.session.binding.model_dump(mode="json")).model_dump(mode="json")
    ensure_safe_projection(envelope, max_bytes=MAX_READ_METADATA_BYTES)
    envelope["details"]["text"] = text
    size = len(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))
    if size > MAX_READ_RESULT_BYTES:
        raise PiCopilotError("pi_workspace_read_projection_too_large", "The workspace read exceeds its bounded file-text channel.",
                             status_code=500, details={"bytes": size, "max_bytes": MAX_READ_RESULT_BYTES})
    return envelope
