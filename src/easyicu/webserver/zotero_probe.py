"""Typed, bounded Zotero Desktop health check, without importing library data."""

from __future__ import annotations

import json
from typing import Literal
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict


class ZoteroConnectionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    available: bool = False
    status: Literal["disabled", "available", "unavailable", "unsupported"]
    reason: Literal[
        "connector_zotero_enabled_false", "local_zotero_api_ready",
        "local_zotero_api_http_error", "local_zotero_api_unavailable",
        "local_zotero_api_invalid_response", "zotero_desktop_only",
    ]
    local_api: str | None = None
    http_status: int | None = None


def probe_zotero(*, enabled: bool, hosted: bool, local_api: str, timeout: float) -> dict:
    if hosted:
        return ZoteroConnectionStatus(enabled=False, status="unsupported", reason="zotero_desktop_only").model_dump(exclude_none=True)
    result = {"enabled": enabled, "local_api": local_api}
    if not enabled:
        return ZoteroConnectionStatus(**result, status="disabled", reason="connector_zotero_enabled_false").model_dump(exclude_none=True)
    try:
        request = urllib.request.Request(f"{local_api}?limit=1", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = int(getattr(response, "status", 200) or 200)
            result["http_status"] = code
            if not 200 <= code < 300:
                reason = "local_zotero_api_http_error"
            else:
                raw = response.read(64_001)
                payload = json.loads(raw) if len(raw) <= 64_000 else None
                valid = isinstance(payload, list) and all(isinstance(row, dict) for row in payload)
                reason = "local_zotero_api_ready" if valid else "local_zotero_api_invalid_response"
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        reason = "local_zotero_api_http_error"
    except (ValueError, UnicodeError):
        reason = "local_zotero_api_invalid_response"
    except (OSError, urllib.error.URLError):
        reason = "local_zotero_api_unavailable"
    ready = reason == "local_zotero_api_ready"
    return ZoteroConnectionStatus(**result, available=ready, status="available" if ready else "unavailable", reason=reason).model_dump(exclude_none=True)
