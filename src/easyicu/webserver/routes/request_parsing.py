"""HTTP request parsing shared by native WebServer route owners."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from easyicu.webserver.input_validation import parse_bool


def body_bool(body: Dict[str, Any], key: str, default: bool = False) -> bool:
    """Parse one request-body boolean or raise the canonical HTTP 400 error."""
    try:
        return parse_bool(body.get(key), default=default)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_boolean", "field": key},
        ) from exc


__all__ = ["body_bool"]
