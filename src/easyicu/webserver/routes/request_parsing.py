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


def body_int(
    body: Dict[str, Any],
    key: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Parse one request-body integer or raise the canonical HTTP 400 error.

    Several list/search routes used a bare ``int(body.get(key) or N)``, which
    turns a non-numeric client value into an HTTP 500 ``ValueError``. A body
    type error is a client error and must answer 400 with the field named.
    """
    raw = (body or {}).get(key)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_integer", "field": key},
            ) from exc
    if min_value is not None and value < min_value:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "integer_out_of_range",
                "field": key,
                "min": min_value,
                "max": max_value,
            },
        )
    if max_value is not None and value > max_value:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "integer_out_of_range",
                "field": key,
                "min": min_value,
                "max": max_value,
            },
        )
    return value


__all__ = ["body_bool", "body_int"]
