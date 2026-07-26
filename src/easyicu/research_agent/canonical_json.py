"""Byte-stable JSON and SHA-256 primitives shared by authority contracts.

Callers remain responsible for schema-specific normalization.  This module
only owns the common wire representation: UTF-8, Unicode preserved, sorted
keys, compact separators, and rejection of non-finite numbers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any, *, trailing_newline: bool = False) -> str:
    """Serialize JSON-compatible ``value`` with the EasyICU canonical form."""

    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"{rendered}\n" if trailing_newline else rendered


def canonical_json_bytes(
    value: Any,
    *,
    trailing_newline: bool = False,
) -> bytes:
    """Return the UTF-8 bytes of :func:`canonical_json`."""

    return canonical_json(value, trailing_newline=trailing_newline).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of exact bytes."""

    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(
    value: Any,
    *,
    trailing_newline: bool = False,
) -> str:
    """Hash the exact canonical representation of a JSON-compatible value."""

    return sha256_bytes(
        canonical_json_bytes(value, trailing_newline=trailing_newline)
    )


__all__ = [
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "sha256_bytes",
]
