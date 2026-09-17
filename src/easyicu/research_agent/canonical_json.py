"""Byte-stable JSON and SHA-256 primitives shared by authority contracts.

Callers remain responsible for schema-specific normalization.  This module
only owns the common wire representation: UTF-8, Unicode preserved, sorted
keys, compact separators, and rejection of non-finite numbers.

It also owns tolerant LLM-response JSON extraction (:func:`extract_json_payload`
and :func:`extract_json_object`).  Agent-facing ``_extract_json`` helpers
elsewhere are thin wrappers over these two functions so fence-stripping and
balanced-scan semantics stay identical across owners.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional


def _strip_fence(text: str) -> str:
    """Remove one leading/trailing Markdown code fence, if present."""

    stripped = text.strip()
    if "```" in stripped:
        stripped = re.sub(r"^```[a-zA-Z0-9]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped.strip())
    return stripped


def extract_json_payload(text: str) -> Any:
    """Parse an LLM response that may wrap JSON in fences or prose.

    Tries a direct parse first, then falls back to the first balanced
    ``{...}`` or ``[...]`` block.  Reasoning/thinking models routinely emit
    prose before the JSON, so a strict whole-string ``json.loads`` would
    reject otherwise-valid payloads.

    Raises:
        ValueError: If the response contains no parseable JSON.
    """

    stripped = _strip_fence(str(text or ""))
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fall back: scan for the first balanced JSON object/array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : idx + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("LLM response did not contain parseable JSON")


def extract_json_object(text: str) -> Optional[dict]:
    """Extract a single JSON object from an LLM response, if present.

    Unlike :func:`extract_json_payload`, a top-level array (or any
    non-object payload) yields ``None`` instead of raising, matching the
    tolerant ``Optional[dict]`` contract of the acquisition-layer helpers.
    """

    try:
        payload = extract_json_payload(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


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


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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
    "extract_json_object",
    "extract_json_payload",
    "sha256_bytes",
    "sha256_file",
]
