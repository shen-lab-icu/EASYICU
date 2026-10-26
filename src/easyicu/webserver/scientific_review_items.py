"""Bounded, path-free review items and immutable source coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
from typing import Any

from easyicu.research_agent.research_context.outbound import project_outbound_counts

MAX_PAGE = 50
MAX_VALUE_BYTES = 2048
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,160}")
_DIGEST = re.compile(r"[a-f0-9]{64}")
_HOST_PATH = re.compile(r"(?:file://|/(?:home|Users|tmp|private|var|etc|opt)/|[A-Za-z]:\\)")


def mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def rows(value: Any) -> Sequence:
    return value if isinstance(value, (list, tuple)) else ()


def token(value: Any) -> str:
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else ""


def digest(value: Any) -> str:
    return value if isinstance(value, str) and _DIGEST.fullmatch(value) else ""


def pointer(*parts: object) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def safe_value(value: Any, depth: int = 0) -> Any:
    """Only bounded metadata; callers select fields before this final guard."""
    if depth > 6:
        return "[bounded]"
    if isinstance(value, Mapping):
        return {str(key)[:120]: safe_value(child, depth + 1)
                for key, child in list(value.items())[:32]
                if not any(term in str(key).lower() for term in ("path", "secret", "credential", "patient_id"))}
    if isinstance(value, (list, tuple)):
        return [safe_value(child, depth + 1) for child in value[:32]]
    if isinstance(value, str):
        return "[withheld]" if _HOST_PATH.search(value) else value[:500]
    if isinstance(value, float) and not math.isfinite(value):
        return "[unavailable]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


@dataclass(frozen=True)
class ReviewSource:
    artifact: str
    json_pointer: str = ""
    evidence_id: str = ""
    step_id: str = ""
    sha256: str = ""
    evidence_json_pointer: str = ""


class ReviewItems:
    """One request's detached items, joined to the run's artifact digests."""

    def __init__(self, artifacts: Sequence = ()) -> None:
        self.digests = {row.get("name"): digest(row.get("sha256")) for row in artifacts if isinstance(row, Mapping)}

    def item(self, code: str, value: Any, source: ReviewSource, **extra: Any) -> dict:
        projected = safe_value(project_outbound_counts(value))
        encoded = json.dumps(projected, ensure_ascii=True, allow_nan=False)
        truncated = len(encoded) > MAX_VALUE_BYTES
        return {
            "code": code, "text": code.removeprefix("SR_").replace("_", " ").capitalize() + ".",
            "value": "[see source]" if truncated else projected,
            "value_truncated": truncated,
            "source": {"artifact": source.artifact, "artifact_sha256": self.digests.get(source.artifact, ""),
                       "json_pointer": source.json_pointer if len(source.json_pointer) <= 512 else "", "evidence_id": token(source.evidence_id),
                       "step_id": token(source.step_id), "sha256": digest(source.sha256),
                       "evidence_json_pointer": source.evidence_json_pointer
                       if isinstance(source.evidence_json_pointer, str) and len(source.evidence_json_pointer) <= 512 else ""},
            **extra,
        }


def page(items: list[dict], *, offset: int, limit: int) -> dict:
    end = min(offset + limit, len(items))
    return {"items": items[offset:end], "total": len(items), "offset": offset,
            "next_offset": end if end < len(items) else None}
