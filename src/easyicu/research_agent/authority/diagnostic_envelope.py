"""Host-owned diagnostic payloads safe for external repair providers.

Raw candidate stdout/stderr is deliberately not an input to the envelope
projection. It remains local evidence. The external payload is derived only
from the validated :class:`RepairPromptAuthority` and closed host coordinates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..repairs.reasons import RepairPromptAuthority


_SCHEMA = "easyicu.diagnostic_envelope/1"
_ALLOWED_FAILURE_KINDS = frozenset(
    {"typed_validation_failure", "untyped_runtime_failure"}
)
_CODE_KEYS = frozenset(
    {"issue_code", "reason", "structured_reason", "validator"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _closed_codes(value: Any) -> set[str]:
    """Collect closed authority tokens without copying arbitrary values."""

    codes: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in _CODE_KEYS and isinstance(nested, str) and nested:
                codes.add(nested)
            elif key == "issue_code" and isinstance(nested, list):
                codes.update(item for item in nested if isinstance(item, str) and item)
            codes.update(_closed_codes(nested))
    elif isinstance(value, list):
        for nested in value:
            codes.update(_closed_codes(nested))
    return codes


def _occurrence_count(ticket: list[dict[str, Any]]) -> int:
    total = 0
    for item in ticket:
        value = item.get("occurrence_count")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            total += value
        elif isinstance(item.get("occurrences"), list):
            total += len(item["occurrences"])
        else:
            total += 1
    return total


@dataclass(frozen=True)
class DiagnosticEnvelope:
    """Canonical external diagnostic with no candidate-controlled prose."""

    canonical_json: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.canonical_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("diagnostic envelope is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "failure_kind",
            "error_codes",
            "frames",
            "path_tokens",
            "fields",
            "repair_authority_sha256",
        }:
            raise ValueError("diagnostic envelope has an invalid schema")
        if payload["schema_version"] != _SCHEMA:
            raise ValueError("diagnostic envelope has an unsupported version")
        if payload["failure_kind"] not in _ALLOWED_FAILURE_KINDS:
            raise ValueError("diagnostic envelope failure_kind is invalid")
        codes = payload["error_codes"]
        if not isinstance(codes, list) or codes != sorted(set(codes)) or any(
            not isinstance(item, str) or not item for item in codes
        ):
            raise ValueError("diagnostic envelope error_codes are invalid")
        if payload["frames"] != [] or payload["path_tokens"] != []:
            raise ValueError(
                "diagnostic envelope v1 does not transport stack or path values"
            )
        fields = payload["fields"]
        if not isinstance(fields, dict) or set(fields) != {
            "attempt",
            "finding_count",
        }:
            raise ValueError("diagnostic envelope fields are invalid")
        if any(
            not isinstance(fields[key], int)
            or isinstance(fields[key], bool)
            or fields[key] < 0
            for key in fields
        ):
            raise ValueError("diagnostic envelope fields must be non-negative ints")
        digest = payload["repair_authority_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("diagnostic envelope authority digest is invalid")
        if self.canonical_json != _canonical_json(payload):
            raise ValueError("diagnostic envelope JSON must be canonical")

    @classmethod
    def from_repair_authority(
        cls,
        authority: RepairPromptAuthority,
        *,
        attempt: int,
    ) -> "DiagnosticEnvelope":
        """Project validated authority without accepting any runtime log."""

        if not isinstance(authority, RepairPromptAuthority):
            raise TypeError("diagnostic envelope requires RepairPromptAuthority")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError("diagnostic envelope attempt must be non-negative")
        authority_payload = authority.payload()
        ticket = authority_payload["typed_ticket"]
        codes = _closed_codes(ticket)
        codes.update(authority_payload["route_codes"])
        payload = {
            "schema_version": _SCHEMA,
            "failure_kind": (
                "untyped_runtime_failure"
                if authority.is_empty
                else "typed_validation_failure"
            ),
            "error_codes": sorted(codes),
            # Future stack/path additions require a closed host-owned registry.
            "frames": [],
            "path_tokens": [],
            "fields": {
                "attempt": attempt,
                "finding_count": _occurrence_count(ticket),
            },
            "repair_authority_sha256": hashlib.sha256(
                authority.canonical_json.encode("utf-8")
            ).hexdigest(),
        }
        return cls(canonical_json=_canonical_json(payload))

    def payload(self) -> dict[str, Any]:
        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):  # pragma: no cover
            raise AssertionError("validated diagnostic envelope must be a mapping")
        return payload

    def render(self) -> str:
        return json.dumps(
            self.payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )


__all__ = ["DiagnosticEnvelope"]
