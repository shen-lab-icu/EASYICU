"""Host-owned diagnostic payloads safe for external repair providers.

Raw candidate stdout/stderr remains local evidence. The external payload uses
validated repair authority and, separately, bounded untrusted input-error codes
whose column identities and dtypes are resolved from the declared context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..repairs.reasons import RepairPromptAuthority
from ..methods.descriptive_inputs import DESCRIPTIVE_INPUT_MARKER, DESCRIPTIVE_INPUT_REASON_CODES


_SCHEMA = "easyicu.diagnostic_envelope/1"
_ALLOWED_FAILURE_KINDS = frozenset(
    {"typed_validation_failure", "untyped_runtime_failure"}
)
_CODE_KEYS = frozenset(
    {"issue_code", "reason", "structured_reason", "validator"}
)


def descriptive_input_diagnostics(run_log: str, *, context: Any, step: Any) -> list[dict[str, Any]]:
    """Project untrusted helper notes onto declared columns and closed codes.

    These are candidate hints, never host findings or retry authority. Column
    identity and dtype come from the sealed context/step; no runtime dtype,
    count, value, path, arbitrary text or extra field can cross this boundary.
    """
    if context is None or step is None:
        return []
    declared = set(step.inputs)
    variables = {v.name: v for v in context.variables if v.name in declared}
    result = []
    for line in str(run_log or "")[-16000:].splitlines():
        if not line.startswith(DESCRIPTIVE_INPUT_MARKER) or len(line) > 1024:
            continue
        try:
            payload = json.loads(line[len(DESCRIPTIVE_INPUT_MARKER):])
        except (ValueError, RecursionError):
            continue
        if not isinstance(payload, dict) or payload.get("schema_version") != "easyicu.descriptive_input_error/1":
            continue
        column, codes = payload.get("column"), payload.get("reason_codes")
        if not isinstance(column, str) or column not in variables or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", column):
            continue
        if not isinstance(codes, list) or not codes or any(not isinstance(code, str) or code not in DESCRIPTIVE_INPUT_REASON_CODES for code in codes):
            continue
        item = {"column": column, "reason_codes": sorted(set(codes)),
                "dtype": variables[column].dtype, "provenance": "untrusted_runtime"}
        if item not in result:
            result.append(item)
        if len(result) == 8:
            break
    return result


def render_repair_diagnostic(*, llm, run_log, repair_authority, attempt,
                            byte_limit, context, step, local_excerpt) -> str:
    """Keep runtime hints separate from trusted repair and routing authority."""
    from ..providers.factory import provider_transport_destination

    hosted = os.environ.get("EASYICU_HOSTED") == "1"
    if hosted or provider_transport_destination(llm) == "external":
        return DiagnosticEnvelope.from_repair_authority(
            repair_authority, attempt=attempt, hosted=hosted,
            input_diagnostics=descriptive_input_diagnostics(run_log, context=context, step=step),
        ).render()
    return local_excerpt(run_log, byte_limit=byte_limit)


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
        if not isinstance(payload, dict) or set(payload) - {"input_diagnostics"} != {
            "schema_version",
            "failure_kind",
            "error_codes",
            "frames",
            "path_tokens",
            "fields",
            "repair_authority_sha256",
        }:
            raise ValueError("diagnostic envelope has an invalid schema")
        input_diagnostics = payload.get("input_diagnostics", [])
        if not isinstance(input_diagnostics, list) or len(input_diagnostics) > 8:
            raise ValueError("diagnostic envelope input diagnostics are invalid")
        for item in input_diagnostics:
            if (not isinstance(item, dict) or set(item) != {"column", "reason_codes", "dtype", "provenance"}
                    or item["provenance"] != "untrusted_runtime"
                    or not isinstance(item["column"], str)
                    or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", item["column"])
                    or not isinstance(item["dtype"], str)
                    or not isinstance(item["reason_codes"], list)
                    or not item["reason_codes"]
                    or any(not isinstance(code, str) or code not in DESCRIPTIVE_INPUT_REASON_CODES for code in item["reason_codes"])):
                raise ValueError("diagnostic envelope input diagnostics are invalid")
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
        hosted: bool = False,
        input_diagnostics: list[dict[str, Any]] | None = None,
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
        if hosted:
            codes.add("raw_diagnostics_unavailable_when_hosted")
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
        if input_diagnostics:
            payload["input_diagnostics"] = input_diagnostics
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
