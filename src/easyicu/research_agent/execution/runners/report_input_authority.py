"""Verify typed inputs consumed by deterministic report-only executors.

This module owns one small boundary: a report may cite an existing typed
product only when the resolved-input manifest binds the exact product key to
contained bytes whose SHA-256 still matches.  It deliberately does not parse
tables or infer scientific meaning; analysis executors own those jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ...contracts.declared_product import RUNTIME_BINDABLE_TYPED_INPUT_KINDS
from .typed_input_binding import contained_regular_file, sha256_file


_TYPED_KEY = re.compile(r"([a-z][a-z0-9_]*):([a-z][a-z0-9_]*)")


@dataclass(frozen=True, slots=True)
class BoundReportInputAuthority:
    """One verified, non-computational report input."""

    input_key: str
    evidence_id: str
    sha256: str
    relative_path: str
    produced_by_step: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_key": self.input_key,
            "evidence_id": self.evidence_id,
            "sha256": self.sha256,
            "relative_path": self.relative_path,
            "produced_by_step": self.produced_by_step,
        }


def verify_report_input_authorities(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    declared_inputs: Sequence[str],
) -> tuple[BoundReportInputAuthority, ...]:
    """Return exact report authorities or fail closed on any mismatch."""

    payload = (
        dict(resolved_inputs)
        if isinstance(resolved_inputs, Mapping)
        else json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("report input manifest does not belong to this step")
    bindings = payload.get("inputs")
    if not isinstance(bindings, dict):
        raise ValueError("report input manifest carries no binding map")
    if set(declared_inputs) - set(bindings):
        raise ValueError("report input authority is incomplete")
    if set(bindings) - set(declared_inputs):
        raise ValueError("report input manifest does not match declared inputs")

    resolved_run_dir = Path(run_dir).resolve()
    authorities: list[BoundReportInputAuthority] = []
    for input_key in declared_inputs:
        match = _TYPED_KEY.fullmatch(str(input_key or "").strip())
        binding = bindings.get(input_key)
        if (
            match is None
            or match.group(1) not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS
            or not isinstance(binding, Mapping)
        ):
            raise ValueError("report input authority is incomplete")

        digest = str(binding.get("sha256") or "")
        relative_path = str(binding.get("relative_path") or "")
        relative = Path(relative_path)
        candidate = resolved_run_dir / relative
        identity = binding.get("identity_row")
        bound_path = contained_regular_file(candidate, resolved_run_dir)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not relative_path
            or relative.is_absolute()
            or bound_path is None
            or sha256_file(bound_path) != digest
            or str(binding.get("declared_kind") or "") != match.group(1)
            or not isinstance(identity, Mapping)
            or str(identity.get("input_key") or "") != input_key
        ):
            raise ValueError("report input lacks an exact digest binding")

        evidence_id = str(binding.get("evidence_id") or "")
        if not evidence_id:
            raise ValueError("report input authority has no evidence id")
        produced_by_step = binding.get("produced_by_step")
        authorities.append(
            BoundReportInputAuthority(
                input_key=input_key,
                evidence_id=evidence_id,
                sha256=digest,
                relative_path=relative_path,
                produced_by_step=(
                    str(produced_by_step) if produced_by_step is not None else None
                ),
            )
        )
    return tuple(authorities)


__all__ = [
    "BoundReportInputAuthority",
    "verify_report_input_authorities",
]
