"""Shared fail-closed semantics for deterministic validator findings."""

from __future__ import annotations

from typing import List, Sequence

from ..schema import ValidationFinding


def blocking_validator_findings(
    *finding_groups: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Return findings that block execution, sealing, or current authority."""

    return [
        finding
        for group in finding_groups
        for finding in group
        if finding.severity == "error"
    ]


__all__ = ["blocking_validator_findings"]
