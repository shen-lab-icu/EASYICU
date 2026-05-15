"""Common contract for audit classes.

This module defines a small Protocol every audit class in this
subpackage should converge on. It is intentionally light-weight:

* ``Audit``  – a runtime-checkable Protocol with a single ``run``
  entry point. Existing classes (``CohortAuditor``, ``ConceptUsageAuditor``,
  ``StatisticalValidator``, ``AnalysisPatternAuditor``, ...) already
  expose ``run`` / ``audit`` methods with similar shapes; this contract
  documents the convention so new audits can be added consistently.
* ``AuditSeverity`` – enum-like literal alias for the existing
  ``ValidationFinding.severity`` strings.
* ``AuditReport`` – the recommended return type for new audits. Existing
  audits return ``Sequence[ValidationFinding]`` directly; that remains
  valid (``AuditReport.findings`` is a thin wrapper around the same
  list). New audits SHOULD return ``AuditReport`` so callers can attach
  a top-level ``passed`` flag and provenance metadata without parsing
  individual findings.

Naming convention
-----------------
Historically classes here used ``Auditor`` / ``Validator`` / ``Guard`` /
``Comparator`` interchangeably. New audit classes SHOULD use the
``Audit`` suffix (e.g. ``CohortAudit``, ``ConceptUsageAudit``). Existing
names are kept for backwards compatibility — see
:mod:`easyicu.research_agent.audits` for re-exports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from ..schema import ValidationFinding

AuditSeverity = Literal["error", "warning", "info"]


@dataclass
class AuditReport:
    """Structured result returned by an :class:`Audit`.

    ``findings`` mirrors the existing ``Sequence[ValidationFinding]``
    return shape used by the older auditor classes, so call sites that
    already iterate over findings keep working when they receive an
    ``AuditReport``.
    """

    findings: Sequence[ValidationFinding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)


@runtime_checkable
class Audit(Protocol):
    """Protocol every audit class in this subpackage should follow.

    The ``ctx`` argument is intentionally typed as ``Any`` because each
    concrete audit takes a different combination of inputs (cohort
    frame, generated script, manuscript text, ...). This Protocol
    documents the *shape* of the entry point, not the input schema.
    """

    name: str

    def run(self, ctx: Any) -> AuditReport | Sequence[ValidationFinding]:
        ...


__all__ = ["Audit", "AuditReport", "AuditSeverity"]
