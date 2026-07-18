"""Compatibility alias for :mod:`easyicu.research_agent.gates.concept`."""

from __future__ import annotations

import sys as _sys

from .gates import concept as _canonical
from .gates.concept import (
    DETERMINISTIC_CODE_GATE_VALIDATORS,
    deterministic_code_gate_findings,
    deterministic_gate_stamp,
    finding_detail_without_source_positions,
    finding_occurrence_identity,
    quarantined_deterministic_errors_resolved_by_current_gate,
    quarantined_errors_superseded_by_current_policy,
)

__all__ = [
    "DETERMINISTIC_CODE_GATE_VALIDATORS",
    "deterministic_code_gate_findings",
    "deterministic_gate_stamp",
    "finding_detail_without_source_positions",
    "finding_occurrence_identity",
    "quarantined_deterministic_errors_resolved_by_current_gate",
    "quarantined_errors_superseded_by_current_policy",
]

_sys.modules[__name__] = _canonical
