"""Compatibility alias for :mod:`easyicu.research_agent.execution.concept_audit`."""

from __future__ import annotations

import sys as _sys

from .execution import concept_audit as _canonical
from .execution.concept_audit import (
    ConceptAuditAuthority,
    ConceptAuditCoordinator,
    ConceptAuditRuntime,
    ConceptQuarantineState,
)

__all__ = [
    "ConceptAuditAuthority",
    "ConceptAuditCoordinator",
    "ConceptAuditRuntime",
    "ConceptQuarantineState",
]

_sys.modules[__name__] = _canonical
