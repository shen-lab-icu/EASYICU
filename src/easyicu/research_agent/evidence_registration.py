"""Compatibility alias for :mod:`easyicu.research_agent.authority.registration`.

The legacy module path is retained for archived scripts and public imports.
It aliases the canonical module object (rather than copying symbols) so
monkeypatching either path affects the implementation actually used.
"""

from __future__ import annotations

import sys as _sys

from .authority import registration as _canonical
from .authority.registration import (
    EvidencePromotionResult,
    EvidenceRegistrar,
    StepEvidenceCommit,
    filter_success_alias_bindings,
)

__all__ = [
    "EvidencePromotionResult",
    "EvidenceRegistrar",
    "StepEvidenceCommit",
    "filter_success_alias_bindings",
]

_sys.modules[__name__] = _canonical
