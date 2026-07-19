"""Compatibility alias for :mod:`easyicu.research_agent.execution.runners.deterministic_cohort_flow`."""

from __future__ import annotations

import sys as _sys

from .execution.runners import deterministic_cohort_flow as _canonical

_sys.modules[__name__] = _canonical
