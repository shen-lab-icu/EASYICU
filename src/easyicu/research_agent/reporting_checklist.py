"""Compatibility alias for :mod:`easyicu.research_agent.reporting.reporting_checklist`."""

from __future__ import annotations

import sys as _sys

from .reporting import reporting_checklist as _canonical

_sys.modules[__name__] = _canonical
