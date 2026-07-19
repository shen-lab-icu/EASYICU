"""Compatibility alias for :mod:`easyicu.research_agent.reporting.display_suite`."""

from __future__ import annotations

import sys as _sys

from .reporting import display_suite as _canonical

_sys.modules[__name__] = _canonical
