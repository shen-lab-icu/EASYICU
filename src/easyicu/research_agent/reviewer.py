"""Compatibility alias for :mod:`easyicu.research_agent.reporting.reviewer`."""

from __future__ import annotations

import sys as _sys

from .reporting import reviewer as _canonical

_sys.modules[__name__] = _canonical
