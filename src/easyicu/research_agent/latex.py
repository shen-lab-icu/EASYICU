"""Compatibility alias for :mod:`easyicu.research_agent.reporting.latex`."""

from __future__ import annotations

import sys as _sys

from .reporting import latex as _canonical

_sys.modules[__name__] = _canonical
