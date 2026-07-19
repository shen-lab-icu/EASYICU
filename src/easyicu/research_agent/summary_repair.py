"""Compatibility alias for :mod:`easyicu.research_agent.repairs.summary`."""

from __future__ import annotations

import sys as _sys

from .repairs import summary as _canonical

_sys.modules[__name__] = _canonical
