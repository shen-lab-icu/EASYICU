"""Compatibility alias for :mod:`easyicu.research_agent.repairs.source`."""

from __future__ import annotations

import sys as _sys

from .repairs import source as _canonical

_sys.modules[__name__] = _canonical
