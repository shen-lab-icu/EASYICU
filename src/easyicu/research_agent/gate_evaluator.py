"""Compatibility alias for :mod:`easyicu.research_agent.gates.visual`."""

from __future__ import annotations

import sys as _sys

from .gates import visual as _canonical

_sys.modules[__name__] = _canonical
