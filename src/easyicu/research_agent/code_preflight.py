"""Compatibility alias for :mod:`easyicu.research_agent.gates.preflight`."""

from __future__ import annotations

import sys as _sys

from .gates import preflight as _canonical

_sys.modules[__name__] = _canonical
