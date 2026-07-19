"""Compatibility alias for :mod:`easyicu.research_agent.discovery.discovery_handoff`."""

from __future__ import annotations

import sys as _sys

from .discovery import discovery_handoff as _canonical

_sys.modules[__name__] = _canonical
