"""Compatibility alias for :mod:`easyicu.research_agent.discovery.idea_mining_eval`."""

from __future__ import annotations

import sys as _sys

from .discovery import idea_mining_eval as _canonical

_sys.modules[__name__] = _canonical
