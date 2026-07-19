"""Compatibility alias for :mod:`easyicu.research_agent.reporting.manuscript_post`."""

from __future__ import annotations

import sys as _sys

from .reporting import manuscript_post as _canonical

_sys.modules[__name__] = _canonical
