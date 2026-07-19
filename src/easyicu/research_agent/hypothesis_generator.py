"""Compatibility alias for :mod:`easyicu.research_agent.discovery.hypothesis_generator`."""

from __future__ import annotations

import sys as _sys

from .discovery import hypothesis_generator as _canonical

_sys.modules[__name__] = _canonical
