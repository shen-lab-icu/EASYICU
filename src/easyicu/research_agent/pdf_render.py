"""Compatibility alias for :mod:`easyicu.research_agent.reporting.pdf_render`."""

from __future__ import annotations

import sys as _sys

from .reporting import pdf_render as _canonical

_sys.modules[__name__] = _canonical
