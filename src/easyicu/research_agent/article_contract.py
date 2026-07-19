"""Compatibility alias for :mod:`easyicu.research_agent.reporting.article_contract`."""

from __future__ import annotations

import sys as _sys

from .reporting import article_contract as _canonical

_sys.modules[__name__] = _canonical
