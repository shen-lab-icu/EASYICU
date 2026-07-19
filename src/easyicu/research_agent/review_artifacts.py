"""Compatibility alias for :mod:`easyicu.research_agent.reporting.review_artifacts`."""

from __future__ import annotations

import sys as _sys

from .reporting import review_artifacts as _canonical

_sys.modules[__name__] = _canonical
