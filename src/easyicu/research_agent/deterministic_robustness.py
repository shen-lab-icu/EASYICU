"""Compatibility alias for :mod:`easyicu.research_agent.execution.runners.deterministic_robustness`."""

from __future__ import annotations

import sys as _sys

from .execution.runners import deterministic_robustness as _canonical

_sys.modules[__name__] = _canonical
