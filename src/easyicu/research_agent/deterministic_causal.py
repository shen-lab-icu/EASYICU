"""Compatibility alias for :mod:`easyicu.research_agent.execution.runners.deterministic_causal`."""

from __future__ import annotations

import sys as _sys

from .execution.runners import deterministic_causal as _canonical

_sys.modules[__name__] = _canonical
