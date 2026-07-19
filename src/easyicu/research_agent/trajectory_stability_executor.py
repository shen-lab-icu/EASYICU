"""Compatibility alias for :mod:`easyicu.research_agent.execution.runners.trajectory_stability_executor`."""

from __future__ import annotations

import sys as _sys

from .execution.runners import trajectory_stability_executor as _canonical

_sys.modules[__name__] = _canonical
