"""Compatibility alias for :mod:`easyicu.research_agent.repairs.helpers`."""

from __future__ import annotations

import sys as _sys

from .repairs import helpers as _canonical

_sys.modules[__name__] = _canonical
