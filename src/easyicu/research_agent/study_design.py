"""Compatibility alias for :mod:`easyicu.research_agent.planning.study_design`."""

import sys as _sys

from .planning import study_design as _canonical

_sys.modules[__name__] = _canonical
