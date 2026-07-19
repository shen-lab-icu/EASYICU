"""Compatibility alias for the canonical research-context builder."""

import sys as _sys

from .research_context import builder as _canonical

_sys.modules[__name__] = _canonical
