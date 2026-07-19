"""Compatibility alias for the canonical typed research-context module."""

import sys as _sys

from .research_context import typed as _canonical

_sys.modules[__name__] = _canonical
