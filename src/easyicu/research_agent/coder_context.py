"""Compatibility alias for the canonical scoped Coder context module."""

import sys as _sys

from .research_context import prompt_scope as _canonical

_sys.modules[__name__] = _canonical
