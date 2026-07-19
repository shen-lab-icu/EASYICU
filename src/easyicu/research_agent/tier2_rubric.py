"""Compatibility alias for the canonical Tier-2 rubric module."""

import sys as _sys

from .evaluation import tier2_rubric as _canonical

_sys.modules[__name__] = _canonical
