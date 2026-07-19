"""Compatibility alias for the canonical Tier-2 jury module."""

import sys as _sys

from .evaluation import tier2_jury as _canonical

_sys.modules[__name__] = _canonical
