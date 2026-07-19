"""Compatibility alias for the canonical methodological-rigor review module."""

import sys as _sys

from .review import methodological_rigor as _canonical

_sys.modules[__name__] = _canonical
