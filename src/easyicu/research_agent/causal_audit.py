"""Compatibility alias for the canonical causal-claim review module."""

import sys as _sys

from .review import causal_audit as _canonical

_sys.modules[__name__] = _canonical
