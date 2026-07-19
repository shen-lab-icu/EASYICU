"""Compatibility alias for the canonical planning figure strategy."""

import sys as _sys

from .planning import figure_strategy as _canonical

_sys.modules[__name__] = _canonical
