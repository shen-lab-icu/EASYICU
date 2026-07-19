"""Compatibility alias for the canonical planning analysis-type registry."""

import sys as _sys

from .planning import analysis_types as _canonical

_sys.modules[__name__] = _canonical
