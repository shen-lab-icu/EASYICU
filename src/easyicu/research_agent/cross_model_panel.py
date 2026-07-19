"""Compatibility alias for the canonical cross-model evaluation module."""

import sys as _sys

from .evaluation import cross_model_panel as _canonical

_sys.modules[__name__] = _canonical
