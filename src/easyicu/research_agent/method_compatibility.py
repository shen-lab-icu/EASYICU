"""Compatibility alias for the canonical method-compatibility gate."""

import sys as _sys

from .gates import method_compatibility as _canonical

_sys.modules[__name__] = _canonical
