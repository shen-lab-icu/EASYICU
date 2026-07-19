"""Compatibility alias for the canonical planning playbook module."""

import sys as _sys

from .planning import study_design_playbook as _canonical

_sys.modules[__name__] = _canonical
