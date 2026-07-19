"""Compatibility alias for :mod:`easyicu.research_agent.discovery.concept_proposal`."""

from __future__ import annotations

import sys as _sys

from .discovery import concept_proposal as _canonical

_sys.modules[__name__] = _canonical
