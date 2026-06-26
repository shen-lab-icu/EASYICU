"""Compatibility shim for the shared EasyICU concept catalog.

The catalog now lives under :mod:`easyicu.concept.catalog`.  Keep this module
so production callers and tests that still import ``easyicu.concept_catalog``
continue to work while the package split is completed.
"""

from __future__ import annotations

from .concept.catalog import *  # noqa: F403
