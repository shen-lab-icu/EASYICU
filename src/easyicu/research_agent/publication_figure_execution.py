"""Compatibility alias for :mod:`easyicu.research_agent.execution.publication_figure`."""

from __future__ import annotations

import sys as _sys

from .execution import publication_figure as _canonical
from .execution.publication_figure import (
    SealedRendererState,
    _deterministic_publication_figure_code,
    _sealed_parent_planner_anchors,
    _sealed_renderer_implementation_digest,
    _sealed_renderer_source_digests,
    _sealed_typed_figure_products,
)

__all__ = [
    "SealedRendererState",
    "_deterministic_publication_figure_code",
    "_sealed_renderer_source_digests",
    "_sealed_renderer_implementation_digest",
    "_sealed_parent_planner_anchors",
    "_sealed_typed_figure_products",
]

_sys.modules[__name__] = _canonical
