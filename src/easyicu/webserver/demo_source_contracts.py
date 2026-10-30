"""Compatibility imports for the shared official-demo source contract."""

from ..official_demo_sources import (
    CACHE_ENV, MARKER_SCHEMA, SOURCE_BY_ID, SOURCES,
    DemoSourceCancelled, DemoSourceError, DemoSourcePaths, DemoSourceSpec,
    allowed_source_ids, build_catalog, check_cancelled, diagnostic_json,
    get_source, is_cancel_requested, json_safe_detail,
)

__all__ = [
    "CACHE_ENV", "MARKER_SCHEMA", "SOURCE_BY_ID", "SOURCES",
    "DemoSourceCancelled", "DemoSourceError", "DemoSourcePaths", "DemoSourceSpec",
    "allowed_source_ids", "build_catalog", "check_cancelled", "diagnostic_json",
    "get_source", "is_cancel_requested", "json_safe_detail",
]
