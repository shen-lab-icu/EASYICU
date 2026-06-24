"""Compatibility shim for the legacy Streamlit concept catalog import path."""

from __future__ import annotations

from easyicu import concept_catalog as _shared_concept_catalog

for _name in dir(_shared_concept_catalog):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_shared_concept_catalog, _name)

__all__ = [
    _name
    for _name in dir(_shared_concept_catalog)
    if not _name.startswith("__")
]
