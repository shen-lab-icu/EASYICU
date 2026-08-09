"""Thin public facade for official, allowlisted demo-dataset preparation.

The immutable catalog/error contract lives in :mod:`demo_source_contracts`,
cache and archive security in :mod:`demo_source_storage`, and conversion/export
orchestration in :mod:`demo_source_prepare`.  Compatibility aliases below keep
the established import and test seams while dependencies are passed explicitly
to the runner at construction time.

No dataset bytes are stored in the repository.
"""

from __future__ import annotations

from typing import Any

from easyicu.webserver import dataio  # noqa: F401 - compatibility test seam
from easyicu.webserver import (  # noqa: F401 - compatibility test seam
    sources as source_store,
)
from easyicu.webserver import demo_source_prepare as _prepare
from easyicu.webserver import demo_source_storage as _storage
from easyicu.webserver.demo_source_contracts import (
    CACHE_ENV,
    MARKER_SCHEMA,
    SOURCE_BY_ID,
    SOURCES,
    DemoSourceCancelled,  # noqa: F401 - compatibility import
    DemoSourceError,
    DemoSourcePaths,
    DemoSourceSpec,
    allowed_source_ids,
    build_catalog,
    check_cancelled,
    diagnostic_json,
    get_source,
    is_cancel_requested,
    json_safe_detail,
)

# Stable compatibility aliases.  The public runner resolves its stage callables
# from these names at construction time, so existing monkeypatch seams continue
# to exercise the real orchestration boundary after the owner split.
_CACHE_ENV = CACHE_ENV
_DEFAULT_CACHE_ROOT = _storage.DEFAULT_CACHE_ROOT
_DOWNLOAD_CHUNK_BYTES = _storage.DOWNLOAD_CHUNK_BYTES
_MAX_ZIP_ENTRIES = _storage.MAX_ZIP_ENTRIES
_MARKER_SCHEMA = MARKER_SCHEMA
_PREPARE_LOCKS = _prepare._PREPARE_LOCKS
_PREPARE_LOCKS_GUARD = _prepare._PREPARE_LOCKS_GUARD
_SOURCES = SOURCES
_SOURCE_BY_ID = SOURCE_BY_ID
_SourcePaths = DemoSourcePaths

_json_safe_detail = json_safe_detail
_diagnostic_json = diagnostic_json
_is_cancel_requested = is_cancel_requested
_check_cancelled = check_cancelled

_cache_root = _storage.cache_root
_source_paths = _storage.source_paths
_now = _storage.now
_read_marker = _storage.read_marker
_write_marker = _storage.write_marker
_archive_ready = _storage.archive_ready
_raw_ready = _storage.raw_ready
_parquet_ready = _storage.parquet_ready
_export_ready = _storage.export_ready
_registry_state = _storage.registry_state
_status_payload = _storage.status_payload
_official_response = _storage.official_response
_validate_download_response_url = _storage.validate_download_response_url
_validate_official_response_url = _storage.validate_official_response_url
_stream_response = _storage.stream_response
_sha256_file = _storage.sha256_file
_download_archive = _storage.download_archive
_safe_zip_members = _storage.safe_zip_members
safe_extract_zip = _storage.safe_extract_zip
_replace_directory_atomically = _storage.replace_directory_atomically
_extract_archive = _storage.extract_archive

_ExportJobProxy = _prepare.ExportJobProxy
_convert_dataset = _prepare.convert_dataset
_export_dataset = _prepare.export_dataset
_register_export = _prepare.register_export
_prepare_lock = _prepare.prepare_lock
_lower_layer_diagnostic = _prepare.lower_layer_diagnostic
_run_prepare_stage = _prepare.run_prepare_stage


def demo_sources_catalog() -> dict[str, Any]:
    """Return public release metadata plus path-free readiness booleans."""

    return build_catalog(_status_payload)


def make_prepare_runner(source_id: str):
    """Return an idempotent runner using the five explicit owner operations."""

    source = get_source(source_id)
    paths = _source_paths(source)
    operations = _prepare.PrepareOperations(
        download=_download_archive,
        extract=_extract_archive,
        convert=_convert_dataset,
        export=_export_dataset,
        register=_register_export,
    )
    return _prepare.build_prepare_runner(source, paths, operations)


__all__ = [
    "DemoSourceError",
    "DemoSourceSpec",
    "allowed_source_ids",
    "demo_sources_catalog",
    "get_source",
    "make_prepare_runner",
    "safe_extract_zip",
]
