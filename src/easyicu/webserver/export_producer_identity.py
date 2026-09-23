"""Exact identity of the code and dictionaries that shape a native export.

A prepared export may be reused only while the extraction stack now running
produced it: an existing export folder proves nothing about the concepts,
receipts, or row boundaries the current code would write.  Like the Research
Agent's cache identity, this binds exact file bytes rather than a version label,
so a changed concept dictionary, score owner, or export writer makes an older
export visibly stale instead of silently reused.

The service computes it once per process, which is the code it will run.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

IDENTITY_SCHEMA = "easyicu.native_export_producer/1"

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Packages that never shape an export's rows, columns, or metadata.
_NON_PRODUCER_PACKAGES = frozenset(
    {"research_agent", "webserver", "visualization", "extensions", "scripts"}
)

#: The web modules that write the export: runner, manifest, column metadata,
#: availability receipts, cohort filter, identifier and raw-source contracts.
_WEB_EXPORT_WRITERS = (
    "dataio.py",
    "entity_ids.py",
    "primary_cohort.py",
    "raw_source_authority.py",
)


def _producer_files() -> list[Path]:
    files: list[Path] = []
    for path in _PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(_PACKAGE_ROOT)
        if "__pycache__" in relative.parts:
            continue
        if relative.parts[0] in _NON_PRODUCER_PACKAGES:
            continue
        files.append(path)
    # Packaged dictionaries and registries; the Research Agent's know-how
    # subfolder does not reach an export.
    files.extend((_PACKAGE_ROOT / "data").glob("*.json"))
    files.extend(_PACKAGE_ROOT / "webserver" / name for name in _WEB_EXPORT_WRITERS)
    return sorted(files)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def native_export_producer_identity() -> str:
    """Return ``sha256:<hex>`` over the producer's exact file bytes."""

    payload = {
        "schema": IDENTITY_SCHEMA,
        "files": {
            path.relative_to(_PACKAGE_ROOT).as_posix(): _sha256_file(path)
            for path in _producer_files()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["IDENTITY_SCHEMA", "native_export_producer_identity"]
