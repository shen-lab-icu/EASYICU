"""Source-bound synthetic provenance, egress scope, and persistent reader labels.

A report label is disclosure only. Clearance comes exclusively from verifying
the converted source against the deterministic generator at scope entry.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from easyicu.synthetic_sources import NOTICE, verify_source

_ACTIVE: ContextVar[dict | None] = ContextVar("verified_synthetic_source", default=None)


def _provenance(manifest):
    return {"source_kind": "synthetic", "contains_patient_data": False,
            "source_database": manifest["database"], "generator": manifest["schema_version"],
            "seed": manifest["seed"], "stays": manifest["stays"], "notice": NOTICE,
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}


def source_fields(path: str | Path) -> dict:
    from easyicu.synthetic_exports import verified_export_source

    manifest = verify_source(path) or verified_export_source(path)
    return {"source_kind": "synthetic", "provenance": _provenance(manifest)} if manifest else {}


def source_description(path: str | Path) -> dict | None:
    manifest = verify_source(path)
    if manifest is None:
        return None
    from easyicu.databases.profiles import get_database_profile

    files = [{"file": name, "rows": receipt["rows"], "module": name.split("/")[0].removesuffix(".parquet")}
             for name, receipt in manifest["files"].items()]
    return {"ok": True, "path": str(Path(path).absolute()), "database": manifest["database"],
            "label": f"Synthetic {get_database_profile(manifest['database']).display_name} — test data",
            "source_kind": "synthetic", "provenance": _provenance(manifest),
            "files": files, "modules": sorted({row["module"] for row in files}),
            "summary": {"stays": manifest["stays"], "file_count": len(files),
                        "total_rows": sum(row["rows"] for row in files)}}


def active_provenance() -> dict:
    return dict(_ACTIVE.get() or {})


def report_disclosure(text: str) -> dict:
    """Retain a saved report's label during reader rebuilding; never grants egress."""
    provenance = active_provenance()
    if not provenance and text.lstrip().startswith(f"> {NOTICE}"):
        return {"source_kind": "synthetic", "contains_patient_data": False, "notice": NOTICE}
    return provenance


@contextmanager
def synthetic_source_scope(path: str | Path):
    """Grant non-patient handling only for this verified database in this thread."""
    token = _ACTIVE.set(source_fields(path).get("provenance"))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def mark_report(text: str, provenance: dict | None = None) -> str:
    """Keep a saved report recognizably synthetic after its source scope closes."""
    provenance = provenance or active_provenance()
    if provenance.get("source_kind") == "synthetic" and NOTICE not in text:
        return f"> {NOTICE}\n\n{text}"
    return text


def write_report(path: Path, text: str, provenance: dict | None = None) -> str:
    """Persist the disclosure before the report's evidence digest is recorded."""
    text = mark_report(text, provenance)
    path.write_text(text, encoding="utf-8")
    return text
