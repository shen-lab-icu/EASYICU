"""Dependency-neutral contracts and allowlist for official demo datasets.

This module owns the immutable public facts about supported releases and the
stable error contract shared by cache, archive, and preparation owners.  It
must remain independent of WebApp persistence, export, and transport modules.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

CACHE_ENV = "EASYICU_DEMO_CACHE_DIR"
MARKER_SCHEMA = "easyicu_official_demo_cache_v1"


def json_safe_detail(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Return a deterministic JSON-safe projection of lower-layer diagnostics."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if _depth >= 20:
        return "<maximum diagnostic depth>"

    seen = _seen if _seen is not None else set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            return "<recursive diagnostic>"
        seen.add(marker)
        try:
            return {
                str(key): json_safe_detail(
                    item,
                    _seen=seen,
                    _depth=_depth + 1,
                )
                for key, item in value.items()
            }
        finally:
            seen.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            return "<recursive diagnostic>"
        seen.add(marker)
        try:
            return [
                json_safe_detail(item, _seen=seen, _depth=_depth + 1) for item in value
            ]
        finally:
            seen.remove(marker)
    if isinstance(value, (set, frozenset)):
        projected = [
            json_safe_detail(item, _seen=seen, _depth=_depth + 1) for item in value
        ]
        return sorted(
            projected,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - diagnostics must never mask the cause.
        return f"<unprintable {type(value).__name__}>"


def diagnostic_json(code: str, detail: Any) -> str:
    """Serialize one stable diagnostic envelope."""

    return json.dumps(
        {
            "code": str(code),
            "detail": json_safe_detail(detail),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class DemoSourceError(RuntimeError):
    """A bounded, user-safe failure raised by the demo preparation pipeline."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        detail: Any = None,
    ) -> None:
        self.code = str(code or "demo_source_error")
        self.detail = json_safe_detail(
            detail if detail is not None else {"message": str(message)}
        )
        rendered = diagnostic_json(self.code, self.detail) if code else str(message)
        super().__init__(rendered)


class DemoSourceCancelled(RuntimeError):
    """Internal cooperative-cancellation signal."""


@dataclass(frozen=True)
class DemoSourceSpec:
    """Immutable allowlist entry for one official public demo release."""

    id: str
    title: str
    version: str
    database: str
    description: str
    scope_summary: str
    patients: int | None
    icu_stays: str | None
    size_label: str
    size_bytes: int
    download_url: str
    landing_page: str
    archive_filename: str
    attribution: str
    citation_url: str
    license_url: str
    max_download_bytes: int
    max_uncompressed_bytes: int
    archive_sha256: str | None = None
    mirror_url: str | None = None


@dataclass(frozen=True)
class DemoSourcePaths:
    """Resolved private cache paths passed between preparation owners."""

    root: Path
    archive: Path
    raw: Path
    export: Path
    extracted_marker: Path
    converted_marker: Path
    prepared_marker: Path


SOURCES: tuple[DemoSourceSpec, ...] = (
    DemoSourceSpec(
        id="mimic_iv_demo_v2_2",
        title="MIMIC-IV Clinical Database Demo",
        version="2.2",
        database="miiv",
        description=(
            "Official deidentified MIMIC-IV demonstration release for testing "
            "the complete EasyICU concept and visualization workflow."
        ),
        scope_summary="100 patients",
        patients=100,
        icu_stays=None,
        size_label="15.5 MB",
        size_bytes=16_189_661,
        download_url=(
            "https://physionet.org/static/published-projects/mimic-iv-demo/"
            "mimic-iv-clinical-database-demo-2.2.zip"
        ),
        landing_page="https://physionet.org/content/mimic-iv-demo/2.2/",
        archive_filename="mimic-iv-clinical-database-demo-2.2.zip",
        attribution=(
            "Johnson A, Bulgarelli L, Pollard T, Horng S, Celi LA, Mark R. "
            "MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. 2023."
        ),
        citation_url="https://doi.org/10.13026/dp1f-ex47",
        license_url="https://physionet.org/content/mimic-iv-demo/view-license/2.2/",
        max_download_bytes=32 * 1024 * 1024,
        max_uncompressed_bytes=768 * 1024 * 1024,
        archive_sha256=(
            "97301a03820e8f41af211cf3462ddc19aefe75bbed05f11753859affaafeb8ec"
        ),
        mirror_url=(
            "https://github.com/shen-lab-icu/EASYICU/releases/download/"
            "official-demo-data-v1/"
            "official-mimic-iv-clinical-database-demo-2.2.zip"
        ),
    ),
    DemoSourceSpec(
        id="eicu_demo_v2_0_1",
        title="eICU Collaborative Research Database Demo",
        version="2.0.1",
        database="eicu_demo",
        description=(
            "Official deidentified eICU demonstration release for testing "
            "multi-centre ICU concepts and the complete visualization workflow."
        ),
        scope_summary="2,500+ ICU stays",
        patients=None,
        icu_stays=">2500",
        size_label="130.6 MB",
        size_bytes=136_773_541,
        download_url=(
            "https://physionet.org/static/published-projects/eicu-crd-demo/"
            "eicu-collaborative-research-database-demo-2.0.1.zip"
        ),
        landing_page="https://physionet.org/content/eicu-crd-demo/2.0.1/",
        archive_filename="eicu-collaborative-research-database-demo-2.0.1.zip",
        attribution=(
            "Johnson A, Pollard T, Badawi O, Raffa J. eICU Collaborative "
            "Research Database Demo (version 2.0.1). PhysioNet. 2021."
        ),
        citation_url="https://doi.org/10.13026/4mxk-na84",
        license_url="https://physionet.org/content/eicu-crd-demo/view-license/2.0.1/",
        max_download_bytes=192 * 1024 * 1024,
        max_uncompressed_bytes=4 * 1024 * 1024 * 1024,
        archive_sha256=(
            "8e33a1094945d6ba07cf613b15b2fe4d98f6b3324601d026e80d445bd5b8b865"
        ),
        mirror_url=(
            "https://github.com/shen-lab-icu/EASYICU/releases/download/"
            "official-demo-data-v1/"
            "official-eicu-collaborative-research-database-demo-2.0.1.zip"
        ),
    ),
)
SOURCE_BY_ID: Mapping[str, DemoSourceSpec] = MappingProxyType(
    {source.id: source for source in SOURCES}
)


def allowed_source_ids() -> tuple[str, ...]:
    """Return the exact IDs accepted by the prepare endpoint."""

    return tuple(source.id for source in SOURCES)


def get_source(source_id: object) -> DemoSourceSpec:
    """Resolve one allowlisted source ID; never interpret it as a path or URL."""

    source = SOURCE_BY_ID.get(str(source_id or "").strip())
    if source is None:
        raise KeyError(str(source_id or ""))
    return source


def build_catalog(
    status_provider: Callable[[DemoSourceSpec], Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine immutable release facts with private-cache readiness booleans."""

    return {
        "ok": True,
        "cache": {
            "location": "user_cache",
            "override_env": CACHE_ENV,
        },
        "sources": [
            {
                "id": source.id,
                "title": source.title,
                "version": source.version,
                "database": source.database,
                "description": source.description,
                "scope": {
                    "summary": source.scope_summary,
                    "patients": source.patients,
                    "icu_stays": source.icu_stays,
                },
                "download": {
                    "size_label": source.size_label,
                    "size_bytes": source.size_bytes,
                    "preferred_transport": (
                        "github_release" if source.mirror_url else "physionet"
                    ),
                    "fallback_transport": (
                        "physionet" if source.mirror_url else None
                    ),
                    "mirror_url": source.mirror_url,
                    "sha256": source.archive_sha256,
                },
                "provenance": {
                    "provider": "PhysioNet",
                    "landing_page": source.landing_page,
                    "download_url": source.download_url,
                    "citation_url": source.citation_url,
                    "license": {
                        "name": "Open Data Commons Open Database License v1.0",
                        "short_name": "ODbL 1.0",
                        "url": source.license_url,
                        "terms_url": ("https://opendatacommons.org/licenses/odbl/1-0/"),
                        "attribution": source.attribution,
                    },
                },
                "status": dict(status_provider(source)),
            }
            for source in SOURCES
        ],
    }


def is_cancel_requested(job: Any) -> bool:
    """Read the two supported job cancellation contracts."""

    check = getattr(job, "is_cancel_requested", None)
    if callable(check):
        return bool(check())
    return bool(getattr(job, "cancel_requested", False))


def check_cancelled(job: Any, phase: str) -> None:
    """Raise the internal cancellation signal at a stage boundary."""

    if is_cancel_requested(job):
        job.emit({"type": "progress", "phase": phase, "stage": "cancelled"})
        raise DemoSourceCancelled(phase)


__all__ = [
    "CACHE_ENV",
    "MARKER_SCHEMA",
    "DemoSourceCancelled",
    "DemoSourceError",
    "DemoSourcePaths",
    "DemoSourceSpec",
    "SOURCES",
    "allowed_source_ids",
    "build_catalog",
    "check_cancelled",
    "diagnostic_json",
    "get_source",
    "is_cancel_requested",
    "json_safe_detail",
]
