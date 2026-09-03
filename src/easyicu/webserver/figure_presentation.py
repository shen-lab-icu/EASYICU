"""Digest-bound presentation layer for registered Research Agent figures."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_MAX_MANIFEST_BYTES = 256_000
_MAX_PRESENTATION_FIGURES = 40
_MAX_VERIFIED_PNG_BYTES = 20_000_000
_HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _read_bounded_json(path: Path, *, max_bytes: int) -> Optional[Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        return None
    if len(raw) > max_bytes:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _clean_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _safe_relative(run_dir: Path, value: Any) -> Optional[Path]:
    relative = _clean_text(value, 300)
    if not relative or Path(relative).is_absolute():
        return None
    try:
        root = run_dir.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return path


def verified_presentation_gallery(
    run_dir: Path,
    canonical_source: Mapping[str, Any],
    *,
    embed_pngs: bool = False,
    max_png_bytes: int = 420_000,
    max_total_bytes: int = 1_400_000,
) -> Optional[Dict[str, Any]]:
    """Load the optional derived gallery only when all source digests match."""

    if not isinstance(canonical_source, Mapping):
        return None
    gallery_path = run_dir / "presentation_figures" / "presentation_figure_gallery.json"
    source = _read_bounded_json(gallery_path, max_bytes=_MAX_MANIFEST_BYTES)
    if not isinstance(source, Mapping):
        return None
    if source.get("schema_version") != "easyicu.presentation-figure-gallery/1":
        return None
    if source.get("status") != "presentation_only":
        return None
    if source.get("authority_ceiling") != "analysis_only":
        return None
    try:
        gallery_dir = gallery_path.parent.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    derived = source.get("derived_from")
    if (
        not isinstance(derived, Mapping)
        or derived.get("artifact") != "figure_gallery.json"
    ):
        return None
    canonical_path = run_dir / "figure_gallery.json"
    canonical_sha256 = _sha256_file(canonical_path)
    if canonical_sha256 is None:
        return None
    if derived.get("sha256") != canonical_sha256:
        return None
    bindings = source.get("source_bindings")
    if not isinstance(bindings, list) or not bindings or len(bindings) > 40:
        return None
    for binding in bindings:
        if not isinstance(binding, Mapping):
            return None
        expected = _clean_text(binding.get("sha256"), 64).lower()
        path = _safe_relative(run_dir, binding.get("relative_path"))
        if (
            path is None
            or not path.is_file()
            or not re.fullmatch(r"[a-f0-9]{64}", expected)
        ):
            return None
        if _sha256_file(path) != expected:
            return None
    figures = source.get("figures")
    if (
        not isinstance(figures, list)
        or not figures
        or len(figures) > _MAX_PRESENTATION_FIGURES
    ):
        return None
    projected_figures = []
    embedded_total = 0
    for row in figures:
        if not isinstance(row, Mapping):
            return None
        path = _safe_relative(run_dir, row.get("relative_path"))
        expected = _clean_text(row.get("sha256"), 64).lower()
        if (
            path is None
            or path.suffix.lower() != ".png"
            or path.parent != gallery_dir
            or not re.fullmatch(r"[a-f0-9]{64}", expected)
        ):
            return None
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size <= 0 or size > _MAX_VERIFIED_PNG_BYTES:
            return None
        if _sha256_file(path) != expected:
            return None
        projected = dict(row)
        if embed_pngs:
            if (
                size <= max_png_bytes
                and embedded_total + size <= max_total_bytes
            ):
                try:
                    png_bytes = path.read_bytes()
                except OSError:
                    return None
                projected["data_url"] = "data:image/png;base64," + base64.b64encode(
                    png_bytes
                ).decode("ascii")
                embedded_total += size
        projected_figures.append(projected)
    result = dict(source)
    result["figures"] = projected_figures
    result["presentation_variant"] = True
    result["original_run_figures_preserved"] = True
    if embed_pngs:
        result["embedded_count"] = sum(
            1 for row in projected_figures if row.get("data_url")
        )
    return result
