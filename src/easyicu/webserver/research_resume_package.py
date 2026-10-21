"""Bounded package-verdict memo for continuation offers, never launch authority."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from easyicu.research_agent.intake.export_package import NATIVE_MANIFEST
from easyicu.webserver import dataio

_PACKAGE_LOCK = RLock()


def prepared_package_identity(
    raw_path: str, binding: Mapping[str, Any], *, verify_content: bool = False,
) -> str:
    """Fence a verified receipt by manifest bytes and every member's stat tuple.

    The receipt already names the exact data inventory. Its manifest also
    names metadata/feature sidecars; those must participate in invalidation.
    Same-inode, same-size edits with restored mtime are the residual case.
    Launch always validates bytes uncached, irrespective of this offer fence.
    The queued runner also hashes receipt-bound bytes before side effects;
    that fence needs no second schema validation or member snapshots.
    """
    root = Path(raw_path).expanduser().resolve(strict=True)
    if (root / NATIVE_MANIFEST).is_symlink():
        raise ValueError("the prepared package native marker is a symlink")
    manifest = dataio.prepared_export_manifest_path(root)
    if manifest is None or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("the prepared package manifest is unavailable")
    if manifest.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("the prepared package manifest exceeds its size bound")
    raw = manifest.read_bytes()
    expected = binding["manifest"]
    digest = hashlib.sha256(raw).hexdigest()
    if (manifest.name != expected["relative_path"] or digest != expected["sha256"]
            or len(raw) != expected["size_bytes"]):
        raise ValueError("the prepared package manifest changed")
    payload = json.loads(raw)
    digests = {item["relative_path"]: item["sha256"] for item in binding["files"]}
    names = set(digests)
    if payload.get("column_metadata"):
        name = payload["column_metadata"]["file"]
        names.add(name)
        digests[name] = binding["column_metadata_sha256"]
    features = payload.get("feature_definitions") or {}
    if features.get("included"):
        names.update(item["file"] for item in features["files"])
        digests.update({
            item["file"]: binding["feature_definitions_sha256"]
            for item in features["files"] if item.get("kind") == "feature_definitions"
        })
    members = []
    for name in sorted(names):
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("the prepared package member escapes its root")
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("the prepared package member traverses a symlink")
        if not path.is_file():
            raise ValueError("the prepared package member is unavailable")
        stat = path.stat()
        members.append((name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
        if verify_content and name in digests:
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digests[name]:
                    raise ValueError("the prepared package bytes changed after launch validation")
    return hashlib.sha256(json.dumps((digest, members), separators=(",", ":")).encode()).hexdigest()


def prepared_package_offer_matches(
    raw_path: str, *, database: Any, binding: Mapping[str, Any], identity: str,
) -> bool:
    """Snapshot at most once per complete package identity, caching either verdict."""
    binding_json = json.dumps(dict(binding), sort_keys=True, separators=(",", ":"))
    with _PACKAGE_LOCK:
        return _package_offer_verdict(raw_path, str(database or ""), binding_json, identity)


@lru_cache(maxsize=128)
def _package_offer_verdict(raw_path: str, database: str, binding_json: str, identity: str) -> bool:
    binding = json.loads(binding_json)
    try:
        dataio.validate_research_pipeline_source(raw_path, database=database, expected_binding=binding)
        # A mutation during the check cannot populate a verdict for old stats.
        return prepared_package_identity(raw_path, binding) == identity
    except (dataio.ExportCohortError, OSError, ValueError):
        return False
