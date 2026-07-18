"""Stable implementation identities for typed ResearchContext metadata.

The typed metadata projector and sidecar schema live outside ``research_agent``
while ICU analysis-plausibility rules live inside it.  All three influence the
physical facts shown to Planner/Coder, so cache and resume authority must bind
their exact implementation bytes rather than relying on package version labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_implementation_identity() -> Mapping[str, str]:
    """Return the exact code identities that shape typed context facts."""

    easyicu_root = Path(__file__).resolve().parents[1]
    implementation = {
        "metadata_projection_sha256": _sha256_file(
            easyicu_root / "concept" / "metadata_projection.py"
        ),
        "metadata_sidecar_sha256": _sha256_file(
            easyicu_root / "concept" / "metadata_sidecar.py"
        ),
        "icu_rules_sha256": _sha256_file(Path(__file__).with_name("icu_rules.py")),
    }
    payload = json.dumps(
        implementation,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **implementation,
        "metadata_implementation_bundle_sha256": hashlib.sha256(payload).hexdigest(),
    }


def metadata_implementation_bundle_sha256() -> str:
    """Digest the canonical implementation-identity payload."""

    return metadata_implementation_identity()["metadata_implementation_bundle_sha256"]


__all__ = [
    "metadata_implementation_bundle_sha256",
    "metadata_implementation_identity",
]
