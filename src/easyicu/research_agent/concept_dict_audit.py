"""[Layer 4: Evidence & Provenance] Concept-dictionary drift audit.

The research agent treats EasyICU concept dictionaries as executable clinical
metadata. A replay is only comparable to the original run when the on-disk
``concept-dict.json`` and ``sofa2-dict.json`` match the hashes recorded in the
run manifest or locked submission profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

CONCEPT_DICT_PACKAGE_PATH = "easyicu/data/concept-dict.json"
SOFA2_DICT_PACKAGE_PATH = "easyicu/data/sofa2-dict.json"


class ConceptDictDriftError(RuntimeError):
    """Raised when on-disk concept dictionaries differ from a locked SHA."""


@dataclass(frozen=True)
class ConceptDictFingerprint:
    """SHA-256 identity for concept dictionaries used by a run."""

    concept_dict_path: str
    concept_dict_sha: str
    sofa2_dict_path: str
    sofa2_dict_sha: str
    computed_at: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dict_path(package_relative_path: str) -> Path:
    return _package_root() / "data" / Path(package_relative_path).name


def compute_concept_dict_fingerprint() -> ConceptDictFingerprint:
    """Return a SHA fingerprint for the current packaged concept dictionaries."""

    concept_path = _dict_path(CONCEPT_DICT_PACKAGE_PATH)
    sofa2_path = _dict_path(SOFA2_DICT_PACKAGE_PATH)
    missing = [str(p) for p in (concept_path, sofa2_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing concept dictionary file(s): " + ", ".join(missing)
        )
    return ConceptDictFingerprint(
        concept_dict_path=CONCEPT_DICT_PACKAGE_PATH,
        concept_dict_sha=_sha256_file(concept_path),
        sofa2_dict_path=SOFA2_DICT_PACKAGE_PATH,
        sofa2_dict_sha=_sha256_file(sofa2_path),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def assert_dict_matches(
    fingerprint: ConceptDictFingerprint,
    *,
    expected_concept_dict_sha: Optional[str] = None,
    expected_sofa2_dict_sha: Optional[str] = None,
    mode: Literal["strict", "soft"] = "strict",
) -> List[str]:
    """Compare current dictionary hashes to locked expectations.

    In ``strict`` mode a mismatch raises :class:`ConceptDictDriftError`. In
    ``soft`` mode all mismatch messages are returned for manifest/logging use.
    """

    warnings: List[str] = []
    if (
        expected_concept_dict_sha
        and fingerprint.concept_dict_sha != expected_concept_dict_sha
    ):
        warnings.append(
            "concept-dict.json SHA mismatch: "
            f"expected={expected_concept_dict_sha} actual={fingerprint.concept_dict_sha}"
        )
    if expected_sofa2_dict_sha and fingerprint.sofa2_dict_sha != expected_sofa2_dict_sha:
        warnings.append(
            "sofa2-dict.json SHA mismatch: "
            f"expected={expected_sofa2_dict_sha} actual={fingerprint.sofa2_dict_sha}"
        )
    if warnings and mode == "strict":
        raise ConceptDictDriftError("; ".join(warnings))
    return warnings


def write_concept_dict_fingerprint(path: Path) -> ConceptDictFingerprint:
    """Write the current dictionary fingerprint to ``path`` and return it."""

    fingerprint = compute_concept_dict_fingerprint()
    path.write_text(json.dumps(fingerprint.to_dict(), indent=2), encoding="utf-8")
    return fingerprint


def _manifest_expected_shas(manifest: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    fingerprint = manifest.get("concept_dict_fingerprint")
    if isinstance(fingerprint, dict):
        return (
            _string_or_none(fingerprint.get("concept_dict_sha")),
            _string_or_none(fingerprint.get("sofa2_dict_sha")),
        )
    return (
        _string_or_none(manifest.get("concept_dict_sha")),
        _string_or_none(manifest.get("sofa2_dict_sha")),
    )


def _string_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def verify_replay_dict_match(
    run_dir: Path | str,
    *,
    mode: Literal["strict", "soft"] = "strict",
) -> List[str]:
    """Verify local concept dictionaries match the hashes in ``manifest.json``."""

    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_concept_sha, expected_sofa2_sha = _manifest_expected_shas(manifest)
    if not expected_concept_sha and not expected_sofa2_sha:
        warnings = ["manifest has no concept dictionary fingerprint to verify"]
        if mode == "strict":
            raise ConceptDictDriftError(warnings[0])
        return warnings
    return assert_dict_matches(
        compute_concept_dict_fingerprint(),
        expected_concept_dict_sha=expected_concept_sha,
        expected_sofa2_dict_sha=expected_sofa2_sha,
        mode=mode,
    )


__all__ = [
    "CONCEPT_DICT_PACKAGE_PATH",
    "SOFA2_DICT_PACKAGE_PATH",
    "ConceptDictDriftError",
    "ConceptDictFingerprint",
    "assert_dict_matches",
    "compute_concept_dict_fingerprint",
    "verify_replay_dict_match",
    "write_concept_dict_fingerprint",
]
