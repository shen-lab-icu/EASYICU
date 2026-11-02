"""[Layer 4: Evidence & Provenance] Concept-dictionary drift audit.

The research agent treats EasyICU concept dictionaries as executable clinical
metadata. A replay is only comparable to the original run when the on-disk
``concept-dict.json`` and ``sofa2-dict.json`` match the hashes recorded in the
run manifest or locked submission profile.
"""

from __future__ import annotations

from .canonical_json import sha256_file as _sha256_file

import json
import hashlib
import re
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
    sic_case_overlay_sha: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    from easyicu.resources import SIC_CASE_OVERLAY_FILE, _sic_case_overlay
    _sic_case_overlay()
    overlay_path = _dict_path(SIC_CASE_OVERLAY_FILE)
    return ConceptDictFingerprint(
        concept_dict_path=CONCEPT_DICT_PACKAGE_PATH,
        concept_dict_sha=_sha256_file(concept_path),
        sofa2_dict_path=SOFA2_DICT_PACKAGE_PATH,
        sofa2_dict_sha=_sha256_file(sofa2_path),
        computed_at=datetime.now(timezone.utc).isoformat(),
        sic_case_overlay_sha=_sha256_file(overlay_path),
    )


def assert_dict_matches(
    fingerprint: ConceptDictFingerprint,
    *,
    expected_concept_dict_sha: Optional[str] = None,
    expected_sofa2_dict_sha: Optional[str] = None,
    expected_sic_case_overlay_sha: Optional[str] = None,
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
    if (
        expected_sofa2_dict_sha
        and fingerprint.sofa2_dict_sha != expected_sofa2_dict_sha
    ):
        warnings.append(
            "sofa2-dict.json SHA mismatch: "
            f"expected={expected_sofa2_dict_sha} actual={fingerprint.sofa2_dict_sha}"
        )
    if (expected_concept_dict_sha or expected_sofa2_dict_sha or expected_sic_case_overlay_sha) and fingerprint.sic_case_overlay_sha != expected_sic_case_overlay_sha:
        warnings.append(
            "sic case overlay SHA mismatch: "
            f"expected={expected_sic_case_overlay_sha} actual={fingerprint.sic_case_overlay_sha}"
        )
    if warnings and mode == "strict":
        raise ConceptDictDriftError("; ".join(warnings))
    return warnings


def write_concept_dict_fingerprint(path: Path) -> ConceptDictFingerprint:
    """Write the current dictionary fingerprint to ``path`` and return it."""

    fingerprint = compute_concept_dict_fingerprint()
    path.write_text(json.dumps(fingerprint.to_dict(), indent=2), encoding="utf-8")
    return fingerprint


def sealed_run_dictionary_manifest_fields(run_dir: Path, evidence: Any = None) -> Dict[str, Any]:
    """Read the original run's verified fingerprint; never adopt today's bytes.

    The evidence is registered before execution. Every checkpoint and final
    manifest must carry those same resource coordinates, and a live package
    change stops publication rather than being recorded as the run's origin.
    """
    from .authority.runtime_artifacts import verified_run_evidence_path
    from .authority.filesystem import AnchoredDirectory, AuthorityFilesystemError

    if evidence is None:
        from .authority.evidence_snapshot import load_current_evidence_snapshot
        snapshot = load_current_evidence_snapshot(run_dir)
        records = [item for item in snapshot.records if item.get("evidence_id") == "concept_dict_fingerprint"]
        record = records[0] if len(records) == 1 else None
    else:
        record = evidence.get("concept_dict_fingerprint")
    path = verified_run_evidence_path(run_dir, record) if record is not None else None
    if path is None:
        raise ConceptDictDriftError("run dictionary fingerprint evidence missing or changed")
    try:
        with AnchoredDirectory.open(path.parent) as directory:
            raw = directory.read_bytes(path.name, max_bytes=16384)
            directory.assert_still_selected()
        record_sha = record.get("sha256") if isinstance(record, dict) else record.sha256
        if len(raw) > 16384 or hashlib.sha256(raw).hexdigest() != record_sha:
            raise ConceptDictDriftError("run dictionary fingerprint evidence changed")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AuthorityFilesystemError) as exc:
        raise ConceptDictDriftError("run dictionary fingerprint evidence unreadable") from exc
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", data[key])
        for key in ("concept_dict_sha", "sofa2_dict_sha", "sic_case_overlay_sha")
    ):
        raise ConceptDictDriftError("run dictionary fingerprint evidence incomplete")
    if (data.get("concept_dict_path") != CONCEPT_DICT_PACKAGE_PATH
            or data.get("sofa2_dict_path") != SOFA2_DICT_PACKAGE_PATH):
        raise ConceptDictDriftError("run dictionary fingerprint evidence paths invalid")
    assert_dict_matches(
        compute_concept_dict_fingerprint(), expected_concept_dict_sha=data["concept_dict_sha"],
        expected_sofa2_dict_sha=data["sofa2_dict_sha"],
        expected_sic_case_overlay_sha=data["sic_case_overlay_sha"], mode="strict",
    )
    return {"concept_dict_path": data["concept_dict_path"], "concept_dict_sha": data["concept_dict_sha"],
            "sofa2_dict_path": data["sofa2_dict_path"], "sofa2_dict_sha": data["sofa2_dict_sha"],
            "concept_dict_fingerprint": data}


def verify_run_dictionary_checkpoint(run_dir: Path, manifest: Dict[str, Any],
                                     expected: tuple[Optional[str], Optional[str], Optional[str]] | None = None) -> None:
    """Bind the selected checkpoint to the run's immutable evidence before writes."""
    verify_recorded_dict_match(manifest, mode="strict", require_recorded=True)
    sealed = sealed_run_dictionary_manifest_fields(run_dir)
    if manifest.get("concept_dict_fingerprint") != sealed["concept_dict_fingerprint"]:
        raise ConceptDictDriftError("checkpoint dictionary fingerprint differs from sealed run evidence")
    if any(key in manifest and manifest[key] != sealed[key] for key in (
        "concept_dict_path", "concept_dict_sha", "sofa2_dict_path", "sofa2_dict_sha",
    )):
        raise ConceptDictDriftError("checkpoint dictionary fields contradict sealed run evidence")
    if expected is not None:
        assert_dict_matches(
            ConceptDictFingerprint(**sealed["concept_dict_fingerprint"]),
            expected_concept_dict_sha=expected[0], expected_sofa2_dict_sha=expected[1],
            expected_sic_case_overlay_sha=expected[2],
            mode="strict",
        )


def _manifest_expected_shas(
    manifest: Dict[str, Any],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    fingerprint = manifest.get("concept_dict_fingerprint")
    if isinstance(fingerprint, dict):
        return (
            _string_or_none(fingerprint.get("concept_dict_sha")),
            _string_or_none(fingerprint.get("sofa2_dict_sha")),
            _string_or_none(fingerprint.get("sic_case_overlay_sha")),
        )
    return (
        _string_or_none(manifest.get("concept_dict_sha")),
        _string_or_none(manifest.get("sofa2_dict_sha")),
        _string_or_none(manifest.get("sic_case_overlay_sha")),
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
    return verify_recorded_dict_match(manifest, mode=mode, require_recorded=True)


def verify_recorded_dict_match(
    manifest: Dict[str, Any],
    *,
    mode: Literal["strict", "soft"] = "strict",
    require_recorded: bool = False,
) -> List[str]:
    """Verify current package dictionaries against one manifest authority.

    Resume callers already hold the checkpoint-selected manifest in memory.
    Verifying that object before any run-directory write avoids rereading a
    stale final manifest and prevents a newer package dictionary from silently
    replacing the authority recorded by the interrupted run.  Very old
    manifests without dictionary coordinates cannot authorize recovery while
    an additional SIC overlay is installed.
    """

    expected_concept_sha, expected_sofa2_sha, expected_overlay_sha = _manifest_expected_shas(manifest)
    if not expected_concept_sha and not expected_sofa2_sha:
        warnings = ["manifest has no concept dictionary fingerprint to verify"]
        # Legacy manifests without any resource coordinates cannot authorize
        # a run against an installed overlay of additional clinical concepts.
        if mode == "strict" and (require_recorded or compute_concept_dict_fingerprint().sic_case_overlay_sha):
            raise ConceptDictDriftError(warnings[0])
        return warnings if mode == "soft" else []
    return assert_dict_matches(
        compute_concept_dict_fingerprint(),
        expected_concept_dict_sha=expected_concept_sha,
        expected_sofa2_dict_sha=expected_sofa2_sha,
        expected_sic_case_overlay_sha=expected_overlay_sha,
        mode=mode,
    )


__all__ = [
    "CONCEPT_DICT_PACKAGE_PATH",
    "SOFA2_DICT_PACKAGE_PATH",
    "ConceptDictDriftError",
    "ConceptDictFingerprint",
    "assert_dict_matches",
    "compute_concept_dict_fingerprint",
    "verify_recorded_dict_match",
    "verify_replay_dict_match",
    "write_concept_dict_fingerprint",
]
