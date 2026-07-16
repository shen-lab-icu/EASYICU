"""Fail-closed regression tests for corrupted legacy evidence ledgers."""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.evidence_authority import (
    EvidenceAuthorityIntegrityError,
)


def _seed(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize(
    ("filename", "payload", "match"),
    [
        ("evidence_index.json", b"{not json", "index"),
        ("evidence_index.json", b"{}", "records"),
        ("evidence_aliases.json", b"not-json-at-all", "aliases"),
        ("evidence_aliases.json", b'["not", "a", "map"]', "aliases"),
        ("numeric_claims.json", b"{broken", "numeric claims"),
        ("numeric_claims.json", b"{}", "numeric claims"),
    ],
)
def test_corrupt_legacy_member_remains_fail_closed_across_reopen(
    ra,
    tmp_path: Path,
    filename: str,
    payload: bytes,
    match: str,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _seed(evidence_dir / "evidence_index.json", b"[]")
    _seed(evidence_dir / "evidence_aliases.json", b"{}")
    if filename == "numeric_claims.json":
        _seed(evidence_dir / "numeric_claims.json", b"[]")
    _seed(evidence_dir / filename, payload)

    for _ in range(2):
        with pytest.raises(EvidenceAuthorityIntegrityError, match=match):
            ra.EvidenceStore(root=tmp_path)

    assert (evidence_dir / filename).read_bytes() == payload


def test_corrupt_legacy_store_cannot_register_fresh_evidence(
    ra,
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _seed(evidence_dir / "evidence_index.json", b"{")
    _seed(evidence_dir / "evidence_aliases.json", b"{}")

    with pytest.raises(EvidenceAuthorityIntegrityError):
        ra.EvidenceStore(root=tmp_path)

    assert not (evidence_dir / "evidence_authority.json").exists()
