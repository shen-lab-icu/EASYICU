"""The registered novelty packet must be the bytes that authorize maturity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.reporting.write_phase import (
    _ensure_unsigned_novelty_positioning_packet,
)


class _RecordingEvidence:
    def __init__(self) -> None:
        self._records: list[SimpleNamespace] = []
        self.calls: list[dict] = []

    def current_verified_records(self, per_step_records):
        del per_step_records
        return list(self._records)

    def get(self, evidence_id):
        for record in self._records:
            if record.evidence_id == evidence_id:
                return record
        return None

    def register_file(self, **kwargs):
        source = Path(kwargs["source_path"])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        evidence_id = str(kwargs["evidence_id"])
        existing = {record.evidence_id: record for record in self._records}
        if evidence_id in existing and existing[evidence_id].sha256 != digest:
            if kwargs.get("on_sha_change") != "new_id":
                raise ValueError("evidence id collision")
            index = 2
            while f"{evidence_id}_v{index}" in existing:
                index += 1
            evidence_id = f"{evidence_id}_v{index}"
        record = SimpleNamespace(evidence_id=evidence_id, sha256=digest)
        self._records.append(record)
        self.calls.append({**kwargs, "evidence_id": evidence_id})
        return record


def test_reviewed_packet_bytes_are_registered_not_the_blank_snapshot(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "novelty_positioning_audit.json"
    packet.write_text('{"status": "review_required"}', encoding="utf-8")
    evidence = _RecordingEvidence()

    _ensure_unsigned_novelty_positioning_packet(
        evidence=evidence,
        context=None,
        plan=None,
        literature=None,
        run_dir=tmp_path,
    )

    blank_digest = hashlib.sha256(packet.read_bytes()).hexdigest()
    assert evidence.get("novelty_positioning_audit").sha256 == blank_digest
    assert len(evidence.calls) == 1

    packet.write_text('{"status": "supported"}', encoding="utf-8")
    _ensure_unsigned_novelty_positioning_packet(
        evidence=evidence,
        context=None,
        plan=None,
        literature=None,
        run_dir=tmp_path,
    )

    reviewed_digest = hashlib.sha256(packet.read_bytes()).hexdigest()
    assert evidence.get("novelty_positioning_audit").sha256 == blank_digest
    assert evidence.get("novelty_positioning_audit_v2").sha256 == reviewed_digest
    assert evidence.calls[-1]["on_sha_change"] == "new_id"

    # A repeated write phase over unchanged bytes must not mint a new version.
    _ensure_unsigned_novelty_positioning_packet(
        evidence=evidence,
        context=None,
        plan=None,
        literature=None,
        run_dir=tmp_path,
    )
    assert len(evidence.calls) == 2
