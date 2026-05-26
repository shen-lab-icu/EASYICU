"""Evidence store ``on_sha_change`` resume-safety contract.

When a pipeline resumes from a checkpoint, certain pipeline-managed
artefacts (``reproducibility_envelope.json``, ``cost_summary.md``,
``cost_records.json``) legitimately have a different sha256 on the
second invocation than the first — the resume adds new per-call
records, new timestamps, possibly a different model version.

Before this fix the strict collision check raised ``ValueError`` and
crashed the entire resumed run. This test pins three properties:

1. ``on_sha_change="raise"`` (default) preserves the strict legacy
   behavior — a fresh `register_file` call with the same id and
   different content still raises.
2. ``on_sha_change="new_id"`` keeps both records, with the new one
   registered under a ``_v2`` suffix and the original alias still
   resolving to the canonical first record.
3. ``on_sha_change="keep_existing"`` returns the original record
   untouched.

Together they let pipeline.py keep its strict default while allowing
the small set of "this-invocation metadata" artefacts to survive
resume without crashing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.evidence import EvidenceStore


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _register(store: EvidenceStore, src: Path, *, evidence_id: str, on_sha_change: str = "raise"):
    return store.register_file(
        kind="log",
        description="test envelope",
        source_path=src,
        evidence_id=evidence_id,
        producer="test",
        generation_mode="system",
        on_sha_change=on_sha_change,
    )


def test_default_raise_on_sha_collision_preserves_strict_behaviour(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    src1 = _write_file(tmp_path / "src" / "envelope.json", '{"call":1}')
    rec1 = _register(store, src1, evidence_id="reproducibility_envelope")
    assert rec1.evidence_id == "reproducibility_envelope"

    src2 = _write_file(tmp_path / "src" / "envelope.json", '{"call":2}')
    with pytest.raises(ValueError, match="Evidence id collision"):
        _register(store, src2, evidence_id="reproducibility_envelope")


def test_new_id_on_sha_collision_appends_v2_and_keeps_alias(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    src1 = _write_file(tmp_path / "src" / "envelope.json", '{"call":1}')
    rec1 = _register(store, src1, evidence_id="reproducibility_envelope")
    sha1 = rec1.sha256

    # Second call simulates the resume — different content, same id
    src2 = _write_file(tmp_path / "src" / "envelope.json", '{"call":1,"call":2}')
    rec2 = _register(store, src2, evidence_id="reproducibility_envelope", on_sha_change="new_id")
    assert rec2.evidence_id == "reproducibility_envelope_v2"
    assert rec2.sha256 != sha1
    assert (rec2.metadata or {}).get("resume_supersedes") == "reproducibility_envelope"

    # The canonical alias still resolves to the original record
    via_alias = store.get("reproducibility_envelope")
    assert via_alias is not None
    assert via_alias.evidence_id == "reproducibility_envelope"
    assert via_alias.sha256 == sha1

    # Both records persisted
    ids = {r.evidence_id for r in store.records()}
    assert "reproducibility_envelope" in ids
    assert "reproducibility_envelope_v2" in ids


def test_new_id_increments_on_repeated_resume(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    src1 = _write_file(tmp_path / "src" / "envelope.json", "v1")
    _register(store, src1, evidence_id="reproducibility_envelope")

    src2 = _write_file(tmp_path / "src" / "envelope.json", "v2")
    rec2 = _register(store, src2, evidence_id="reproducibility_envelope", on_sha_change="new_id")
    assert rec2.evidence_id == "reproducibility_envelope_v2"

    src3 = _write_file(tmp_path / "src" / "envelope.json", "v3")
    rec3 = _register(store, src3, evidence_id="reproducibility_envelope", on_sha_change="new_id")
    assert rec3.evidence_id == "reproducibility_envelope_v3"


def test_keep_existing_returns_original_on_collision(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    src1 = _write_file(tmp_path / "src" / "envelope.json", "first")
    rec1 = _register(store, src1, evidence_id="reproducibility_envelope")

    src2 = _write_file(tmp_path / "src" / "envelope.json", "second")
    rec2 = _register(
        store, src2, evidence_id="reproducibility_envelope", on_sha_change="keep_existing"
    )
    assert rec2.evidence_id == rec1.evidence_id
    assert rec2.sha256 == rec1.sha256


def test_unknown_on_sha_change_mode_raises(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    src1 = _write_file(tmp_path / "src" / "envelope.json", "x")
    _register(store, src1, evidence_id="x")

    src2 = _write_file(tmp_path / "src" / "envelope.json", "y")
    with pytest.raises(ValueError, match="Unknown on_sha_change mode"):
        _register(store, src2, evidence_id="x", on_sha_change="overwrite")
