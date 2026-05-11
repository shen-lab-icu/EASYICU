"""Regression tests for ``EvidenceStore`` corrupted-index handling.

Previously ``_load_records`` and ``_load_aliases`` silently swallowed any
exception (including malformed JSON) and returned empty containers. That
hid evidence loss from operators. These tests pin the new behaviour:

* a corrupt ``evidence_index.json`` is moved aside with a ``.broken-*``
  suffix and a warning is emitted;
* the same applies to ``evidence_aliases.json`` (including the case where
  the JSON parses but is not a mapping);
* fresh registrations after a quarantine recreate clean files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


def _seed_corrupt(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _quarantined_files(directory: Path, stem: str) -> list[Path]:
    return sorted(directory.glob(f"{stem}.*.broken-*")) + sorted(
        directory.glob(f"{stem}.broken-*")
    )


def test_corrupt_index_is_quarantined_with_warning(ra, tmp_path: Path, caplog):
    # Create the directory layout the store expects, then pre-seed a broken
    # index file.
    evidence_dir = tmp_path / "evidence"
    bad_index = _seed_corrupt(evidence_dir / "evidence_index.json", b"{not json")

    with caplog.at_level(logging.WARNING, logger="easyicu.research_agent.evidence"):
        store = ra.EvidenceStore(root=tmp_path)

    assert store.records() == []
    assert not bad_index.exists(), "corrupt index should be moved aside"
    backups = _quarantined_files(evidence_dir, "evidence_index.json")
    assert backups, "expected a .broken-* backup of the corrupt index"
    assert any("corrupt" in rec.message for rec in caplog.records)


def test_corrupt_aliases_is_quarantined(ra, tmp_path: Path, caplog):
    evidence_dir = tmp_path / "evidence"
    _seed_corrupt(evidence_dir / "evidence_aliases.json", b"not-json-at-all")

    with caplog.at_level(logging.WARNING, logger="easyicu.research_agent.evidence"):
        store = ra.EvidenceStore(root=tmp_path)

    assert store.aliases() == {}
    backups = _quarantined_files(evidence_dir, "evidence_aliases.json")
    assert backups, "expected a .broken-* backup of the corrupt aliases"


def test_aliases_wrong_type_is_quarantined(ra, tmp_path: Path, caplog):
    # JSON parses but the top-level value is a list, not a dict — the old
    # code silently dropped this; the new code quarantines and warns.
    evidence_dir = tmp_path / "evidence"
    _seed_corrupt(evidence_dir / "evidence_aliases.json", b"[\"not\", \"a\", \"map\"]")

    with caplog.at_level(logging.WARNING, logger="easyicu.research_agent.evidence"):
        store = ra.EvidenceStore(root=tmp_path)

    assert store.aliases() == {}
    backups = _quarantined_files(evidence_dir, "evidence_aliases.json")
    assert backups
    assert any(
        "not a JSON object" in rec.message for rec in caplog.records
    ), "operator must see a clear warning explaining what was wrong"


def test_after_quarantine_store_can_register_fresh_evidence(ra, tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    _seed_corrupt(evidence_dir / "evidence_index.json", b"{")

    store = ra.EvidenceStore(root=tmp_path)
    src = tmp_path / "table_one.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    rec = store.register_file(kind="table", description="t1", source_path=src)

    # Fresh write succeeds and the store is now resolvable again.
    assert store.get(rec.evidence_id) is not None
    assert store.get("table_one") is not None
    # The freshly written index parses cleanly (i.e. the corrupt file did
    # not poison subsequent state).
    import json
    parsed = json.loads((evidence_dir / "evidence_index.json").read_text())
    assert any(p["evidence_id"] == rec.evidence_id for p in parsed)
