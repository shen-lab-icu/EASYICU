from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.discovery.idea_registry import (
    CandidateAlreadyRegisteredError,
    CandidateNotExecutableError,
    CandidateNotRegisteredError,
    CandidateRegistryEntry,
    IdeaCandidateRegistry,
    IdeaRegistryError,
)


def _entry(
    candidate_id: str,
    *,
    family: str = "family_a",
    source_snapshot_id: str = "manual-note/snapshot-alpha",
) -> CandidateRegistryEntry:
    return CandidateRegistryEntry(
        hypothesis_family_id=family,
        candidate_id=candidate_id,
        source_snapshot_id=source_snapshot_id,
    )


def test_registry_entry_requires_nonempty_ids() -> None:
    with pytest.raises(ValidationError):
        CandidateRegistryEntry(
            hypothesis_family_id="",
            candidate_id="candidate_a",
            source_snapshot_id="snapshot_a",
        )

    with pytest.raises(ValidationError):
        CandidateRegistryEntry(
            hypothesis_family_id="family_a",
            candidate_id="candidate_a",
            source_snapshot_id="",
        )


def test_selection_event_requires_human_reason() -> None:
    with pytest.raises(ValidationError):
        CandidateRegistryEntry(
            hypothesis_family_id="family_a",
            candidate_id="candidate_a",
            source_snapshot_id="snapshot_a",
            selection_status="accepted",
            selected_by="",
            selection_reason="clinically coherent and feasible",
        )

    with pytest.raises(ValidationError):
        CandidateRegistryEntry(
            hypothesis_family_id="family_a",
            candidate_id="candidate_a",
            source_snapshot_id="snapshot_a",
            selection_status="rejected",
            selected_by="reviewer",
            selection_reason="",
        )


def test_register_candidate_writes_append_only_ledger(tmp_path: Path) -> None:
    registry = IdeaCandidateRegistry(tmp_path / "idea_registry.json")
    entry = registry.register_candidate(_entry("candidate_a"))

    payload = json.loads((tmp_path / "idea_registry.json").read_text())
    assert payload["schema_version"] == "easyicu.idea_candidate_registry/1"
    assert payload["entries"] == [entry.model_dump(mode="json")]
    assert payload["entries"][0]["selection_status"] == "proposed"
    assert payload["entries"][0]["source_snapshot_id"] == "manual-note/snapshot-alpha"


def test_existing_empty_registry_file_initializes_empty_ledger(tmp_path: Path) -> None:
    path = tmp_path / "idea_registry.json"
    path.touch()

    registry = IdeaCandidateRegistry(path)

    assert registry.records == ()
    payload = json.loads(path.read_text())
    assert payload == {
        "schema_version": "easyicu.idea_candidate_registry/1",
        "entries": [],
    }


def test_existing_whitespace_registry_file_initializes_empty_ledger(
    tmp_path: Path,
) -> None:
    path = tmp_path / "idea_registry.json"
    path.write_text("\n\t  \n", encoding="utf-8")

    registry = IdeaCandidateRegistry(path)

    assert registry.records == ()
    payload = json.loads(path.read_text())
    assert payload["entries"] == []


def test_duplicate_candidate_registration_is_rejected(tmp_path: Path) -> None:
    registry = IdeaCandidateRegistry(tmp_path / "idea_registry.json")
    registry.register_candidate(_entry("candidate_a"))

    with pytest.raises(CandidateAlreadyRegisteredError):
        registry.register_candidate(_entry("candidate_a"))

    payload = json.loads((tmp_path / "idea_registry.json").read_text())
    assert len(payload["entries"]) == 1


def test_record_selection_appends_event_and_replays_latest_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "idea_registry.json"
    registry = IdeaCandidateRegistry(path)
    registry.register_candidate(_entry("candidate_a"))
    first_payload = json.loads(path.read_text())

    accepted = registry.record_selection(
        "candidate_a",
        "accepted",
        by="human-reviewer",
        reason="passes feasibility and clinical relevance gate",
    )

    payload = json.loads(path.read_text())
    assert payload["entries"][0] == first_payload["entries"][0]
    assert payload["entries"][1] == accepted.model_dump(mode="json")
    assert payload["entries"][1]["selection_status"] == "accepted"
    assert payload["entries"][1]["selected_by"] == "human-reviewer"

    reloaded = IdeaCandidateRegistry(path)
    assert reloaded.latest_entry("candidate_a").selection_status == "accepted"
    assert reloaded.assert_executable("candidate_a") is True


def test_strict_gate_rejects_unregistered_proposed_and_rejected_candidates(
    tmp_path: Path,
) -> None:
    registry = IdeaCandidateRegistry(tmp_path / "idea_registry.json")

    with pytest.raises(CandidateNotRegisteredError):
        registry.assert_executable("candidate_missing")

    registry.register_candidate(_entry("candidate_a"))
    with pytest.raises(CandidateNotExecutableError):
        registry.assert_executable("candidate_a")

    registry.record_selection(
        "candidate_a",
        "rejected",
        by="human-reviewer",
        reason="not executable after feasibility review",
    )
    with pytest.raises(CandidateNotExecutableError):
        registry.assert_executable("candidate_a")


def test_record_selection_requires_registered_candidate(tmp_path: Path) -> None:
    registry = IdeaCandidateRegistry(tmp_path / "idea_registry.json")

    with pytest.raises(CandidateNotRegisteredError):
        registry.record_selection(
            "candidate_missing",
            "accepted",
            by="human-reviewer",
            reason="attempted selection without preregistration",
        )


def test_family_size_counts_preregistered_candidates_not_selection_events(
    tmp_path: Path,
) -> None:
    registry = IdeaCandidateRegistry(tmp_path / "idea_registry.json")
    registry.register_candidate(_entry("candidate_a", family="family_a"))
    registry.register_candidate(_entry("candidate_b", family="family_a"))
    registry.register_candidate(_entry("candidate_c", family="family_b"))
    registry.record_selection(
        "candidate_a",
        "accepted",
        by="human-reviewer",
        reason="selected for dry run",
    )
    registry.record_selection(
        "candidate_b",
        "rejected",
        by="human-reviewer",
        reason="lower feasibility",
    )

    assert registry.family_size("family_a") == 2
    assert registry.family_size("family_b") == 1
    assert registry.family_size("family_missing") == 0


def test_source_snapshot_id_is_required_but_opaque(tmp_path: Path) -> None:
    registry = IdeaCandidateRegistry(tmp_path / "idea_registry.json")
    registry.register_candidate(
        _entry(
            "candidate_a",
            source_snapshot_id="pubmed-query-20260604/manual-excerpt-7",
        )
    )

    payload = json.loads((tmp_path / "idea_registry.json").read_text())
    assert payload["entries"][0]["source_snapshot_id"] == (
        "pubmed-query-20260604/manual-excerpt-7"
    )
    assert not payload["entries"][0]["source_snapshot_id"].startswith("sha256:")


def test_registry_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "idea_registry.json"
    path.write_text(
        json.dumps({"schema_version": "easyicu.idea_candidate_registry/0", "entries": []}),
        encoding="utf-8",
    )

    with pytest.raises(IdeaRegistryError):
        IdeaCandidateRegistry(path)


def test_package_lazy_exports_registry_api() -> None:
    import easyicu.research_agent as ra

    assert ra.CandidateRegistryEntry is CandidateRegistryEntry
    assert ra.IdeaCandidateRegistry is IdeaCandidateRegistry
    assert ra.SelectionStatus is not None


def test_concurrent_instances_do_not_lose_appends(tmp_path: Path) -> None:
    # Two registry handles on the same ledger. `a` loaded its snapshot before
    # `b` appended; without reload-before-append, a's write would clobber b's
    # entry (lost update). The locked read-modify-write must preserve both.
    path = tmp_path / "registry.json"
    a = IdeaCandidateRegistry(path)
    b = IdeaCandidateRegistry(path)

    b.register_candidate(_entry("candidate_b"))
    a.register_candidate(_entry("candidate_a"))  # a still holds a stale snapshot

    on_disk = IdeaCandidateRegistry(path)
    ids = {rec.candidate_id for rec in on_disk.records}
    assert ids == {"candidate_a", "candidate_b"}


def test_gate_sees_decision_written_by_another_instance(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    a = IdeaCandidateRegistry(path)
    b = IdeaCandidateRegistry(path)

    a.register_candidate(_entry("candidate_a"))
    a.record_selection("candidate_a", "accepted", by="dr_x", reason="plausible")

    # b never saw the registration in memory, but the gate reloads from disk.
    assert b.assert_executable("candidate_a") is True


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = IdeaCandidateRegistry(path)
    registry.register_candidate(_entry("candidate_a"))

    # ledger is valid JSON and no .tmp sidecars are left behind
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [e["candidate_id"] for e in payload["entries"]] == ["candidate_a"]
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
