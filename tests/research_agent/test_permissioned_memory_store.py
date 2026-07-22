from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.learning.store import (
    FileSystemMemoryStore,
    LangGraphMemoryStoreAdapter,
    MemoryAccessPolicy,
    MemoryObject,
    MemoryReviewAttestation,
    payload_sha256,
    promote_quarantined_memory,
    select_memory,
)


def _quarantined() -> MemoryObject:
    return MemoryObject.create(
        namespace="run_lessons/quarantine/project-a",
        key="run-001",
        version="1.0.0",
        payload={"finding_codes": ["missing_input"]},
        source="run-001",
        producer="test",
        review_status="quarantined",
        created_at="2026-07-22T00:00:00Z",
    )


@pytest.mark.parametrize(
    "namespace",
    [
        "run_lessons/quarantine/project-a",
        "preferences/user-a",
        "runtime/session-a",
    ],
)
def test_canonical_policy_rejects_nonreviewed_namespace(namespace: str) -> None:
    with pytest.raises(ValidationError, match="reviewed/promoted memory only"):
        MemoryAccessPolicy(
            canonical=True,
            allowed_namespaces=(namespace,),
        )


def test_payload_tampering_fails_closed() -> None:
    memory = _quarantined().model_dump(mode="json")
    memory["payload"] = {"finding_codes": ["different"]}
    with pytest.raises(ValidationError, match="payload digest mismatch"):
        MemoryObject.model_validate(memory)


def test_filesystem_store_is_write_once_and_revalidates(tmp_path: Path) -> None:
    store = FileSystemMemoryStore(tmp_path)
    memory = _quarantined()
    store.put(memory)
    store.put(memory)
    assert store.get(memory.namespace, memory.key, memory.version) == memory
    assert store.list(memory.namespace) == (memory,)

    path = next(tmp_path.rglob("*.json"))
    path.write_text(path.read_text().replace("missing_input", "bad_input"))
    with pytest.raises(ValidationError, match="payload digest mismatch"):
        store.get(memory.namespace, memory.key, memory.version)


def test_store_rejects_nested_payload_mutation_before_write(tmp_path: Path) -> None:
    store = FileSystemMemoryStore(tmp_path)
    memory = _quarantined()
    memory.payload["finding_codes"].append("mutated")
    with pytest.raises(ValidationError, match="payload digest mismatch"):
        store.put(memory)


def test_promotion_requires_digest_bound_attestation(tmp_path: Path) -> None:
    store = FileSystemMemoryStore(tmp_path)
    source = _quarantined()
    store.put(source)
    bad = MemoryReviewAttestation(
        reviewer="regression-suite",
        reviewed_at="2026-07-22T01:00:00Z",
        review_scope="held-out tests",
        payload_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="does not bind"):
        promote_quarantined_memory(
            store,
            source=source,
            target_version="1.0.0",
            profile_ref="npj_dm/20260722",
            attestation=bad,
        )

    good = bad.model_copy(update={"payload_sha256": source.payload_sha256})
    promoted, receipt = promote_quarantined_memory(
        store,
        source=source,
        target_version="1.0.0",
        profile_ref="npj_dm/20260722",
        attestation=good,
    )
    assert receipt.source_payload_sha256 == source.payload_sha256
    policy = MemoryAccessPolicy(
        canonical=True,
        profile_ref="npj_dm/20260722",
        allowed_namespaces=("promoted_lessons/1.0.0",),
    )
    assert select_memory(store, policy=policy) == (promoted,)
    wrong_profile = policy.model_copy(update={"profile_ref": "npj_dm/other"})
    assert select_memory(store, policy=wrong_profile) == ()


def test_reviewed_memory_cannot_self_attest() -> None:
    payload = {"rule": "use explicit time zero"}
    with pytest.raises(ValidationError, match="requires a review attestation"):
        MemoryObject.create(
            namespace="reviewed_knowledge/npj_dm",
            key="time-zero",
            version="1.0.0",
            payload=payload,
            source="protocol-card",
            producer="model",
            review_status="reviewed",
            profile_ref="npj_dm/20260722",
            created_at="2026-07-22T00:00:00Z",
        )
    assert payload_sha256(payload)


@dataclass
class _Item:
    value: dict


class _FakeLangGraphStore:
    def __init__(self) -> None:
        self.values: dict[tuple[tuple[str, ...], str], dict] = {}

    def put(self, namespace: tuple[str, ...], key: str, value: dict) -> None:
        self.values[(namespace, key)] = value

    def get(self, namespace: tuple[str, ...], key: str) -> _Item | None:
        value = self.values.get((namespace, key))
        return _Item(value) if value is not None else None

    def search(self, namespace: tuple[str, ...]) -> list[_Item]:
        return [
            _Item(value)
            for (candidate, _key), value in self.values.items()
            if candidate == namespace
        ]


def test_langgraph_adapter_round_trip_keeps_easyicu_schema_authority() -> None:
    adapter = LangGraphMemoryStoreAdapter(_FakeLangGraphStore())
    memory = _quarantined()
    adapter.put(memory)
    assert adapter.get(memory.namespace, memory.key, memory.version) == memory
    assert adapter.list(memory.namespace) == (memory,)
