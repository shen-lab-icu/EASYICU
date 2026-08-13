from __future__ import annotations

from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import (
    registered_source_fingerprints_match as compatibility_match,
)
from easyicu.research_agent.authority.source_fingerprints import (
    registered_source_fingerprints_match,
)


class _Lookup:
    def __init__(self, **digests: str) -> None:
        self._records = {
            evidence_id: SimpleNamespace(sha256=digest)
            for evidence_id, digest in digests.items()
        }

    def get(self, evidence_id: str) -> object | None:
        return self._records.get(evidence_id)


def test_exact_registered_source_fingerprints_match() -> None:
    lookup = _Lookup(source_a="a" * 64, source_b="b" * 64)

    assert registered_source_fingerprints_match(
        lookup,
        {
            "source_evidence_ids": ["source_a"],
            "source_evidence_id": "source_b",
            "source_evidence_sha256": {
                "source_a": "a" * 64,
                "source_b": "b" * 64,
            },
        },
    )


def test_absent_unknown_or_stale_source_coordinates_fail_closed() -> None:
    lookup = _Lookup(source_a="a" * 64)

    assert not registered_source_fingerprints_match(lookup, {})
    assert not registered_source_fingerprints_match(
        lookup,
        {
            "source_evidence_ids": ["unknown"],
            "source_evidence_sha256": {"unknown": "b" * 64},
        },
    )
    assert not registered_source_fingerprints_match(
        lookup,
        {
            "source_evidence_ids": "source_a",
            "source_evidence_sha256": {"source_a": "b" * 64},
        },
    )


def test_evidence_store_compatibility_export_is_the_owner_function() -> None:
    assert compatibility_match is registered_source_fingerprints_match
