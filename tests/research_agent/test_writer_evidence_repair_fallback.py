from __future__ import annotations

import pytest

from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
)
from easyicu.research_agent.reporting import write_phase


_REJECTED = "This unsupported interpretation must not survive."
_SCAFFOLD = f"## Results\n\n{_REJECTED}\n"


def _repair() -> tuple[str, list[dict[str, object]], dict[str, object] | None]:
    return write_phase._repair_rejected_writer_sentences(
        _SCAFFOLD,
        llm=object(),
        evidence_ids=["registered_result"],
        evidence_digest="registered_result: observed counts",
        rejected_sentences=[_REJECTED],
        scientific_claims={},
        claim_required_sentences=[],
        allowed_claim_refs=[],
        language="en",
    )


def test_invalid_structured_repair_falls_back_to_deterministic_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = StructuredResponseFailure([], role="writer_evidence_citation_repair")
    monkeypatch.setattr(
        write_phase,
        "decide_writer_evidence_repairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    repaired, applied, fallback = _repair()

    assert _REJECTED not in repaired
    assert applied[0]["action"] == "drop"
    assert fallback == {
        "reason_code": "writer_evidence_repair_deterministic_drop",
        "exception_type": "StructuredResponseFailure",
        "rejected_sentence_count": 1,
        "structured_attempts": [],
    }


def test_invalid_repair_application_falls_back_to_deterministic_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        write_phase,
        "decide_writer_evidence_repairs",
        lambda *args, **kwargs: [{"index": 9, "action": "drop", "evidence_ids": []}],
    )

    repaired, applied, fallback = _repair()

    assert _REJECTED not in repaired
    assert applied[0]["action"] == "drop"
    assert fallback is not None
    assert fallback["exception_type"] == "ValueError"


def test_provider_failure_is_not_hidden_by_writer_drop_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProviderTransportFailure(RuntimeError):
        pass

    monkeypatch.setattr(
        write_phase,
        "decide_writer_evidence_repairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ProviderTransportFailure("provider transport failed")
        ),
    )

    with pytest.raises(ProviderTransportFailure):
        _repair()
