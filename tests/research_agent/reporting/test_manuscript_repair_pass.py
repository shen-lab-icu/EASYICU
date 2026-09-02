from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementError
from easyicu.research_agent.reporting.manuscript_post import (
    _apply_writer_evidence_repair_decisions,
    _writer_repair_target_span,
)
from easyicu.research_agent.reporting.manuscript_repair_pass import (
    ManuscriptRepairPass,
    ManuscriptRepairResult,
)
from easyicu.research_agent.reporting.writer_repair_decision import (
    WriterRepairDecision,
)


_SENTENCE = "This unsupported interpretation must not survive."
_SCAFFOLD = f"## Results\n\n{_SENTENCE}\n"


def _repair_pass(decision_provider) -> ManuscriptRepairPass:
    return ManuscriptRepairPass(
        decision_provider=decision_provider,
        decision_applier=_apply_writer_evidence_repair_decisions,
        target_locator=_writer_repair_target_span,
    )


def _run(repair_pass: ManuscriptRepairPass, *, enforce_scaffold=lambda _text: None):
    return repair_pass.run(
        _SCAFFOLD,
        llm=object(),
        evidence_ids=["registered_result"],
        evidence_digest="registered_result: observed counts",
        rejected_sentences=[_SENTENCE],
        scientific_claims={},
        claim_required_sentences=[],
        allowed_claim_refs=[],
        language="en",
        enforce_scaffold=enforce_scaffold,
    )


def test_pass_consumes_writer_repair_decisions_and_returns_one_typed_receipt() -> None:
    seen = []

    def decide(*_args, **_kwargs):
        decision = WriterRepairDecision.drop(0)
        seen.append(decision)
        return [decision]

    result = _run(_repair_pass(decide))

    assert isinstance(result, ManuscriptRepairResult)
    assert all(isinstance(decision, WriterRepairDecision) for decision in seen)
    assert _SENTENCE not in result.scaffold
    assert result.evidence_repairs[0]["action"] == "drop"
    assert result.fallback_detail is None


def test_invalid_decision_fails_closed_to_a_host_owned_drop() -> None:
    result = _run(
        _repair_pass(lambda *_args, **_kwargs: [WriterRepairDecision.drop(index=9)])
    )

    assert _SENTENCE not in result.scaffold
    assert result.evidence_repairs[0]["action"] == "drop"
    assert result.fallback_detail == {
        "reason_code": "writer_evidence_repair_deterministic_drop",
        "exception_type": "ValueError",
        "rejected_sentence_count": 1,
        "structured_attempts": [],
    }


def test_strict_revalidation_drops_a_citation_that_cannot_authorize_a_claim() -> None:
    calls = []

    def enforce(scaffold: str) -> None:
        calls.append(scaffold)
        if len(calls) == 1:
            rejected = scaffold.split("\n\n", 1)[1].strip()
            raise EvidenceEnforcementError(
                "unsupported result",
                detail={"removed_sentences": [rejected]},
            )

    result = _run(
        _repair_pass(
            lambda *_args, **_kwargs: [
                WriterRepairDecision.cite(0, ["registered_result"])
            ]
        ),
        enforce_scaffold=enforce,
    )

    assert len(calls) == 2
    assert _SENTENCE not in result.scaffold
    assert result.evidence_repairs[0]["action"] == "cite"
    assert result.residual_strict_drops[0]["action"] == "drop"
    assert result.residual_drop_detail is not None


def test_provider_transport_failure_retains_its_owner_boundary() -> None:
    class ProviderTransportFailure(RuntimeError):
        pass

    def fail(*_args, **_kwargs):
        raise ProviderTransportFailure("provider unavailable")

    with pytest.raises(ProviderTransportFailure, match="provider unavailable"):
        _run(_repair_pass(fail))
