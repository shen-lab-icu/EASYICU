"""A repair decision validates once, at construction.

The decision used to cross the producer/applier seam as
``List[Dict[str, object]]``: ``writer_evidence_repair`` validated every
invariant on the way out, ``manuscript_post`` re-derived the same invariants
from scratch on the way in, and ``write_phase`` hand-built the literal in two
further places without going through either. Nothing tied the four together,
so the only way to know a decision was legal was to read all four.

These tests pin the split: shape is the value type's invariant; whether an id
is registered *for this run* and whether the sentence is still in *this* draft
stay with the producer and the applier, because only they know.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.writer_repair_decision import (
    WriterRepairDecision,
    coerce_writer_repair_decisions,
    drop_every_sentence,
)


def test_cite_carries_its_ids_deduplicated_and_ordered() -> None:
    decision = WriterRepairDecision.cite(0, [" a ", "b", "a"])
    assert decision.evidence_ids == ("a", "b")
    assert decision.as_dict() == {
        "index": 0,
        "action": "cite",
        "evidence_ids": ["a", "b"],
    }


def test_claim_ref_appears_in_the_receipt_only_when_it_means_something() -> None:
    assert "claim_ref" not in WriterRepairDecision.drop(1).as_dict()
    assert WriterRepairDecision.claim(1, "step.claim").as_dict()["claim_ref"] == (
        "step.claim"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"index": -1, "action": "drop"}, "must not be negative"),
        ({"index": True, "action": "drop"}, "must be an integer"),
        ({"index": 0, "action": "rewrite"}, "cite, claim, or drop"),
        ({"index": 0, "action": "cite"}, "require 1-3"),
        (
            {"index": 0, "action": "cite", "evidence_ids": ("a", "b", "c", "d")},
            "require 1-3",
        ),
        (
            {"index": 0, "action": "cite", "evidence_ids": ("a",), "claim_ref": "c"},
            "cite decisions cannot select a claim_ref",
        ),
        ({"index": 0, "action": "claim"}, "require one registered host claim_ref"),
        (
            {"index": 0, "action": "claim", "evidence_ids": ("a",), "claim_ref": "c"},
            "claim decisions cannot include evidence ids",
        ),
        (
            {"index": 0, "action": "drop", "evidence_ids": ("a",)},
            "drop decisions cannot include evidence ids",
        ),
        (
            {"index": 0, "action": "drop", "claim_ref": "c"},
            "drop decisions cannot select a claim_ref",
        ),
    ],
)
def test_an_illegal_shape_cannot_be_constructed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        WriterRepairDecision(**kwargs)


def test_a_decision_is_frozen() -> None:
    decision = WriterRepairDecision.drop(0)
    with pytest.raises(Exception):
        decision.action = "cite"  # type: ignore[misc]


def test_a_mapping_is_validated_by_the_same_constructor() -> None:
    decision = WriterRepairDecision.from_mapping(
        {"index": 2, "action": " CITE ", "evidence_ids": ["x"]}
    )
    assert decision == WriterRepairDecision.cite(2, ["x"])

    with pytest.raises(ValueError, match="drop decisions cannot include evidence ids"):
        WriterRepairDecision.from_mapping(
            {"index": 0, "action": "drop", "evidence_ids": ["x"]}
        )


def test_a_non_object_is_refused() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        WriterRepairDecision.from_mapping("drop")  # type: ignore[arg-type]


def test_evidence_ids_must_be_a_sequence() -> None:
    with pytest.raises(ValueError, match="must be a sequence"):
        WriterRepairDecision.from_mapping(
            {"index": 0, "action": "cite", "evidence_ids": "x"}
        )


def test_coercion_accepts_both_shapes_without_a_second_validator() -> None:
    coerced = coerce_writer_repair_decisions(
        [WriterRepairDecision.drop(0), {"index": 1, "action": "cite", "evidence_ids": ["x"]}]
    )
    assert coerced == [WriterRepairDecision.drop(0), WriterRepairDecision.cite(1, ["x"])]


def test_the_host_fallback_drops_each_sentence_in_order() -> None:
    assert drop_every_sentence(3) == [WriterRepairDecision.drop(i) for i in range(3)]
    assert drop_every_sentence(0) == []
