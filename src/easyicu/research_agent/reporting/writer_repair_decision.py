"""One bounded repair decision for one STRICT-rejected manuscript sentence.

Why this is a type and not a dict: the decision crossed the producer/applier
seam as ``List[Dict[str, object]]``. ``writer_evidence_repair`` validated every
invariant on the way out, ``manuscript_post`` re-derived the same invariants
from scratch on the way in, and ``write_phase`` hand-built the literal in two
further places without going through either. Four constructions, two
validators, and nothing in the type system tying them together — so the only
way to know whether a decision was legal was to read all four.

The invariants below are the ones that hold regardless of which manuscript the
decision is applied to:

* the sentence index is a real, non-negative position;
* the action is exactly one of cite, claim, or drop;
* ``cite`` names 1-3 evidence ids and no claim;
* ``claim`` names exactly one host claim ref and no evidence ids — the model
  selects an existing claim, it never authors one;
* ``drop`` names neither.

They are checked once, in ``__post_init__``, so a ``WriterRepairDecision`` that
exists is a decision of a legal shape. What stays with the applier is the part
that genuinely depends on the manuscript: whether the named ids are registered
for *this* run, and whether the target sentence is still in *this* draft. That
is a different question and it belongs where the draft is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence

WriterRepairAction = Literal["cite", "claim", "drop"]

#: A cite decision names at least one and at most this many evidence ids.
MAX_CITE_EVIDENCE_IDS = 3

__all__ = [
    "MAX_CITE_EVIDENCE_IDS",
    "WriterRepairAction",
    "WriterRepairDecision",
    "coerce_writer_repair_decisions",
    "drop_every_sentence",
]


def _clean_ids(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError("writer evidence_ids must be a sequence")
    return tuple(
        dict.fromkeys(
            str(value).strip() for value in values if str(value).strip()
        )
    )


@dataclass(frozen=True)
class WriterRepairDecision:
    """What the host will do to one rejected sentence, in a legal shape."""

    index: int
    action: WriterRepairAction
    evidence_ids: tuple[str, ...] = ()
    claim_ref: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool):
            raise ValueError("writer evidence decision index must be an integer")
        if self.index < 0:
            raise ValueError("writer evidence decision index must not be negative")
        if self.action not in ("cite", "claim", "drop"):
            raise ValueError("writer evidence action must be cite, claim, or drop")
        if self.action == "cite":
            if not 1 <= len(self.evidence_ids) <= MAX_CITE_EVIDENCE_IDS:
                raise ValueError(
                    f"cite decisions require 1-{MAX_CITE_EVIDENCE_IDS} registered "
                    "evidence ids"
                )
            if self.claim_ref:
                raise ValueError("cite decisions cannot select a claim_ref")
        elif self.action == "claim":
            if self.evidence_ids:
                raise ValueError("claim decisions cannot include evidence ids")
            if not self.claim_ref:
                raise ValueError(
                    "claim decisions require one registered host claim_ref"
                )
        else:
            if self.evidence_ids:
                raise ValueError("drop decisions cannot include evidence ids")
            if self.claim_ref:
                raise ValueError("drop decisions cannot select a claim_ref")

    # -- constructors ---------------------------------------------------

    @classmethod
    def cite(cls, index: int, evidence_ids: Sequence[str]) -> "WriterRepairDecision":
        return cls(index=index, action="cite", evidence_ids=_clean_ids(evidence_ids))

    @classmethod
    def claim(cls, index: int, claim_ref: str) -> "WriterRepairDecision":
        return cls(index=index, action="claim", claim_ref=str(claim_ref).strip())

    @classmethod
    def drop(cls, index: int) -> "WriterRepairDecision":
        return cls(index=index, action="drop")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WriterRepairDecision":
        """Build from an untrusted mapping — model output, or a stored receipt."""
        if not isinstance(payload, Mapping):
            raise ValueError("each writer evidence decision must be an object")
        return cls(
            index=payload.get("index"),  # type: ignore[arg-type]
            action=str(payload.get("action") or "").strip().lower(),  # type: ignore[arg-type]
            evidence_ids=_clean_ids(payload.get("evidence_ids", ())),
            claim_ref=str(payload.get("claim_ref") or "").strip(),
        )

    # -- projection -----------------------------------------------------

    def as_dict(self) -> Dict[str, object]:
        """The receipt shape. ``claim_ref`` appears only where it means something."""
        record: Dict[str, object] = {
            "index": self.index,
            "action": self.action,
            "evidence_ids": list(self.evidence_ids),
        }
        if self.claim_ref:
            record["claim_ref"] = self.claim_ref
        return record


def coerce_writer_repair_decisions(
    decisions: Sequence[Any],
) -> List[WriterRepairDecision]:
    """Accept decisions in either shape and return validated values.

    Mappings still arrive from stored receipts and from callers that predate
    the type. They go through the same constructor, so there is still exactly
    one place a decision is checked.
    """
    return [
        decision
        if isinstance(decision, WriterRepairDecision)
        else WriterRepairDecision.from_mapping(decision)
        for decision in decisions
    ]


def drop_every_sentence(count: int) -> List[WriterRepairDecision]:
    """The conservative host fallback: drop each rejected sentence, in order."""
    return [WriterRepairDecision.drop(index) for index in range(int(count))]
