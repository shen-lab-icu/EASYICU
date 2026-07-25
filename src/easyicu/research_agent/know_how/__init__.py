"""Curated, source-backed research know-how for planner retrieval.

The know-how library is deliberately separate from ``ExperienceBank``:
experience records are run-derived procedural hints, while these cards are
versioned methodological candidates backed by external sources.  Retrieval is
offline and deterministic; cards never acquire cohort, method, or estimand
authority merely by being selected.
"""

from .registry import (
    KnowHowCard,
    KnowHowClaim,
    KnowHowCitation,
    KnowHowDesignCandidates,
    KnowHowHit,
    KnowHowIntegrityError,
    KnowHowReviewAttestation,
    KnowHowRegistry,
    reviewable_card_content_sha256,
)

__all__ = [
    "KnowHowCard",
    "KnowHowClaim",
    "KnowHowCitation",
    "KnowHowDesignCandidates",
    "KnowHowHit",
    "KnowHowIntegrityError",
    "KnowHowReviewAttestation",
    "KnowHowRegistry",
    "reviewable_card_content_sha256",
]
