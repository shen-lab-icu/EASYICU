"""Owner policy for concepts that require an explicit user selection.

Availability and scientific suitability are different contracts.  A concept
may be physically present while still being an experimental, deprecated, or
sensitivity-only alternative that must not replace the ordinary meaning of a
user's question.  This module owns that small dependency-free boundary so Web,
Idea Mining, and Research Agent launchers do not each infer it independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class ConceptSelectionPolicy:
    concept_id: str
    selection_mode: str
    rationale: str
    explicit_terms: tuple[str, ...]
    canonical_alternative: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "selection_mode": self.selection_mode,
            "selection_note": self.rationale,
            "canonical_alternative": self.canonical_alternative,
        }


@dataclass(frozen=True, slots=True)
class ConceptSelectionDecision:
    concept_id: str
    allowed: bool
    reason_code: str
    selection_mode: str
    canonical_alternative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "selection_mode": self.selection_mode,
            "canonical_alternative": self.canonical_alternative,
        }


_POLICIES = {
    "sep3_sofa2": ConceptSelectionPolicy(
        concept_id="sep3_sofa2",
        selection_mode="explicit_only",
        rationale=(
            "Experimental SOFA-2 sensitivity phenotype; not the canonical "
            "2016 Sepsis-3 definition. The user must explicitly request the "
            "SOFA-2 variant before it can be selected."
        ),
        explicit_terms=(
            "sep3_sofa2",
            "sofa-2",
            "sofa 2",
            "sofa2",
            "sepsis-3 sofa-2",
            "sepsis 3 sofa 2",
            "sofa-2 based sepsis",
            "sofa 2 based sepsis",
            "sofa-2 sepsis",
            "sofa 2 sepsis",
            "experimental sepsis sensitivity",
            "基于sofa-2的脓毒症",
            "基于 sofa-2 的脓毒症",
            "sofa-2脓毒症",
            "sofa-2 脓毒症",
            "实验性脓毒症敏感性",
        ),
        canonical_alternative="sep3_sofa1",
    ),
}

_NEGATED_EXPLICIT_SELECTION = re.compile(
    r"(?:do\s+not|don't|not\s+using|without|exclude|不要|不使用|排除|并非)"
    r".{0,40}(?:sofa\s*[- ]?2|sep3_sofa2)",
    flags=re.IGNORECASE,
)


def concept_selection_policy(concept_id: Any) -> Optional[ConceptSelectionPolicy]:
    """Return the owner-issued selection policy for one canonical concept id."""

    return _POLICIES.get(str(concept_id or "").strip())


def evaluate_concept_selection(
    concept_id: Any,
    *,
    user_intent: Any,
) -> ConceptSelectionDecision:
    """Decide whether ``user_intent`` authorizes this concept selection.

    Ordinary concepts remain allowed.  An ``explicit_only`` concept requires a
    positive literal reference to its owner-issued aliases in the user's
    scientific question; a generic disease label is intentionally
    insufficient. Callers must not append model-generated plan/configuration
    prose to ``user_intent`` because that would let the model self-authorize an
    experimental variant.
    """

    normalized_id = str(concept_id or "").strip()
    policy = concept_selection_policy(normalized_id)
    if policy is None:
        return ConceptSelectionDecision(
            concept_id=normalized_id,
            allowed=True,
            reason_code="concept_selection_ordinary",
            selection_mode="ordinary",
        )
    text = " ".join(str(user_intent or "").casefold().split())
    explicitly_named = any(term.casefold() in text for term in policy.explicit_terms)
    negated = bool(_NEGATED_EXPLICIT_SELECTION.search(text))
    allowed = explicitly_named and not negated
    return ConceptSelectionDecision(
        concept_id=normalized_id,
        allowed=allowed,
        reason_code=(
            "concept_selection_explicit"
            if allowed
            else "concept_explicit_selection_required"
        ),
        selection_mode=policy.selection_mode,
        canonical_alternative=policy.canonical_alternative,
    )


__all__ = [
    "ConceptSelectionDecision",
    "ConceptSelectionPolicy",
    "concept_selection_policy",
    "evaluate_concept_selection",
]
