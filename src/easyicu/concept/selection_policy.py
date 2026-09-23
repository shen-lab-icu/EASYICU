"""Owner policy for concepts whose availability does not imply suitability.

Availability and scientific suitability are different contracts.  A concept
may be physically present while still being an experimental, deprecated, or
sensitivity-only alternative that must not replace the ordinary meaning of a
user's question.  This module owns that small dependency-free boundary so Web,
Idea Mining, and Research Agent launchers do not each infer it independently.

Two modes exist.  ``explicit_only`` withholds a variant until the user names
it.  ``prefer_alternative`` never withholds anything -- the concept is a real,
citable definition -- but carries the owner's caution and the canonical
alternative to whoever is choosing a role for it, so the caution arrives while
the design is being written rather than as a refusal afterwards.
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


#: A variant is withheld until the user names it.
EXPLICIT_ONLY = "explicit_only"
#: A definition is selectable, but a better-suited one exists for most roles.
PREFER_ALTERNATIVE = "prefer_alternative"

#: KDIGO stage bindings whose zero category absorbs unobserved evidence.  Every
#: one of them follows the upstream rule that a component with no usable
#: evidence contributes zero, so ``stage 0`` means "no positive evidence was
#: found", not "kidney injury was ruled out".  They remain the right columns
#: for reproducing the published cross-database phenotype; they are the wrong
#: ones for an exposure or endpoint, because the reference group then contains
#: every never-assessed stay.  The evidence receipts are deliberately absent
#: from this list: they are the remedy, not the defect.
OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS: tuple[str, ...] = (
    "aki",
    "aki_reference",
    "aki_severe",
    "aki_severe_reference",
    "aki_severe_source_native",
    "aki_source_native",
    "aki_stage",
    "aki_stage_creat_reference",
    "aki_stage_creat_source_native",
    "aki_stage_crrt_source_native",
    "aki_stage_reference",
    "aki_stage_reference_smoothed_6h",
    "aki_stage_rrt_reference",
    "aki_stage_source_native",
    "aki_stage_source_native_smoothed",
    "aki_stage_uo_reference",
    "aki_stage_uo_source_native",
    "kdigo_aki",
    "kdigo_stage",
)

_KDIGO_OBSERVABILITY_NOTE = (
    "stage 0 here absorbs never-assessed stays (a component with no usable "
    "evidence counts as zero); as an exposure or endpoint use aki_stage_strict, "
    "which keeps unassessed stays unknown"
)


_POLICIES = {
    concept: ConceptSelectionPolicy(
        concept_id=concept,
        selection_mode=PREFER_ALTERNATIVE,
        rationale=_KDIGO_OBSERVABILITY_NOTE,
        explicit_terms=(),
        canonical_alternative="aki_stage_strict",
    )
    for concept in OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS
}
_POLICIES.update({
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
})

_MODULE_CONCEPT_IDS = {
    "sepsis3_sofa2": "sep3_sofa2",
}

_NEGATED_EXPLICIT_SELECTION = re.compile(
    r"(?:do\s+not|don't|not\s+using|without|exclude|不要|不使用|排除|并非)"
    r".{0,40}(?:sofa\s*[- ]?2|sep3_sofa2)",
    flags=re.IGNORECASE,
)


def concept_selection_policy(concept_id: Any) -> Optional[ConceptSelectionPolicy]:
    """Return the owner-issued selection policy for one canonical concept id."""

    return _POLICIES.get(str(concept_id or "").strip())


def concept_id_for_module(module: Any) -> Optional[str]:
    """Return the governed concept selected by one feature module, if any."""

    return _MODULE_CONCEPT_IDS.get(str(module or "").strip().casefold())


def _explicit_term_present(text: str, term: str) -> bool:
    """Match an explicit alias without accepting numeric continuations.

    ``sofa 2`` is an authorization phrase; the ``2`` in ``SOFA 28-day`` or
    ``SOFA 2.5`` is not.  ASCII identifier boundaries also prevent aliases
    such as ``sofa2`` from matching longer tokens while retaining the Chinese
    forms whose adjacent characters are meaningful parts of the issued alias.
    """

    escaped = re.escape(term.casefold()).replace(r"\ ", r"\s+")
    return bool(
        re.search(
            rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_]|\s*[.\uff0e]\s*\d)",
            text,
        )
    )


def evaluate_concept_selection(
    concept_id: Any,
    *,
    user_intent: Any,
    owner_confirmed: bool = False,
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
    if policy.selection_mode != EXPLICIT_ONLY:
        # An advisory policy carries a caution, never a refusal: the concept is
        # a real definition and withholding it would remove the published
        # phenotype from every legitimate reproduction.
        return ConceptSelectionDecision(
            concept_id=normalized_id,
            allowed=True,
            reason_code="concept_selection_advisory",
            selection_mode=policy.selection_mode,
            canonical_alternative=policy.canonical_alternative,
        )
    text = " ".join(str(user_intent or "").casefold().split())
    explicitly_named = any(
        _explicit_term_present(text, term) for term in policy.explicit_terms
    )
    negated = bool(_NEGATED_EXPLICIT_SELECTION.search(text))
    allowed = (explicitly_named and not negated) or bool(owner_confirmed)
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


def concept_selection_confirmation_key(concept_id: Any) -> str:
    """Return the typed owner-confirmation key for an explicit concept."""

    normalized_id = re.sub(r"[^a-z0-9_]+", "_", str(concept_id or "").casefold())
    return f"concept_selection_{normalized_id}_authorized"


def concept_selection_authority_key(concept_id: Any) -> str:
    """Return the server-owned receipt key for a verified user selection."""

    normalized_id = re.sub(r"[^a-z0-9_]+", "_", str(concept_id or "").casefold())
    return f"concept_selection_{normalized_id}_user_turn_verified"


def is_concept_selection_authority_key(value: Any) -> bool:
    """Return whether one confirmation key is reserved for the host owner."""

    return bool(
        re.fullmatch(
            r"concept_selection_[a-z0-9_]+_user_turn_verified",
            str(value or ""),
        )
    )


__all__ = [
    "EXPLICIT_ONLY",
    "OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS",
    "PREFER_ALTERNATIVE",
    "ConceptSelectionDecision",
    "ConceptSelectionPolicy",
    "concept_id_for_module",
    "concept_selection_authority_key",
    "concept_selection_policy",
    "concept_selection_confirmation_key",
    "evaluate_concept_selection",
    "is_concept_selection_authority_key",
]
