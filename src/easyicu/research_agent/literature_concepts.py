"""Dependency-neutral literature identities for owner-issued concepts.

One execution concept may have conventional names in article metadata without
changing its scientific meaning.  This owner supplies retrieval alternatives
and the stricter terms that must be present for direct-comparator screening.
It performs no query, network call, screening decision, or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LiteratureConceptIdentity:
    concept_id: str
    canonical_phrase: str
    retrieval_alternatives: tuple[tuple[str, ...], ...]
    screening_role_term: str
    screening_required_terms: tuple[str, ...]


_MATERIALIZED_SUFFIXES = (
    "_first_time",
    "_last_time",
    "_measured",
    "_first",
    "_mean",
    "_max",
    "_min",
    "_n",
)

_IDENTITIES = {
    "sep3_sofa1": LiteratureConceptIdentity(
        concept_id="sep3_sofa1",
        canonical_phrase="Sepsis-3",
        retrieval_alternatives=(("Sepsis-3",),),
        screening_role_term="Sepsis-3",
        screening_required_terms=("Sepsis-3",),
    ),
    "sep3_sofa2": LiteratureConceptIdentity(
        concept_id="sep3_sofa2",
        canonical_phrase="SOFA-2 sepsis",
        retrieval_alternatives=(
            ("SOFA-2",),
            ("Sepsis-3", "SOFA"),
            ("Sepsis 3", "Sequential Organ Failure Assessment"),
        ),
        screening_role_term="SOFA-2",
        screening_required_terms=("Sepsis-3", "SOFA-2"),
    ),
    "death": LiteratureConceptIdentity(
        concept_id="death",
        canonical_phrase="mortality",
        retrieval_alternatives=(("mortality",), ("death",)),
        screening_role_term="mortality",
        screening_required_terms=(),
    ),
    "mortality": LiteratureConceptIdentity(
        concept_id="mortality",
        canonical_phrase="mortality",
        retrieval_alternatives=(("mortality",), ("death",)),
        screening_role_term="mortality",
        screening_required_terms=(),
    ),
    "lact": LiteratureConceptIdentity(
        concept_id="lact",
        canonical_phrase="lactate",
        retrieval_alternatives=(("lactate",), ("lactic acid",)),
        screening_role_term="lactate",
        screening_required_terms=(),
    ),
    "lactate": LiteratureConceptIdentity(
        concept_id="lactate",
        canonical_phrase="lactate",
        retrieval_alternatives=(("lactate",), ("lactic acid",)),
        screening_role_term="lactate",
        screening_required_terms=(),
    ),
    "aki": LiteratureConceptIdentity(
        concept_id="aki",
        canonical_phrase="acute kidney injury",
        retrieval_alternatives=(("acute kidney injury",), ("AKI",)),
        screening_role_term="acute kidney injury",
        screening_required_terms=(),
    ),
    "kdigo_stage": LiteratureConceptIdentity(
        concept_id="kdigo_stage",
        canonical_phrase="KDIGO acute kidney injury",
        retrieval_alternatives=(("KDIGO", "acute kidney injury"), ("AKI",)),
        screening_role_term="acute kidney injury",
        screening_required_terms=(),
    ),
}

_PHRASE_ALIASES = {
    "hospital_mortality": "hospital mortality",
}


def concept_id(value: Any) -> str:
    token = " ".join(str(value or "").split()).casefold()[:180]
    for suffix in _MATERIALIZED_SUFFIXES:
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def literature_concept_identity(value: Any) -> Optional[LiteratureConceptIdentity]:
    return _IDENTITIES.get(concept_id(value))


def literature_concept_phrase(value: Any, *, fallback: Any = None) -> str:
    """Return one stable literature phrase without case-specific callers.

    The mapping is concept metadata, not benchmark routing.  Unknown concepts
    use an owner-supplied display label when available and otherwise retain a
    compact, human-readable form of the original token.
    """

    raw = " ".join(str(value or "").split())[:180]
    if not raw:
        return ""
    identity = literature_concept_identity(raw)
    if identity is not None:
        return identity.canonical_phrase
    phrase_alias = _PHRASE_ALIASES.get(concept_id(raw))
    if phrase_alias:
        return phrase_alias
    fallback_text = " ".join(str(fallback or "").split())[:180]
    return fallback_text or raw.replace("_", " ")


__all__ = [
    "LiteratureConceptIdentity",
    "concept_id",
    "literature_concept_identity",
    "literature_concept_phrase",
]
