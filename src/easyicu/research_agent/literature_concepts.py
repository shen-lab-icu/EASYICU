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
        retrieval_alternatives=(("Sepsis-3",),),
        screening_role_term="Sepsis-3",
        screening_required_terms=("Sepsis-3",),
    ),
    "sep3_sofa2": LiteratureConceptIdentity(
        concept_id="sep3_sofa2",
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
        retrieval_alternatives=(("mortality",), ("death",)),
        screening_role_term="mortality",
        screening_required_terms=(),
    ),
    "mortality": LiteratureConceptIdentity(
        concept_id="mortality",
        retrieval_alternatives=(("mortality",), ("death",)),
        screening_role_term="mortality",
        screening_required_terms=(),
    ),
}


def concept_id(value: Any) -> str:
    token = " ".join(str(value or "").split()).casefold()[:180]
    for suffix in _MATERIALIZED_SUFFIXES:
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def literature_concept_identity(value: Any) -> Optional[LiteratureConceptIdentity]:
    return _IDENTITIES.get(concept_id(value))


__all__ = [
    "LiteratureConceptIdentity",
    "concept_id",
    "literature_concept_identity",
]
