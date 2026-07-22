"""PubMed / prior-art query-construction helpers for idea mining.

Pure text + query-clause utilities extracted from ``idea_mining.py`` (P1
split, 2026-06-10). Zero behavior change: every name here is re-exported
from ``idea_mining`` for backward compatibility. This module is a leaf — it
must not import from ``idea_mining`` (a module-boundary test enforces that),
so it depends only on the concept catalog/availability helpers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..concept_availability import normalize_concept_name
from ..concept_catalog import SYNONYM_GROUPS

_GENERIC_DIFFERENTIATOR_PATTERNS = (
    "icu",
    "icu stay",
    "early icu stay",
    "adult",
    "adult patient",
    "adult patients",
    "critically ill",
    "critical illness",
    "patients",
    "patient",
    "outcome",
    "mortality",
    "association",
    "any exposure",
    "exposure",
    "trajectory summary",
)


_GENERIC_CONCEPT_WORDS = {
    "association",
    "available",
    "average",
    "binary",
    # Specimen/carrier word, not a standalone physiologic construct.  Keeping
    # it as a concept signal allowed an unrelated phrase such as "pulsatile
    # blood flow" to bind to "pH of blood" when the catalog was restricted.
    "blood",
    "candidate",
    "change",
    "changes",
    "clinical",
    "computed",
    "duration",
    "durations",
    "early",
    "event",
    "events",
    "exposure",
    "feature",
    "features",
    "first",
    "hour",
    "hours",
    "icu",
    "indicator",
    "intensity",
    "level",
    "levels",
    "measurement",
    "measurements",
    "measure",
    "observed",
    "patient",
    "patients",
    "raw",
    "score",
    "status",
    "study",
    "studies",
    "supporting",
    "total",
    "value",
    "values",
    "window",
    "within",
}


_PRIOR_ART_QUERY_STOPWORDS = _GENERIC_CONCEPT_WORDS | {
    "abstract",
    "admission",
    "adult",
    "adults",
    "after",
    "analysis",
    "and",
    "before",
    "care",
    "centered",
    "critical",
    "critically",
    "database",
    "during",
    "endpoint",
    "endpoints",
    "hospital",
    "ill",
    "illness",
    "intensive",
    "of",
    "or",
    "therapy",
    "the",
    "unit",
    "with",
}


_PRIOR_ART_SINGLETON_STOPWORDS = _PRIOR_ART_QUERY_STOPWORDS | {
    # Single-token facets from these words make PubMed relevance explode
    # without preserving same-topic specificity. Keep the full phrase and
    # multiword facets instead.
    "balance",
    "clearance",
    "count",
    "dose",
    "driving",
    "energy",
    "exposure",
    "failure",
    "free",
    "index",
    "injury",
    "mechanical",
    "modified",
    "pattern",
    "power",
    "pressure",
    "profile",
    "ratio",
    "red",
    "setting",
    "shock",
    "signature",
    "strategy",
    "timing",
    "trajectory",
}


_PRIOR_ART_QUERY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "mortality": ("death",),
    "death": ("mortality",),
}


def _top_values(values: Sequence[str], *, limit: int = 5) -> List[str]:
    counts: Dict[str, int] = {}
    for value in values:
        text = str(value or "").strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return sorted(counts, key=lambda item: (-counts[item], item))[:limit]


def _clean_literature_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _is_specific_differentiator(value: str) -> bool:
    text = _clean_literature_phrase(value).lower()
    if not text:
        return False
    normalised = normalize_concept_name(text)
    generic = {
        normalize_concept_name(item) for item in _GENERIC_DIFFERENTIATOR_PATTERNS
    }
    if normalised in generic:
        return False
    return not any(pattern in text for pattern in _GENERIC_DIFFERENTIATOR_PATTERNS)


def _pubmed_phrase_clause(phrase: str) -> str:
    text = _clean_literature_phrase(phrase)
    if not text:
        return ""
    escaped = text.replace('"', "")
    if " " in escaped or "-" in escaped:
        return f'"{escaped}"[Title/Abstract]'
    return f"{escaped}[Title/Abstract]"


def _pubmed_or_clause(clauses: Sequence[str]) -> str:
    items = _ordered_unique([clause for clause in clauses if clause])
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "(" + " OR ".join(items) + ")"


def _pubmed_recall_clause(phrase: str) -> str:
    """Return a recall-oriented Title/Abstract clause from literature words."""

    text = _clean_literature_phrase(phrase)
    if not text:
        return ""
    clauses = [_pubmed_phrase_clause(text), _pubmed_mesh_clause(text)]
    for facet in _prior_art_phrase_facets(text):
        clauses.append(_pubmed_phrase_clause(facet))
    return _pubmed_or_clause(clauses)


def _pubmed_core_recall_clause(
    core_phrase: Optional[str],
    *,
    fallback_phrase: str,
) -> str:
    """Return a broad PubMed clause from core concept facets plus source words."""

    core = _clean_literature_phrase(str(core_phrase or ""))
    fallback = _clean_literature_phrase(fallback_phrase)
    phrases: List[str] = []
    if core:
        phrases.append(core)
        phrases.extend(_prior_art_synonym_phrases(core))
        clauses = [_pubmed_recall_clause(phrase) for phrase in phrases]
        if fallback and normalize_concept_name(fallback) != normalize_concept_name(
            core
        ):
            clauses.append(_pubmed_phrase_clause(fallback))
        return _pubmed_or_clause(clauses)
    if fallback:
        phrases.append(fallback)
    return _pubmed_or_clause([_pubmed_recall_clause(phrase) for phrase in phrases])


def _pubmed_mesh_clause(phrase: str) -> str:
    text = _clean_literature_phrase(phrase)
    if not text:
        return ""
    escaped = text.replace('"', "")
    if " " in escaped or "-" in escaped:
        return f'"{escaped}"[MeSH Terms]'
    return f"{escaped}[MeSH Terms]"


def _pubmed_population_recall_clause(population: str) -> str:
    """Extract only durable population facets instead of an over-tight phrase."""

    text = _clean_literature_phrase(population).lower()
    if not text:
        return ""
    clauses: List[str] = []
    if "adult" in text:
        clauses.append(_pubmed_phrase_clause("adult"))
    if (
        "icu" in text
        or "intensive care" in text
        or "critical illness" in text
        or "critically ill" in text
    ):
        clauses.extend(
            [
                _pubmed_phrase_clause("ICU"),
                _pubmed_phrase_clause("intensive care"),
                _pubmed_phrase_clause("critically ill"),
                _pubmed_phrase_clause("critical illness"),
            ]
        )
    if (
        "mechanically ventilated" in text
        or "mechanical ventilation" in text
        or "ventilated" in text
    ):
        clauses.extend(
            [
                _pubmed_phrase_clause("mechanically ventilated"),
                _pubmed_phrase_clause("mechanical ventilation"),
            ]
        )
    if "septic shock" in text:
        clauses.append(_pubmed_phrase_clause("septic shock"))
    elif "sepsis" in text:
        clauses.append(_pubmed_phrase_clause("sepsis"))
    elif "shock" in text:
        clauses.append(_pubmed_phrase_clause("shock"))
    return _pubmed_or_clause(clauses)


def _prior_art_phrase_facets(phrase: str) -> List[str]:
    """Derive phrase-local recall facets without using EasyICU concept keys."""

    tokens = _prior_art_query_tokens(phrase)
    facets: List[str] = _prior_art_synonym_phrases(phrase)
    for size in (3, 2):
        for idx in range(0, len(tokens) - size + 1):
            facets.append(" ".join(tokens[idx : idx + size]))
    facets.extend(
        token for token in tokens if token not in _PRIOR_ART_SINGLETON_STOPWORDS
    )
    for token in list(tokens):
        facets.extend(_PRIOR_ART_QUERY_SYNONYMS.get(token, ()))
    original = _clean_literature_phrase(phrase).lower()
    return _ordered_unique([item for item in facets if item != original])[:6]


def _prior_art_synonym_phrases(phrase: str) -> List[str]:
    target = _clean_literature_phrase(phrase).lower()
    if not target:
        return []
    target_key = normalize_concept_name(target)
    out: List[str] = []
    for group in SYNONYM_GROUPS:
        group_keys = {normalize_concept_name(item) for item in group}
        if target_key in group_keys:
            out.extend(sorted(group))
    return _ordered_unique(
        [item for item in out if normalize_concept_name(item) != target_key]
    )


def _prior_art_query_tokens(phrase: str) -> List[str]:
    text = _clean_literature_phrase(phrase).lower().replace("-", " ")
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [
        token
        for token in raw_tokens
        if len(token) >= 3 and token not in _PRIOR_ART_QUERY_STOPWORDS
    ]
    return _ordered_unique(tokens)


def _ordered_unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
