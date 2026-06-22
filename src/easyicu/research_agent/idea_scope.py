"""Literature scope spec → PubMed query builder for idea-mining (discovery lever 1).

Why this exists
---------------
The S1–S6 discovery prototype is single-user and, by default, everyone would
mine the *same* corpus ("recent reviews/editorials from the top critical-care
journals") against the *same* database — so concurrent users get near-identical
candidate lists. Letting a user **scope** the corpus (journal set, date window,
publication types, topic) is both:

1. a better product (users mine their own area), and
2. the cheapest lever against cross-user duplication — different scopes produce
   different candidate spaces.

This module only turns a declarative ``LiteratureScopeSpec`` into a PubMed query
string. That string is fed to ``literature.PubMedLiteratureClient.search(...)``.
It does no network I/O and is fully deterministic (the only time-dependent input,
``last_n_years``, takes an explicit ``reference_year``).

Case-neutrality (G1)
--------------------
Journal presets and publication-type mappings are *publication metadata*, not
clinical cases — they hardcode no disease/exposure/score/database. ``topic_terms``
are always user/scope supplied; nothing clinical is baked in here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# PubMed Title Abbreviations (the form accepted by the [Journal] tag).
# Curated, approximate "high impact" critical-care set; impact-factor rankings
# shift year to year, so treat these as presets a user can edit, not a claim
# about the exact current top-N.
JOURNAL_PRESETS: Dict[str, List[str]] = {
    "critical_care_high_impact": [
        "Lancet Respir Med",
        "Intensive Care Med",
        "Am J Respir Crit Care Med",
        "Crit Care",
        "Crit Care Med",
    ],
    "critical_care_specialty_broad": [
        "Lancet Respir Med",
        "Intensive Care Med",
        "Am J Respir Crit Care Med",
        "Crit Care",
        "Crit Care Med",
        "Ann Intensive Care",
        "Chest",
        "J Crit Care",
        "Shock",
        "Crit Care Explor",
        "J Intensive Care",
        "Intensive Care Med Exp",
        "Respir Care",
        "Resuscitation",
        "Br J Anaesth",
        "Anesthesiology",
        "Anaesthesia",
        "Thorax",
        "Ann Am Thorac Soc",
    ],
    "critical_care_specialty_wide": [
        "Lancet Respir Med",
        "Intensive Care Med",
        "Am J Respir Crit Care Med",
        "Crit Care",
        "Crit Care Med",
        "Ann Intensive Care",
        "Chest",
        "J Crit Care",
        "Shock",
        "Crit Care Explor",
        "J Intensive Care",
        "Intensive Care Med Exp",
        "Respir Care",
        "Resuscitation",
        "Br J Anaesth",
        "Anesthesiology",
        "Anaesthesia",
        "Thorax",
        "Ann Am Thorac Soc",
        "Curr Opin Crit Care",
        "Crit Care Clin",
        "Semin Respir Crit Care Med",
        "Neurocrit Care",
        "Pediatr Crit Care Med",
        "J Trauma Acute Care Surg",
        "Anaesth Crit Care Pain Med",
        "Acute Crit Care",
        "Indian J Crit Care Med",
        "Intensive Crit Care Nurs",
        "J Cardiothorac Vasc Anesth",
        "Burns",
        "Clin Chest Med",
    ],
    "critical_care_top3": [
        "Lancet Respir Med",
        "Intensive Care Med",
        "Crit Care",
    ],
    "general_medicine_high_impact": [
        "N Engl J Med",
        "Lancet",
        "JAMA",
        "BMJ",
    ],
}

# Map friendly publication-type names to PubMed [pt] / [sb] clauses.
_PUB_TYPE_CLAUSES: Dict[str, str] = {
    "review": "review[pt]",
    "editorial": "editorial[pt]",
    "guideline": "guideline[pt]",
    "practice_guideline": "practice guideline[pt]",
    "systematic_review": "systematic review[pt]",
    "meta_analysis": "meta-analysis[pt]",
    "letter": "letter[pt]",
    "comment": "comment[pt]",
}


class LiteratureScopeSpec(BaseModel):
    """Declarative scope for the idea-mining literature corpus.

    At least one of ``journals`` / ``journal_preset`` / ``topic_terms`` /
    ``extra_terms`` must be set, otherwise the query would match all of PubMed.
    """

    model_config = ConfigDict(extra="forbid")

    journals: List[str] = Field(
        default_factory=list,
        description="Explicit PubMed Title Abbreviations, e.g. 'Intensive Care Med'.",
    )
    journal_preset: Optional[str] = Field(
        default=None,
        description=f"One of {sorted(JOURNAL_PRESETS)}; merged with `journals`.",
    )
    pub_types: List[str] = Field(
        default_factory=lambda: ["review", "editorial"],
        description="Friendly publication-type names; mapped to PubMed [pt] clauses.",
    )
    start_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    end_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    last_n_years: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Relative window; resolved against an explicit reference_year.",
    )
    topic_terms: List[str] = Field(
        default_factory=list,
        description="User-supplied topic terms; OR-ed together, AND-ed into the query.",
    )
    extra_terms: Optional[str] = Field(
        default=None,
        description="Raw PubMed fragment appended verbatim (advanced escape hatch).",
    )

    @model_validator(mode="after")
    def _validate_scope(self) -> "LiteratureScopeSpec":
        if self.journal_preset is not None and self.journal_preset not in JOURNAL_PRESETS:
            raise ValueError(
                f"unknown journal_preset {self.journal_preset!r}; "
                f"known: {sorted(JOURNAL_PRESETS)}"
            )
        if self.start_year is not None and self.end_year is not None:
            if self.start_year > self.end_year:
                raise ValueError("start_year cannot be after end_year")
        if self.last_n_years is not None and (
            self.start_year is not None or self.end_year is not None
        ):
            raise ValueError(
                "use either last_n_years or explicit start_year/end_year, not both"
            )
        if not (
            self.journals
            or self.journal_preset
            or self.topic_terms
            or self.extra_terms
        ):
            raise ValueError(
                "scope is too broad: set at least one of journals / journal_preset "
                "/ topic_terms / extra_terms"
            )
        return self


def resolve_journals(scope: LiteratureScopeSpec) -> List[str]:
    """Merge explicit journals with the preset, preserving order and de-duping."""
    merged: List[str] = []
    seen = set()
    preset = JOURNAL_PRESETS.get(scope.journal_preset or "", [])
    for name in [*scope.journals, *preset]:
        key = str(name or "").strip()
        if key and key.lower() not in seen:
            seen.add(key.lower())
            merged.append(key)
    return merged


def resolve_year_range(
    scope: LiteratureScopeSpec,
    *,
    reference_year: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    """Return (start_year, end_year) or None when no date window was requested.

    ``last_n_years`` is resolved against ``reference_year``; when that is omitted
    the current UTC year is used (the only non-deterministic path — pass
    ``reference_year`` for reproducible/frozen queries).
    """
    if scope.start_year is not None or scope.end_year is not None:
        start = scope.start_year or scope.end_year
        end = scope.end_year or scope.start_year
        return int(start), int(end)
    if scope.last_n_years is not None:
        ref = reference_year or datetime.now(timezone.utc).year
        start = ref - scope.last_n_years + 1
        return int(start), int(ref)
    return None


def _phrase(term: str) -> str:
    return f'"{term}"' if (" " in term or "-" in term) else term


def _journal_clause(journals: List[str]) -> str:
    parts = [f'"{name}"[Journal]' for name in journals]
    return "(" + " OR ".join(parts) + ")" if parts else ""


def _pub_type_clause(pub_types: List[str]) -> str:
    parts: List[str] = []
    seen = set()
    for name in pub_types:
        key = str(name or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        parts.append(_PUB_TYPE_CLAUSES.get(key, f"{key}[pt]"))
    return "(" + " OR ".join(parts) + ")" if parts else ""


def _topic_clause(topic_terms: List[str]) -> str:
    parts = [_phrase(str(t).strip()) for t in topic_terms if str(t or "").strip()]
    return "(" + " OR ".join(parts) + ")" if parts else ""


def build_pubmed_query_from_scope(
    scope: LiteratureScopeSpec,
    *,
    reference_year: Optional[int] = None,
) -> str:
    """Build a PubMed query string from a scope spec.

    Clauses are AND-ed: journals AND publication-types AND date AND topic AND extra.
    Empty clauses are omitted. The output is deterministic given ``reference_year``.
    """
    clauses: List[str] = []

    journal_clause = _journal_clause(resolve_journals(scope))
    if journal_clause:
        clauses.append(journal_clause)

    pub_clause = _pub_type_clause(scope.pub_types)
    if pub_clause:
        clauses.append(pub_clause)

    year_range = resolve_year_range(scope, reference_year=reference_year)
    if year_range is not None:
        start, end = year_range
        clauses.append(f"{start}:{end}[dp]")

    topic_clause = _topic_clause(scope.topic_terms)
    if topic_clause:
        clauses.append(topic_clause)

    if scope.extra_terms and scope.extra_terms.strip():
        clauses.append(f"({scope.extra_terms.strip()})")

    return " AND ".join(clauses)


__all__ = [
    "JOURNAL_PRESETS",
    "LiteratureScopeSpec",
    "build_pubmed_query_from_scope",
    "resolve_journals",
    "resolve_year_range",
]
