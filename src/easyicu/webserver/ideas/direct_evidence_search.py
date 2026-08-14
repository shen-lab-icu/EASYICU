"""Typed direct-observational evidence search contract for Web Idea Mining.

Owner: ``easyicu.webserver.ideas.direct_evidence_search``.  This module is a
pure compiler/screening projection: it derives one bounded PubMed query and
one source-backed comparator decision from typed study slots.  It performs no
network request, persistence, planning, or novelty inference.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

from easyicu.research_agent.literature import (
    CitationRecord,
    screen_source_backed_direct_comparator,
)
from easyicu.research_agent.literature_concepts import (
    concept_id,
    literature_concept_identity,
    literature_concept_phrase as _owned_literature_concept_phrase,
)

DIRECT_COMPARATOR_FALLBACK_STRATUM = "typed_direct_observational_comparator"

_HOSPITAL_MORTALITY_ALTERNATIVES = (
    ("hospital mortality",),
    ("in-hospital mortality",),
    ("hospital death",),
    ("mortality",),
)


def scope_complete(scope: Dict[str, Any]) -> bool:
    """Return whether typed P/E/O and analysis design can compile a query."""

    return bool(
        _axis_query_alternatives(scope, "exposure")
        and _axis_query_alternatives(scope, "outcome")
        and _clean(scope.get("analysis_family"), 80)
    )


def focus_terms(scope: Dict[str, Any]) -> Tuple[str, ...]:
    """Exact typed terms whose source sentences must survive excerpting."""

    exposure_terms = _flatten_alternatives(
        _axis_query_alternatives(scope, "exposure")
    )
    outcome_terms = _flatten_alternatives(_axis_query_alternatives(scope, "outcome"))
    database = database_phrase(scope.get("database"))
    population = "adult" if population_kind(scope) == "adult" else ""
    return tuple(
        value
        for value in dict.fromkeys(
            (
                *exposure_terms,
                *outcome_terms,
                database,
                population,
                "ICU",
                "intensive care",
                "cohort",
                "observational",
                "prevalence",
                "incidence",
                "epidemiology",
            )
        )
        if value
    )


def screen_article(article: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    """Project a source-backed article through the shared comparator screen."""

    pmid = _clean(article.get("pmid") or "unknown", 32)
    record = CitationRecord(
        key=f"web_pubmed_{pmid}",
        title=_clean(article.get("title") or "Untitled PubMed record", 500),
        year=str(article.get("year") or "n/a"),
        relevance=(
            "Study-design excerpt: "
            + _clean(
                article.get("design_excerpt")
                or article.get("abstract_excerpt")
                or article.get("evidence_sentence")
                or "",
                1_200,
            )
        ),
        pmid=pmid if pmid.isdigit() else None,
        publication_types=[
            _clean(value, 120)
            for value in list(article.get("publication_types") or [])[:20]
            if _clean(value, 120)
        ],
    )
    exposure_identity = literature_concept_identity(scope.get("exposure_concept"))
    decision = screen_source_backed_direct_comparator(
        exposure=(
            exposure_identity.screening_role_term
            if exposure_identity is not None
            else _axis_phrase(scope, "exposure")
        ),
        outcome=_axis_phrase(scope, "outcome"),
        adult_required=population_kind(scope) == "adult",
        record=record,
        source="web_pubmed_retrieval",
        query=None,
        exposure_required_terms=(
            exposure_identity.screening_required_terms
            if exposure_identity is not None
            else ()
        ),
    )
    return decision.model_dump(mode="json")


def build_query(scope: Dict[str, Any]) -> str:
    """Compile one bounded typed retrieval query.

    Conventional concept aliases improve recall only.  Every returned article
    still passes :func:`screen_article`, which uses the exact declared axes and
    source-backed text before granting direct-comparator status.
    """

    p_e_o = build_scope_clause(scope)
    if not p_e_o:
        return ""
    database = database_phrase(scope.get("database"))
    database_clause = (
        f'"{database.replace(chr(34), "")}"[Title/Abstract] OR '
        if database
        else ""
    )
    analysis_family = _clean(scope.get("analysis_family"), 80).casefold()
    design_intent = (
        "(prevalence[Title/Abstract] OR incidence[Title/Abstract] OR "
        "epidemiolog*[Title/Abstract])"
        if analysis_family == "descriptive_epidemiology"
        else "(association[Title/Abstract] OR risk[Title/Abstract] OR "
        "validation[Title/Abstract] OR prevalence[Title/Abstract])"
    )
    return (
        f"({p_e_o}) AND {design_intent} AND "
        '(ICU[Title/Abstract] OR "critical care"[Title/Abstract] OR '
        '"intensive care"[Title/Abstract])'
        + population_filter(scope)
        + " AND ("
        + database_clause
        + "cohort[Title/Abstract] OR observational[Title/Abstract] OR "
        "retrospective[Title/Abstract] OR prospective[Title/Abstract] OR "
        "database[Title/Abstract])"
        " NOT (Review[Publication Type] OR Meta-Analysis[Publication Type] OR "
        "Guideline[Publication Type] OR Practice Guideline[Publication Type] OR "
        "Randomized Controlled Trial[Publication Type] OR Clinical Trial[Publication Type])"
    )


def build_scope_clause(scope: Dict[str, Any]) -> str:
    """Compile the typed exposure/outcome portion of a PubMed query.

    The returned syntax is an immutable retrieval projection.  It may include
    conventional names for one owner-issued concept, but it never changes the
    scientific exposure/outcome stored in the StudyContext.
    """

    exposure = _axis_query_alternatives(scope, "exposure")
    outcome = _axis_query_alternatives(scope, "outcome")
    if not exposure or not outcome:
        return ""
    return " AND ".join(
        (_pubmed_alternative_clause(exposure), _pubmed_alternative_clause(outcome))
    )


def build_concept_anchor_query(scope: Dict[str, Any]) -> str:
    """Compile one all-year definition/development query for the exposure.

    A study plan needs both direct P/E/O comparators and the source literature
    that defines or validates its main construct.  This stratum intentionally
    omits the outcome, so its records can provide related clinical background;
    it cannot satisfy the direct-comparator screen by query membership.
    """

    exposure = _axis_query_alternatives(scope, "exposure")
    if not exposure:
        return ""
    # The first alternative is the owner-issued canonical literature name.
    # Keeping this anchor narrow prevents a broader legacy synonym from
    # displacing the actual development/validation paper in a bounded result
    # set; the remaining P/E/O strata still use every retrieval alternative.
    return (
        f"({_pubmed_alternative_clause(exposure[:1])}) AND "
        "(definition[Title/Abstract] OR criteria[Title/Abstract] OR "
        "development[Title/Abstract] OR validation[Title/Abstract] OR "
        "consensus[Title/Abstract]) AND "
        '(ICU[Title/Abstract] OR "critical care"[Title/Abstract] OR '
        '"intensive care"[Title/Abstract])'
        + population_filter(scope)
    )


def population_filter(scope: Dict[str, Any]) -> str:
    kind = population_kind(scope)
    if kind == "adult":
        return (
            " AND (adult[Title/Abstract] OR adults[Title/Abstract])"
            " NOT (child[Title/Abstract] OR children[Title/Abstract] OR "
            "pediatric[Title/Abstract] OR paediatric[Title/Abstract] OR "
            "neonat*[Title/Abstract])"
        )
    if kind == "pediatric":
        return (
            " AND (child[Title/Abstract] OR children[Title/Abstract] OR "
            "pediatric[Title/Abstract] OR paediatric[Title/Abstract])"
            " NOT (adult[Title/Abstract] OR adults[Title/Abstract])"
        )
    return ""


def population_kind(scope: Dict[str, Any]) -> str:
    population = " ".join(
        str(scope.get(key) or "")
        for key in ("topic", "population", "cohort", "population_scope")
    ).casefold()
    if any(token in population for token in ("adult", "adults", "成人")):
        return "adult"
    if any(
        token in population
        for token in ("pediatric", "paediatric", "children", "child", "儿科", "儿童")
    ):
        return "pediatric"
    return ""


def concept_phrase(value: Any) -> str:
    return _owned_literature_concept_phrase(value)


def _axis_query_alternatives(
    scope: Dict[str, Any], axis: str
) -> tuple[tuple[str, ...], ...]:
    concept = concept_id(scope.get(f"{axis}_concept"))
    display = _clean(scope.get(axis), 180)
    if axis == "outcome" and concept in {"death", "mortality"}:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", display.casefold())
        if any(token in normalized for token in ("in hospital", "hospital", "院内")):
            return _HOSPITAL_MORTALITY_ALTERNATIVES
    identity = literature_concept_identity(concept)
    if identity is not None:
        return identity.retrieval_alternatives
    phrase = concept_phrase(concept or display)
    return ((phrase,),) if phrase else ()


def _flatten_alternatives(
    alternatives: Iterable[Iterable[str]],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _clean(term, 180)
            for alternative in alternatives
            for term in alternative
            if _clean(term, 180)
        )
    )


def _pubmed_alternative_clause(
    alternatives: tuple[tuple[str, ...], ...],
) -> str:
    clauses = []
    for alternative in alternatives:
        terms = [
            f'"{_clean(term, 180).replace(chr(34), "")}"[Title/Abstract]'
            for term in alternative
            if _clean(term, 180)
        ]
        if not terms:
            continue
        clauses.append(terms[0] if len(terms) == 1 else f"({' AND '.join(terms)})")
    if not clauses:
        return ""
    return clauses[0] if len(clauses) == 1 else f"({' OR '.join(clauses)})"


def _axis_phrase(scope: Dict[str, Any], axis: str) -> str:
    concept = scope.get(f"{axis}_concept")
    display = _clean(scope.get(axis), 180)
    phrase = concept_phrase(concept or display)
    # The typed execution concept ``death`` deliberately stays database-neutral,
    # while the conversational slot may bind the actual endpoint window. Keep
    # that typed specificity in the retrieval query instead of widening every
    # death endpoint to generic mortality.
    if axis == "outcome" and concept_phrase(concept) == "mortality":
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", display.casefold())
        if any(token in normalized for token in ("in hospital", "hospital", "院内")):
            return "hospital mortality"
        if any(token in normalized for token in ("intensive care", "icu", "重症")):
            return "ICU mortality"
    return phrase


def database_phrase(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    aliases = {
        "miiv": "MIMIC-IV",
        "mimiciv": "MIMIC-IV",
        "mimic4": "MIMIC-IV",
        "eicu": "eICU",
        "eicucrd": "eICU",
        "hirid": "HiRID",
        "aumc": "Amsterdam UMCdb",
        "amsterdamumcdb": "Amsterdam UMCdb",
    }
    return aliases.get(normalized, "")


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


__all__ = [
    "DIRECT_COMPARATOR_FALLBACK_STRATUM",
    "build_concept_anchor_query",
    "build_query",
    "build_scope_clause",
    "focus_terms",
    "scope_complete",
    "screen_article",
]
