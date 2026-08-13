"""Typed direct-observational evidence search contract for Web Idea Mining.

Owner: ``easyicu.webserver.ideas.direct_evidence_search``.  This module is a
pure compiler/screening projection: it derives one bounded PubMed query and
one source-backed comparator decision from typed study slots.  It performs no
network request, persistence, planning, or novelty inference.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from easyicu.research_agent.literature import (
    CitationRecord,
    screen_source_backed_direct_comparator,
)

DIRECT_COMPARATOR_FALLBACK_STRATUM = "typed_direct_observational_comparator"

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


def scope_complete(scope: Dict[str, Any]) -> bool:
    """Return whether typed P/E/O and analysis design can compile a query."""

    return bool(
        _axis_phrase(scope, "exposure")
        and _axis_phrase(scope, "outcome")
        and _clean(scope.get("analysis_family"), 80)
    )


def focus_terms(scope: Dict[str, Any]) -> Tuple[str, ...]:
    """Exact typed terms whose source sentences must survive excerpting."""

    exposure = _axis_phrase(scope, "exposure")
    outcome = _axis_phrase(scope, "outcome")
    database = database_phrase(scope.get("database"))
    population = "adult" if population_kind(scope) == "adult" else ""
    return tuple(
        value
        for value in dict.fromkeys(
            (
                exposure,
                outcome,
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
    decision = screen_source_backed_direct_comparator(
        exposure=_axis_phrase(scope, "exposure"),
        outcome=_axis_phrase(scope, "outcome"),
        adult_required=population_kind(scope) == "adult",
        record=record,
        source="web_pubmed_retrieval",
        query=None,
    )
    return decision.model_dump(mode="json")


def build_query(scope: Dict[str, Any]) -> str:
    """Compile one bounded typed query without widening exposure or outcome."""

    exposure = _axis_phrase(scope, "exposure")
    outcome = _axis_phrase(scope, "outcome")
    if not exposure or not outcome:
        return ""
    p_e_o = " AND ".join(
        f'"{value.replace(chr(34), "")}"[Title/Abstract]'
        for value in (exposure, outcome)
    )
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
    raw = _clean(value, 180)
    if not raw:
        return ""
    concept_id = raw
    for suffix in _MATERIALIZED_SUFFIXES:
        if concept_id.endswith(suffix):
            concept_id = concept_id[: -len(suffix)]
            break
    labels = {
        "sep3_sofa1": "Sepsis-3",
        "sep3_sofa2": "SOFA-2 sepsis",
        "death": "mortality",
        "mortality": "mortality",
        "hospital_mortality": "hospital mortality",
        "lact": "lactate",
        "lactate": "lactate",
    }
    return labels.get(concept_id.casefold(), raw.replace("_", " "))


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
    "build_query",
    "focus_terms",
    "scope_complete",
    "screen_article",
]
