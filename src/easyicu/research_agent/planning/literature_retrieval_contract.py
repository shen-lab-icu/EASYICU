"""Minimal typed retrieval contract for Track 3 literature hard dependency.

Owner: research-agent planning authority.

This module records, in one typed place, the three facts the plan-finalization
gate needs:

* ``searches`` — what was searched (bibliographic source + exact query).
* ``hits`` — what the search returned (citation key + which query returned it).
* ``supports`` — which retrieved record supports which of the three hard
  decisions (population / outcome-window / method applicability), and with
  which supporting sentence.

Live path reuse: build a contract from an existing
:class:`~easyicu.research_agent.literature.LiteratureBundle` via
:func:`contract_from_bundle`. That projection only reads the PubMed/Tavily
lineage the existing ``LiteratureAgent`` already produced
(``search_provenance.search_queries`` / ``record_queries`` plus screening
sources from ``PubMedLiteratureClient.search_context_strata``). It performs no
network I/O, constructs no provider client, and calls no LLM.

Offline/mock rule: tests and offline runs reuse the same projection over a
bundle built with a stubbed PubMed client (injected into ``LiteratureAgent``,
never a new HTTP path). A mock/offline LLM choice needs no external opt-in
per ``easyicu.ai_optin.is_offline_llm_choice``; the contract itself never
requires opt-in because it never calls out.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LITERATURE_RETRIEVAL_CONTRACT_SCHEMA_VERSION = "easyicu.literature_retrieval_contract/1"

LiteratureRetrievalDecision = Literal[
    "population",
    "outcome_window",
    "method_applicability",
]

LITERATURE_RETRIEVAL_DECISIONS: tuple[LiteratureRetrievalDecision, ...] = (
    "population",
    "outcome_window",
    "method_applicability",
)

_CITATION_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$"

#: Sources that count as bibliographic retrieval. ``llm_extension`` suggestions
#: and bare ``curated`` seeds are deliberately excluded: they are not a search.
BIBLIOGRAPHIC_RETRIEVAL_SOURCES = frozenset({"pubmed", "tavily", "bound_search"})

#: Planner design-element authority projected onto the three hard decisions.
#: ``reporting`` is intentionally unmapped: it is not a hard dependency.
DESIGN_ELEMENT_TO_DECISION: dict[str, LiteratureRetrievalDecision] = {
    "population": "population",
    "outcome": "outcome_window",
    "time_zero": "outcome_window",
    "exposure": "outcome_window",
    "estimand": "method_applicability",
    "adjustment": "method_applicability",
    "dependence": "method_applicability",
    "missing_data": "method_applicability",
    "robustness": "method_applicability",
}

#: Seven-dimension design authority projected onto the three hard decisions.
DIMENSION_TO_DECISION: dict[str, LiteratureRetrievalDecision] = {
    "study_population": "population",
    "time_zero_and_windows": "outcome_window",
    "variable_operationalization": "outcome_window",
    "missingness_and_censoring": "method_applicability",
    "primary_model_and_sensitivities": "method_applicability",
}


def decision_for_design_element(element: str) -> LiteratureRetrievalDecision | None:
    """Map one planner design element onto a hard decision, if any."""
    return DESIGN_ELEMENT_TO_DECISION.get(str(element or "").strip())


def decision_for_design_dimension(dimension: str) -> LiteratureRetrievalDecision | None:
    """Map one seven-dimension authority dimension onto a hard decision."""
    return DIMENSION_TO_DECISION.get(str(dimension or "").strip())


def is_bibliographic_source(source: str) -> bool:
    """Return True for a source that counts as bibliographic retrieval."""
    return str(source or "").strip().lower() in BIBLIOGRAPHIC_RETRIEVAL_SOURCES


class LiteratureRetrievalSearch(BaseModel):
    """One exact query issued to one bibliographic source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=3, max_length=2000)

    @field_validator("source", "query", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class LiteratureRetrievalHit(BaseModel):
    """One retrieved record plus the exact query lineage that returned it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str = Field(pattern=_CITATION_KEY_PATTERN)
    source: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=3, max_length=2000)
    pmid: str | None = Field(default=None, max_length=32)

    @field_validator("source", "query", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("citation_key", "pmid", mode="before")
    @classmethod
    def _strip_optional(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class LiteratureRetrievalSupport(BaseModel):
    """Which retrieved record supports which hard decision, and with which sentence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: LiteratureRetrievalDecision
    citation_key: str = Field(pattern=_CITATION_KEY_PATTERN)
    supporting_statement: str = Field(min_length=12, max_length=1200)
    locator: str | None = Field(default=None, max_length=500)

    @field_validator("citation_key", "supporting_statement", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("locator", mode="before")
    @classmethod
    def _strip_locator(cls, value: Any) -> Any:
        if value is None:
            return None
        return value.strip() if isinstance(value, str) else value


class LiteratureRetrievalContract(BaseModel):
    """Typed retrieval evidence bound to a research question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[LITERATURE_RETRIEVAL_CONTRACT_SCHEMA_VERSION] = (
        LITERATURE_RETRIEVAL_CONTRACT_SCHEMA_VERSION
    )
    research_question: str = Field(min_length=8, max_length=2000)
    searches: list[LiteratureRetrievalSearch] = Field(default_factory=list)
    hits: list[LiteratureRetrievalHit] = Field(default_factory=list)
    supports: list[LiteratureRetrievalSupport] = Field(default_factory=list)
    search_conducted: bool = False
    sources_returning: list[str] = Field(default_factory=list)
    # Review finding: a recorded supporting sentence is not proof its
    # content came from the literature. ``source_backed`` maps each cited
    # record to the hard decisions for which the bundle holds reviewed
    # full-text design evidence (``design_evidence_cards``); the gate
    # reports uncovered decisions as warnings so reviewers see exactly
    # which claims lack source backing.
    source_backed: dict[str, list[LiteratureRetrievalDecision]] = Field(
        default_factory=dict
    )

    @field_validator("research_question", mode="before")
    @classmethod
    def _strip_question(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> "LiteratureRetrievalContract":
        search_pairs = {(item.source, item.query) for item in self.searches}
        if len(search_pairs) != len(self.searches):
            raise ValueError("literature retrieval searches must be unique")
        hit_keys = [item.citation_key for item in self.hits]
        if len(hit_keys) != len(set(hit_keys)):
            raise ValueError("literature retrieval hits must be unique")
        support_pairs = [
            (item.decision, item.citation_key) for item in self.supports
        ]
        if len(support_pairs) != len(set(support_pairs)):
            raise ValueError("literature retrieval supports must be unique")
        for hit in self.hits:
            if (hit.source, hit.query) not in search_pairs:
                raise ValueError(
                    "literature retrieval hit query is not among recorded searches: "
                    f"{hit.citation_key!r}"
                )
        hit_key_set = set(hit_keys)
        for support in self.supports:
            if support.citation_key not in hit_key_set:
                raise ValueError(
                    "literature retrieval support cites a non-retrieved record: "
                    f"{support.citation_key!r}"
                )
        return self


def _provenance_text_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for raw_source, raw_queries in value.items():
        source = str(raw_source or "").strip()
        if not source or not isinstance(raw_queries, list):
            continue
        queries = [
            str(item).strip() for item in raw_queries if str(item or "").strip()
        ]
        if queries:
            out[source] = queries
    return out


def supports_from_plan(plan: Any) -> list[LiteratureRetrievalSupport]:
    """Project plan literature bindings onto contract supports.

    Steps contribute ``literature_design_bindings`` (design elements +
    citation key + application sentence); selected design candidates
    contribute ``literature_design_decisions`` (dimension + rationale +
    citation keys).  Malformed bindings (unmapped element, bad key, short
    statement) are skipped: they surface downstream as uncovered-decision
    findings, never as finalization crashes.
    """

    supports: list[LiteratureRetrievalSupport] = []
    seen: set[tuple[str, str]] = set()

    def _offer(decision: Any, key: Any, statement: Any) -> None:
        if decision is None:
            return
        pair = (str(decision), str(key or "").strip())
        if pair in seen:
            return
        try:
            item = LiteratureRetrievalSupport(
                decision=decision,
                citation_key=str(key or "").strip(),
                supporting_statement=str(statement or "").strip(),
            )
        except ValueError:
            return
        seen.add(pair)
        supports.append(item)

    for step in list(getattr(plan, "steps", None) or []):
        for binding in list(getattr(step, "literature_design_bindings", None) or []):
            key = str(getattr(binding, "citation_key", "") or "").strip()
            statement = str(getattr(binding, "application", "") or "").strip()
            for element in list(getattr(binding, "design_elements", None) or []):
                _offer(decision_for_design_element(str(element or "")), key, statement)
    selection = getattr(plan, "design_selection", None)
    candidates = (
        list(getattr(selection, "candidates", None) or [])
        if selection is not None
        else []
    )
    for candidate in candidates:
        if str(getattr(candidate, "disposition", "") or "") != "selected":
            continue
        for item in list(getattr(candidate, "literature_design_decisions", None) or []):
            decision = decision_for_design_dimension(
                str(getattr(item, "dimension", "") or "")
            )
            rationale = str(getattr(item, "rationale", "") or "").strip()
            for key in list(getattr(item, "citation_keys", None) or []):
                _offer(decision, key, rationale)
    return supports


def contract_for_plan_finalization(
    *,
    plan: Any,
    preplan_literature: Any = None,
) -> LiteratureRetrievalContract:
    """Build the finalization contract from bundle plus plan supports.

    No bundle produces an explicitly unconducted contract (with a sentinel
    question): the gate then reports ``literature_search_not_conducted``
    instead of finalization crashing on a missing input.
    """

    if preplan_literature is None:
        return LiteratureRetrievalContract(
            research_question="retrieval evidence absent for this plan",
            search_conducted=False,
        )
    question = str(
        getattr(preplan_literature, "research_question", "") or ""
    ).strip()
    return contract_from_bundle(
        preplan_literature,
        supports=supports_from_plan(plan),
        research_question=(
            question or "retrieval bundle without a recorded research question"
        ),
    )


def contract_from_bundle(
    bundle: Any,
    supports: Sequence[LiteratureRetrievalSupport | dict[str, Any]] = (),
    research_question: str | None = None,
) -> LiteratureRetrievalContract:
    """Project an existing LiteratureBundle into a retrieval contract.

    Pure projection: reads ``search_provenance`` (exact queries + per-record
    lineage produced by the live-PubMed path), ``citations`` (PMID lookup),
    and ``screening_decisions`` (per-record source). Skips ``llm_extension``
    sources. Performs no network, provider, or filesystem I/O.
    """

    provenance = getattr(bundle, "search_provenance", None)
    search_queries = _provenance_text_map(
        getattr(provenance, "search_queries", None)
    )
    record_queries = _provenance_text_map(
        getattr(provenance, "record_queries", None)
    )
    search_conducted = bool(getattr(provenance, "search_conducted", False))
    sources_returning = [
        str(item).strip()
        for item in (getattr(provenance, "sources_returning", None) or [])
        if str(item or "").strip()
    ]
    citations = list(getattr(bundle, "citations", None) or [])
    pmid_by_key: dict[str, str] = {}
    for record in citations:
        key = str(getattr(record, "key", "") or "").strip()
        pmid = str(getattr(record, "pmid", "") or "").strip()
        if key and pmid:
            pmid_by_key[key] = pmid
    source_by_key: dict[str, str] = {}
    for decision in list(getattr(bundle, "screening_decisions", None) or []):
        key = str(getattr(decision, "citation_key", "") or "").strip()
        source = str(getattr(decision, "source", "") or "").strip()
        if key and source and key not in source_by_key:
            source_by_key[key] = source

    searches = [
        LiteratureRetrievalSearch(source=source, query=query)
        for source, queries in sorted(search_queries.items())
        for query in queries
        if source != "llm_extension"
    ]
    query_source_by_query: dict[str, str] = {}
    for item in searches:
        query_source_by_query.setdefault(item.query, item.source)

    hits: list[LiteratureRetrievalHit] = []
    for key in sorted(record_queries):
        queries = record_queries[key]
        if not queries:
            continue
        primary_query = queries[0]
        source = source_by_key.get(key) or query_source_by_query.get(
            primary_query, ""
        )
        if not source or source == "llm_extension":
            continue
        hits.append(
            LiteratureRetrievalHit(
                citation_key=key,
                source=source,
                query=primary_query,
                pmid=pmid_by_key.get(key),
            )
        )

    normalised_supports = [
        item if isinstance(item, LiteratureRetrievalSupport) else LiteratureRetrievalSupport.model_validate(item)
        for item in supports
    ]
    backed: dict[str, list[LiteratureRetrievalDecision]] = {}
    for card in list(getattr(bundle, "design_evidence_cards", None) or []):
        key = str(getattr(card, "citation_key", "") or "").strip()
        if not key:
            continue
        decisions: list[LiteratureRetrievalDecision] = []
        for item in list(getattr(card, "evidence", None) or []):
            decision = decision_for_design_dimension(
                str(getattr(item, "dimension", "") or "")
            )
            if decision is not None and decision not in decisions:
                decisions.append(decision)
        if decisions:
            backed[key] = decisions
    override = str(research_question or "").strip()
    return LiteratureRetrievalContract(
        research_question=override
        or str(getattr(bundle, "research_question", "") or "").strip(),
        searches=searches,
        hits=hits,
        supports=normalised_supports,
        search_conducted=search_conducted,
        sources_returning=sources_returning,
        source_backed=backed,
    )


__all__ = [
    "BIBLIOGRAPHIC_RETRIEVAL_SOURCES",
    "DESIGN_ELEMENT_TO_DECISION",
    "DIMENSION_TO_DECISION",
    "LITERATURE_RETRIEVAL_CONTRACT_SCHEMA_VERSION",
    "LITERATURE_RETRIEVAL_DECISIONS",
    "LiteratureRetrievalContract",
    "LiteratureRetrievalDecision",
    "LiteratureRetrievalHit",
    "LiteratureRetrievalSearch",
    "LiteratureRetrievalSupport",
    "contract_for_plan_finalization",
    "contract_from_bundle",
    "supports_from_plan",
    "decision_for_design_dimension",
    "decision_for_design_element",
    "is_bibliographic_source",
]
