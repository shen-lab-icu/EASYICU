"""Source-verified author metadata, independent of scientific citation eligibility.

PubMed AuthorList and Crossref author metadata, checked 2026-09-08.
The offline snapshot only fills empty authors on an exact bibliographic match.
It never selects papers, changes screening, or grants full-text/claim authority.
"""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Protocol
from datetime import datetime, timezone
from threading import Lock
from time import monotonic, sleep
import urllib.parse
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from .literature_excerpt import select_source_backed_excerpt
from .planning.literature_contract import ImpactProxies, RecordReview, ReviewPolicy


PUBMED_SNAPSHOT_SHA256 = "3fb8330bbe3a874d9f52db3cec04f1712c4d37eebbd944c6b385b97cd05ddd52"
_SOURCES = (
    {
        "pmid": "36596836", "doi": "10.1038/s41597-022-01899-x", "year": "2023",
        "venue": "Scientific Data",
        "title": "MIMIC-IV, a freely accessible electronic health record dataset.",
        "authors": (
            "Alistair E W Johnson", "Lucas Bulgarelli", "Lu Shen", "Alvin Gayles",
            "Ayad Shammout", "Steven Horng", "Tom J Pollard", "Sicheng Hao",
            "Benjamin Moody", "Brian Gow", "Li-Wei H Lehman", "Leo A Celi", "Roger G Mark",
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/36596836/",
        "snapshot_sha256": "2571e086a745c373ad518b395bc5ec49897c6d8277d0cc76e399171e9ed97944",
    },
    {
        "pmid": "26903338", "doi": "10.1001/jama.2016.0287", "year": "2016", "venue": "JAMA",
        "title": "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).",
        "authors": (
            "Mervyn Singer", "Clifford S Deutschman", "Christopher Warren Seymour",
            "Manu Shankar-Hari", "Djillali Annane", "Michael Bauer", "Rinaldo Bellomo",
            "Gordon R Bernard", "Jean-Daniel Chiche", "Craig M Coopersmith",
            "Richard S Hotchkiss", "Mitchell M Levy", "John C Marshall", "Greg S Martin",
            "Steven M Opal", "Gordon D Rubenfeld", "Tom van der Poll", "Jean-Louis Vincent",
            "Derek C Angus",
        ),
    },
    {
        "pmid": "18056625", "doi": "10.1093/aje/kwm324", "year": "2008",
        "venue": "American Journal of Epidemiology",
        "title": "Immortal time bias in pharmaco-epidemiology.",
        "authors": ("Samy Suissa",),
    },
    {
        "pmid": "17938396", "doi": "10.7326/0003-4819-147-8-200710160-00010", "year": "2007",
        "venue": "Annals of Internal Medicine",
        "title": "The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies.",
        "authors": (
            "Erik von Elm", "Douglas G Altman", "Matthias Egger", "Stuart J Pocock",
            "Peter C Gøtzsche", "Jan P Vandenbroucke", "STROBE Initiative",
        ),
    },
    {
        "pmid": "26440803", "doi": "10.1371/journal.pmed.1001885", "year": "2015",
        "venue": "PLoS Medicine",
        "title": "The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) statement.",
        "authors": (
            "Eric I Benchimol", "Liam Smeeth", "Astrid Guttmann", "Katie Harron",
            "David Moher", "Irene Petersen", "Henrik T Sørensen", "Erik von Elm",
            "Sinéad M Langan", "RECORD Working Committee",
        ),
    },
    {
        "pmid": "19564179",
        "title": "Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls.",
        "venue": "BMJ",
        "year": "2009",
        "doi": "10.1136/bmj.b2393",
        "authors": [
            "Jonathan A C Sterne",
            "Ian R White",
            "John B Carlin",
            "Michael Spratt",
            "Patrick Royston",
            "Michael G Kenward",
            "Angela M Wood",
            "James R Carpenter",
        ],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/19564179/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "6668489",
        "title": "Analysis of survival by tumor response.",
        "venue": "Journal of Clinical Oncology",
        "year": "1983",
        "doi": "10.1200/JCO.1983.1.11.710",
        "authors": ["J R Anderson", "K C Cain", "R D Gelber"],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/6668489/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "21611958",
        "title": "The use of restricted mean survival time to estimate the treatment effect in randomized clinical trials when the proportional hazards assumption is in doubt.",
        "venue": "Statistics in medicine",
        "year": "2011",
        "doi": "10.1002/sim.4274",
        "authors": ["Patrick Royston", "Mahesh K B Parmar"],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/21611958/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "2657958",
        "title": "Flexible regression models with cubic splines.",
        "venue": "Statistics in medicine",
        "year": "1989",
        "doi": "10.1002/sim.4780080504",
        "authors": ["S Durrleman", "R Simon"],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/2657958/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "",
        "doi": "10.1093/biomet/81.3.515",
        "title": "Proportional hazards tests and diagnostics based on weighted residuals",
        "venue": "Biometrika",
        "year": "1994",
        "authors": ["PATRICIA M. GRAMBSCH", "TERRY M. THERNEAU"],
        "source_url": "https://api.crossref.org/works/10.1093/biomet/81.3.515",
        "snapshot_sha256": "90f3d350e2fda0727ee80a13c56a64649a35ff02b3fd134e6154319c83d3c40d",
    },
)


def _text_identity(value: Any) -> str:
    # Punctuation/capitalisation only; no fuzzy title or author-key guessing.
    return re.sub(r"[\W_]", "", str(value or "").casefold())


def complete_missing_authors(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a copy plus a versioned metadata-only receipt; leave conflicts alone."""
    copied = dict(record)
    if record.get("authors"):
        return copied, None
    for source in _SOURCES:
        if not all(
            str(record.get(key) or "").strip().casefold() == source[key].casefold()
            for key in ("pmid", "doi", "year")
        ) or not all(
            _text_identity(record.get(key)) == _text_identity(source[key])
            for key in ("title", "venue")
        ):
            continue
        copied["authors"] = list(source["authors"])
        return copied, {
            "schema_version": "easyicu.bibliographic_metadata_completion/1",
            "scope": "bibliographic_metadata_only",
            "fields": ["authors"], "verified_on": "2026-09-08",
            "source_url": source.get("source_url", f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/"),
            "snapshot_sha256": source.get("snapshot_sha256", PUBMED_SNAPSHOT_SHA256),
            "record_sha256": sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        }
    return copied, None


class CitationRecord(BaseModel):
    """A single literature reference."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="Stable citation key, e.g. 'vincent_sofa_1996'.")
    title: str
    year: str
    authors: List[str] = Field(
        default_factory=list,
        description=(
            "Author display names retained in source order from bibliographic "
            "metadata. Missing names must not be inferred from the citation key."
        ),
    )
    venue: Optional[str] = None
    relevance: Optional[str] = Field(
        default=None,
        description="Why this paper is cited in this run — used by the writer to pick what to mention.",
    )
    doi: Optional[str] = None
    url: Optional[str] = None
    pmid: Optional[str] = None
    publication_types: List[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Source-issued bibliographic publication types. These are retained "
            "for deterministic comparator eligibility; absence is not proof of "
            "an observational design."
        ),
    )
    impact: ImpactProxies = Field(default_factory=ImpactProxies)
    review: RecordReview | None = None
    bibliographic_notices: List[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Source-issued corrections, retractions, expressions of concern, "
            "or other notices discovered during bibliographic verification."
        ),
    )


class PubMedContextSearchResult(BaseModel):
    """Bounded multi-stratum PubMed result with exact query lineage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.pubmed_context_search/1"
    records: List[CitationRecord] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    record_queries: Dict[str, List[str]] = Field(default_factory=dict)
    query_roles: Dict[str, List[str]] = Field(default_factory=dict)


class LiteratureScreeningDecision(BaseModel):
    """Deterministic, inspectable eligibility decision for one retrieved record."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str
    source: str
    disposition: Literal["include", "exclude"]
    evidence_role: Literal[
        "direct_comparator",
        "design_analogue",
        "definition",
        "method",
        "database",
        "related_context",
    ]
    rationale: str
    query: Optional[str] = None
    population_match: bool = False
    exposure_match: bool = False
    outcome_match: bool = False
    design_excerpt_available: bool = False
    publication_type_eligible: bool = True


_lock = Lock()
_next_request = 0.0


def _pace_request() -> None:
    global _next_request
    with _lock:
        delay = _next_request - monotonic()
        if delay > 0:
            sleep(delay)
        _next_request = monotonic() + 0.35


class OpenAlexSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CitationRecord] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    record_queries: dict[str, list[str]] = Field(default_factory=dict)
    query_roles: dict[str, list[str]] = Field(default_factory=dict)


def openalex_query(pubmed_query: str) -> str:
    """Preserve Boolean text alternatives while removing PubMed-only tags."""
    return re.sub(r"\[(?:Title/Abstract|Title|MeSH Terms)\]", "", pubmed_query)


def abstract_text(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions = {}
    for word, offsets in index.items():
        if not isinstance(word, str) or not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int) and 0 <= offset < 20_000:
                positions[offset] = word
    return " ".join(positions[i] for i in sorted(positions))


class OpenAlexLiteratureClient:
    name = "openalex"

    def __init__(self, *, timeout: float = 15.0, pace: Callable[[], None] = _pace_request):
        self.timeout = timeout
        self.pace = pace
        self.transport_failures: list[str] = []
        self._venues: dict[str, dict[str, Any]] = {}

    def reset_transport_failures(self) -> None:
        self.transport_failures.clear()
        self._venues.clear()

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any] | None:
        try:
            self.pace()
            url = "https://api.openalex.org/" + path + "?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                value = json.loads(response.read())
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except Exception as exc:
            code = getattr(exc, "code", None)
            reason = f"HTTP {code}" if code else f"{type(exc).__name__}: {exc}"
            self.transport_failures.append(f"{path}: {reason}")
            return None

    def _venue_stats(self, source: dict[str, Any]) -> dict[str, Any]:
        source_id = str(source.get("id") or "").rsplit("/", 1)[-1]
        if not re.fullmatch(r"S[0-9]+", source_id):
            return {}
        if source_id not in self._venues:
            payload = self._get_json(f"sources/{source_id}", {})
            stats = (payload or {}).get("summary_stats")
            self._venues[source_id] = stats if isinstance(stats, dict) else {}
            if payload is not None and not isinstance(stats, dict):
                self.transport_failures.append(f"sources/{source_id}: missing summary_stats")
        return self._venues[source_id]

    def _record(self, work: dict[str, Any], focus_terms: Sequence[str]) -> CitationRecord:
        work_id = str(work.get("id") or "").rsplit("/", 1)[-1]
        if not re.fullmatch(r"W[0-9]+", work_id) or not work.get("display_name"):
            raise ValueError("work has no valid id/title")
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        stats = self._venue_stats(source)
        count = work.get("cited_by_count")
        year = work.get("publication_year")
        current_year = datetime.now(timezone.utc).year
        age = current_year - year + 1 if isinstance(year, int) and 0 < year <= current_year else None
        metrics = {
            "citations_per_year": count / age if isinstance(count, int) and age else None,
            "venue_two_year_mean_citedness": stats.get("2yr_mean_citedness"),
            "venue_h_index": stats.get("h_index"),
        }
        impact = ImpactProxies(
            source="openalex", citation_count=count,
            proxies_used=[name for name, value in metrics.items() if value is not None],
            **metrics,
        )
        excerpt = select_source_backed_excerpt(
            abstract_text(work.get("abstract_inverted_index")), focus_terms=focus_terms,
            design_terms=("method", "regression", "defined", "adjusted", "database", "reporting", "sensitivity"),
            max_sentences=8, max_chars=2400,
        )
        ids = work.get("ids") or {}
        pmid = str(ids.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1]
        doi = str(work.get("doi") or "").removeprefix("https://doi.org/")
        publication_type = {
            "article": "Journal Article", "review": "Review", "editorial": "Editorial",
            "letter": "Letter", "preprint": "Preprint",
        }.get(str(work.get("type")), str(work.get("type") or ""))
        return CitationRecord(
            key="openalex_" + work_id, title=work["display_name"], year=str(year or ""),
            authors=[item["author"]["display_name"] for item in work.get("authorships", [])
                     if isinstance(item.get("author"), dict) and item["author"].get("display_name")],
            venue=source.get("display_name"), doi=doi or None,
            pmid=pmid if pmid.isdigit() else None, url=f"https://openalex.org/{work_id}",
            relevance="Study-design excerpt: " + excerpt if excerpt else None,
            publication_types=[publication_type] if publication_type else [],
            bibliographic_notices=["Retracted"] if work.get("is_retracted") else [],
            impact=impact,
        )

    def search_review(
        self, *, families: dict[str, list[str]], policy: ReviewPolicy,
        excerpt_terms: Sequence[str] = (),
    ) -> OpenAlexSearchResult:
        result = OpenAlexSearchResult()
        rows_by_query: list[list[dict[str, Any]]] = []
        for offset in range(max(map(len, families.values()), default=0)):
            for role, values in families.items():
                if offset >= len(values):
                    continue
                query = openalex_query(values[offset])
                result.query_roles.setdefault(role, []).append(query)
                if query in result.search_queries:
                    continue
                result.search_queries.append(query)
                payload = self._get_json("works", {"search": query, "per-page": str(policy.candidates_per_query)})
                rows = (payload or {}).get("results", [])
                if payload is not None and ("results" not in payload or not isinstance(rows, list)):
                    self.transport_failures.append("works: missing or malformed results")
                    rows = []
                rows_by_query.append(rows)
        by_id: dict[str, CitationRecord] = {}
        for offset in range(max(map(len, rows_by_query), default=0)):
            for query, rows in zip(result.search_queries, rows_by_query):
                if offset >= len(rows):
                    continue
                work = rows[offset]
                try:
                    work_id = str(work.get("id") or "")
                    record = by_id.get(work_id)
                    if record is None:
                        if len(by_id) >= policy.candidates_per_source:
                            continue
                        record = self._record(work, excerpt_terms)
                        by_id[work_id] = record
                    lineage = result.record_queries.setdefault(record.key, [])
                    if query not in lineage:
                        lineage.append(query)
                except (ValueError, TypeError, AttributeError, KeyError) as exc:
                    self.transport_failures.append(f"works: malformed record ({type(exc).__name__})")
        result.records = list(by_id.values())
        return result


class PubMedReviewTransport(Protocol):
    def _esearch(self, query: str, *, retmax: int) -> list[str]: ...
    def _hydrate_ids(self, pmids: Sequence[str], *, excerpt_terms: Sequence[str]) -> list[CitationRecord]: ...


def search_pubmed_review(
    client: PubMedReviewTransport, *, families: dict[str, list[str]],
    policy: ReviewPolicy, excerpt_terms: Sequence[str],
) -> PubMedContextSearchResult:
    """Allocate depth to every role before hydrating once, without a top-eight cut.

    Identified records are the bounded hydrated candidate pool, not ESearch's
    estimated total. Every returned candidate proceeds to the owner's screen.
    """
    # Round-robin query families so extra topic strata cannot starve other roles.
    queries = list(dict.fromkeys(
        values[offset] for offset in range(max(map(len, families.values()), default=0))
        for values in families.values() if offset < len(values)
    ))
    ids_by_query = [client._esearch(q, retmax=policy.candidates_per_query) for q in queries]
    membership: dict[str, list[str]] = {}
    for query, ids in zip(queries, ids_by_query):
        for pmid in ids:
            membership.setdefault(pmid, []).append(query)
    selected = list(dict.fromkeys(
        ids[offset] for offset in range(max(map(len, ids_by_query), default=0))
        for ids in ids_by_query if offset < len(ids)
    ))[:policy.candidates_per_source]
    records = client._hydrate_ids(selected, excerpt_terms=excerpt_terms) if selected else []
    return PubMedContextSearchResult(
        records=records, search_queries=queries,
        record_queries={r.key: membership.get(r.pmid or "", []) for r in records},
    )
