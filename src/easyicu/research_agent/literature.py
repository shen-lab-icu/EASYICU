"""LiteratureAgent — ground the manuscript in prior work.

Inspired by the literature-review module of OpenLens-AI [1] but
designed to fit the EasyICU traceability story:

* every citation becomes a registered :class:`EvidenceRecord`, so
  manuscript sentences cite literature the same way they cite tables;
* the agent works fully offline through a small curated registry of
  canonical ICU references — useful in CI, useful as a baseline, and
  the only behaviour reviewers can audit deterministically;
* a real LLM client can populate richer citations through the
  standard ``LLMClient`` protocol; the same evidence-binding flow
  applies;
* T2.2 — when ``enable_pubmed=True`` and the host can reach NCBI's
  E-utilities, :class:`PubMedLiteratureClient` augments the bundle
  with live PubMed hits. All three sources merge into the same
  ``CitationRecord`` shape so the manuscript binder treats them
  uniformly.

References
----------
[1] OpenLens-AI: Fully Autonomous Research Agent for Health Informatics.
    https://github.com/jarrycyx/openlens-ai
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .llm import LLMClient, LLMMessage, MockLLMClient
from .schema import ResearchContext, VariableRole


class CitationRecord(BaseModel):
    """A single literature reference."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="Stable citation key, e.g. 'vincent_sofa_1996'.")
    title: str
    year: str
    venue: Optional[str] = None
    relevance: Optional[str] = Field(
        default=None,
        description="Why this paper is cited in this run — used by the writer to pick what to mention.",
    )
    doi: Optional[str] = None
    url: Optional[str] = None
    pmid: Optional[str] = None


class LiteratureBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    research_question: str
    citations: List[CitationRecord]


# ---------------------------------------------------------------------------
# Curated offline registry
# ---------------------------------------------------------------------------


_CURATED: List[CitationRecord] = [
    CitationRecord(
        key="vincent_sofa_1996",
        title="The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure.",
        year="1996", venue="Intensive Care Medicine",
        relevance="Defines SOFA components (0-4 ordinal); foundational for any SOFA-based analysis.",
        pmid="8844239",
    ),
    CitationRecord(
        key="singer_sepsis3_2016",
        title="The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).",
        year="2016", venue="JAMA",
        relevance="Sepsis-3 reframes sepsis around SOFA-defined organ dysfunction.",
        pmid="26903338",
    ),
    CitationRecord(
        key="kdigo_aki_2012",
        title="KDIGO Clinical Practice Guideline for Acute Kidney Injury.",
        year="2012", venue="Kidney International Supplements",
        relevance="Defines KDIGO AKI staging used by EasyICU's AKI module.",
    ),
    CitationRecord(
        key="ricu_2023",
        title="ricu: R's interface to intensive care data.",
        year="2023", venue="Software",
        relevance="Conceptual ancestor of EasyICU's concept dictionary and table model.",
        url="https://github.com/eth-mds/ricu",
    ),
    CitationRecord(
        key="pollard_eicu_2018",
        title="The eICU Collaborative Research Database, a freely available multi-center database for critical care research.",
        year="2018", venue="Scientific Data",
        relevance="Source database used in cross-database replication.",
        pmid="30204154",
    ),
    CitationRecord(
        key="johnson_mimiciv_2023",
        title="MIMIC-IV, a freely accessible electronic health record dataset.",
        year="2023", venue="Scientific Data",
        relevance="Primary source database used by EasyICU.",
    ),
    CitationRecord(
        key="hyland_hirid_2020",
        title="Early prediction of circulatory failure in the intensive care unit using machine learning.",
        year="2020", venue="Nature Medicine",
        relevance="Source paper for HiRID and circEWS-style circulatory-failure definitions.",
    ),
    CitationRecord(
        key="openlens_ai_2025",
        title="OpenLens-AI: Fully Autonomous Research Agent for Health Informatics.",
        year="2025", venue="Software",
        relevance=("Referenced as a baseline general medical research agent. EasyICU's research-agent layer "
                   "differs by injecting ICU-aware concept context and routing every artefact through a "
                   "hashed evidence store."),
        url="https://github.com/jarrycyx/openlens-ai",
    ),
    CitationRecord(
        key="m4_clinical_research_2025",
        title="M4: Infrastructure for AI-Assisted Clinical Research (MCP + clinical-skills tooling).",
        year="2025", venue="Software",
        relevance="Inspires the clinical-skills registry and MCP server in this layer.",
    ),
    CitationRecord(
        key="healthflow_2025",
        title="HealthFlow: A Self-Evolving AI Agent with Meta-Planning for Autonomous Healthcare Research.",
        year="2025", venue="Preprint",
        relevance="Inspires the run-memory module and the self-improving planner story.",
    ),
]


def _curated_for(ctx: ResearchContext) -> List[CitationRecord]:
    """Filter the curated list by which concepts appear in the context.

    Matching is *prefix-aware* — ``kdigo_stage`` triggers the KDIGO
    citation, ``sofa2_resp`` triggers the Vincent SOFA citation, and
    so on. This sidesteps the previous fragility where renaming a
    column from ``kdigo`` to ``kdigo_stage`` silently dropped the
    canonical reference.
    """
    names = {v.name.lower() for v in ctx.variables}
    out: List[CitationRecord] = []

    def _add(c: CitationRecord) -> None:
        if c not in out:
            out.append(c)

    def _matches_prefix(prefixes: Sequence[str]) -> bool:
        return any(n.startswith(tuple(prefixes)) for n in names)

    if _matches_prefix(("sofa", "sofa2")):
        _add(_CURATED[0])  # Vincent 1996
    if _matches_prefix(("sep3", "sepsis", "sep2", "lact", "susp_inf")):
        _add(_CURATED[1])  # Sepsis-3
    if _matches_prefix(("creat", "kdigo", "aki")):
        _add(_CURATED[2])  # KDIGO
    # Always cite the database papers and EasyICU lineage.
    _add(_CURATED[3])  # ricu
    db = ctx.cohort.database.lower()
    if db.startswith("eicu"):
        _add(_CURATED[4])
    if db.startswith("mim") or db == "miiv":
        _add(_CURATED[5])
    if db.startswith("hirid"):
        _add(_CURATED[6])
    # Always cite related agent work for transparency.
    _add(_CURATED[7])  # OpenLens
    _add(_CURATED[8])  # M4
    _add(_CURATED[9])  # HealthFlow
    return out


# ---------------------------------------------------------------------------
# PubMed live client (T2.2)
# ---------------------------------------------------------------------------


_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedLiteratureClient:
    """NCBI E-utilities client for PubMed lookups (T2.2).

    Uses ``esearch.fcgi`` to find PMIDs, then ``esummary.fcgi`` to pull
    titles, journals, dates and DOIs. Returns the same
    :class:`CitationRecord` shape as the offline registry so the
    LiteratureAgent can merge them without further translation.

    Designed for graceful degradation: every network/parse error
    becomes an empty result list, never a raised exception. Callers
    can rely on a non-empty list to mean "PubMed succeeded" and
    treat the empty case identically to a network outage.

    NCBI etiquette
    --------------
    NCBI's E-utilities ask for ``tool=`` and ``email=`` parameters on
    every request, plus an ``api_key`` if you have one (raises the
    rate limit from 3 req/s to 10). All three are passed through
    unchanged.

    Pure stdlib (``urllib.request`` + ``json``) — no SDK to add to the
    dependency surface; the network round-trips are tiny.
    """

    name = "pubmed"

    def __init__(
        self,
        *,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        tool: str = "easyicu-research-agent",
        timeout: float = 15.0,
        base_url: str = _PUBMED_BASE,
    ) -> None:
        self.email = email
        self.api_key = api_key
        self.tool = tool
        self.timeout = float(timeout)
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _with_etiquette(self, params: Dict[str, str]) -> Dict[str, str]:
        """Add NCBI etiquette parameters (tool, email, api_key) to the
        outgoing params *before* they reach the transport layer.

        Keeping this merge above ``_http_get`` is intentional: tests
        that stub ``_http_get`` (the transport seam) need to see the
        etiquette parameters in the captured params dict, otherwise
        the stub would be blind to NCBI policy compliance.
        """
        merged = dict(params)
        merged.setdefault("tool", self.tool)
        if self.email:
            merged.setdefault("email", self.email)
        if self.api_key:
            merged.setdefault("api_key", self.api_key)
        return merged

    def _http_get(self, path: str, params: Dict[str, str]) -> Optional[bytes]:
        url = f"{self.base_url}/{path}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return resp.read()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, *, retmax: int = 8) -> List[CitationRecord]:
        """Search PubMed for ``query`` and return up to ``retmax`` records."""
        if not query:
            return []
        ids = self._esearch(query, retmax=retmax)
        if not ids:
            return []
        return self._esummary(ids)

    def search_for_context(
        self,
        context: ResearchContext,
        *,
        retmax: int = 8,
    ) -> List[CitationRecord]:
        """Build a query from the :class:`ResearchContext` and search PubMed."""
        return self.search(build_pubmed_query_for_context(context), retmax=retmax)

    # ------------------------------------------------------------------
    # E-utilities calls (private)
    # ------------------------------------------------------------------

    def _esearch(self, query: str, *, retmax: int) -> List[str]:
        body = self._http_get("esearch.fcgi", self._with_etiquette({
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(int(retmax)),
            "sort": "relevance",
        }))
        if not body:
            return []
        try:
            payload = json.loads(body)
        except Exception:
            return []
        ids = payload.get("esearchresult", {}).get("idlist", [])
        return [str(x) for x in ids if x]

    def _esummary(self, pmids: Sequence[str]) -> List[CitationRecord]:
        body = self._http_get("esummary.fcgi", self._with_etiquette({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }))
        if not body:
            return []
        try:
            payload = json.loads(body)
        except Exception:
            return []
        return parse_pubmed_esummary(payload)


# ---------------------------------------------------------------------------
# Query construction + esummary parsing — kept module-level so they can be
# unit-tested without spinning up a network round-trip.
# ---------------------------------------------------------------------------


_ICU_FILTER = (
    "(intensive care[Title/Abstract] OR \"critical care\"[Title/Abstract] "
    "OR ICU[Title/Abstract])"
)

# Variables in these roles are good PubMed query terms; ids/timestamps are not.
_QUERY_ROLES = {
    VariableRole.COMPOSITE_SCORE,
    VariableRole.ORDINAL_SCORE,
    VariableRole.LAB,
    VariableRole.INTERVENTION,
    VariableRole.OUTCOME,
}


def build_pubmed_query_for_context(context: ResearchContext) -> str:
    """Compose a PubMed search query from a :class:`ResearchContext`.

    The shape is::

        ("<research question, sanitised>") AND <var1> AND <var2> ... AND <ICU filter>

    The ICU filter is always appended so we don't drag in unrelated
    biomedical literature when a variable name (e.g. "lactate") is
    common in non-ICU contexts.
    """
    terms: List[str] = []
    q = (context.research_question or "").strip()
    q_clean = re.sub(r"[^A-Za-z0-9\s\-]", " ", q)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    if q_clean:
        terms.append(f"({q_clean})")
    var_names: List[str] = []
    for v in context.variables:
        if v.role not in _QUERY_ROLES:
            continue
        n = v.name.strip().lower()
        if n and n not in var_names:
            var_names.append(n)
    # Cap the variable list so the query stays under PubMed's URL
    # length budget for esearch.
    for n in var_names[:4]:
        terms.append(n)
    terms.append(_ICU_FILTER)
    return " AND ".join(terms)


_SKIP_TITLE_WORDS = {"the", "and", "for", "with", "from", "into", "this", "that",
                     "study", "review", "analysis", "based"}


def _slug_from_title(title: str) -> str:
    for w in re.split(r"\W+", (title or "").lower()):
        if len(w) >= 4 and w not in _SKIP_TITLE_WORDS:
            return w
    return ""


def _surname_from_authors(authors: Any) -> str:
    if not isinstance(authors, list):
        return ""
    for a in authors:
        if not isinstance(a, dict):
            continue
        name = (a.get("name") or "").strip()
        if not name:
            continue
        # PubMed esummary names are formatted "Vincent JL" → surname is first token.
        first = name.split()[0]
        return re.sub(r"[^A-Za-z]", "", first).lower()
    return ""


def _doi_from_articleids(articleids: Any) -> Optional[str]:
    if not isinstance(articleids, list):
        return None
    for el in articleids:
        if isinstance(el, dict) and el.get("idtype") == "doi":
            value = (el.get("value") or "").strip()
            if value:
                return value
    return None


def _year_from_pubdate(pubdate: Any) -> str:
    s = str(pubdate or "")
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else "n/a"


def parse_pubmed_esummary(payload: Dict[str, Any]) -> List[CitationRecord]:
    """Parse an ``esummary.fcgi?retmode=json`` payload into ``CitationRecord``s.

    Tolerant of missing fields: each record is best-effort and any
    record that cannot be turned into a valid pydantic instance is
    silently dropped, matching the LiteratureAgent's "never inject
    something we cannot validate" rule.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return []
    uids = result.get("uids")
    if not isinstance(uids, list):
        return []
    out: List[CitationRecord] = []
    for uid in uids:
        rec = result.get(str(uid))
        if not isinstance(rec, dict):
            continue
        title = (rec.get("title") or "").rstrip(" .")
        year = _year_from_pubdate(rec.get("pubdate"))
        venue = (rec.get("fulljournalname") or rec.get("source") or None) or None
        doi = _doi_from_articleids(rec.get("articleids"))
        surname = _surname_from_authors(rec.get("authors"))
        slug = _slug_from_title(title)
        key_parts = [p for p in (surname, slug, year if year != "n/a" else "") if p]
        key = "_".join(key_parts) if key_parts else f"pmid_{uid}"
        try:
            citation = CitationRecord(
                key=key,
                title=title or f"PMID {uid}",
                year=year,
                venue=venue,
                doi=doi,
                pmid=str(uid),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                relevance=None,
            )
        except Exception:
            continue
        out.append(citation)
    return out


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LiteratureAgent:
    """Produce a :class:`LiteratureBundle` for a research context.

    Three composable sources:

    1. **Curated offline registry** (default) — a small, hand-vetted
       list of canonical ICU references filtered by the concepts in
       scope. Deterministic, no network, no API key.
    2. **PubMed live** — when ``enable_pubmed=True`` AND a
       :class:`PubMedLiteratureClient` is supplied (or constructed
       implicitly), the agent issues an esearch + esummary query
       built from the research question and the variables in scope,
       and merges the hits into the bundle.
    3. **LLM extension** — when given a real :class:`LLMClient`, asks
       the model to extend the merged list with extra recent
       references. Output is parsed back into ``CitationRecord``
       objects; anything that fails parsing is silently dropped,
       never injected into the manuscript without provenance.

    All three sources serialise to the same :class:`CitationRecord`
    schema so the manuscript binder treats them uniformly. Sources
    later in the list deduplicate against earlier ones (key first,
    PMID second) so curated entries always win on conflict.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        enable_pubmed: bool = False,
        pubmed_client: Optional["PubMedLiteratureClient"] = None,
        pubmed_retmax: int = 8,
    ) -> None:
        self.llm = llm
        self.enable_pubmed = bool(enable_pubmed)
        self.pubmed_client = pubmed_client
        self.pubmed_retmax = int(pubmed_retmax)

    def run(self, context: ResearchContext) -> LiteratureBundle:
        baseline = _curated_for(context)
        merged = list(baseline)
        seen_keys = {c.key for c in merged}
        seen_pmids = {c.pmid for c in merged if c.pmid}

        # 2) PubMed live (T2.2). Errors are swallowed: the bundle is
        #    still useful even if the network is unreachable.
        if self.enable_pubmed:
            client = self.pubmed_client or PubMedLiteratureClient()
            try:
                hits = client.search_for_context(context, retmax=self.pubmed_retmax)
            except Exception:
                hits = []
            for rec in hits:
                if rec.key in seen_keys or (rec.pmid and rec.pmid in seen_pmids):
                    continue
                seen_keys.add(rec.key)
                if rec.pmid:
                    seen_pmids.add(rec.pmid)
                merged.append(rec)

        # 3) LLM extension (only when a real client is provided).
        if self.llm is not None and not isinstance(self.llm, MockLLMClient):
            existing_keys = ", ".join(c.key for c in merged)
            msgs = [
                LLMMessage(role="system", content=(
                    "You are a literature-review agent for an ICU research pipeline. "
                    "Return JSON with a 'citations' array; each item has key, title, year, "
                    "venue, relevance, optionally doi/url/pmid. Cite only papers you are "
                    "confident exist; do not fabricate."
                )),
                LLMMessage(role="user", content=(
                    f"LITERATURE REVIEW request for the question: {context.research_question!r}. "
                    f"Concepts in scope: {[v.name for v in context.variables]}. "
                    f"Already-included keys (do not duplicate): {existing_keys}. "
                    "Add 3-6 additional canonical references (or recent preprints) most "
                    "relevant to this exact ICU question."
                )),
            ]
            try:
                raw = self.llm.complete(msgs, max_tokens=1024, temperature=0.0)
                data = _parse_citation_json(raw)
            except Exception:
                data = []
            for d in data:
                try:
                    rec = CitationRecord.model_validate(d)
                except Exception:
                    continue
                if rec.key in seen_keys or (rec.pmid and rec.pmid in seen_pmids):
                    continue
                seen_keys.add(rec.key)
                if rec.pmid:
                    seen_pmids.add(rec.pmid)
                merged.append(rec)

        return LiteratureBundle(
            research_question=context.research_question,
            citations=merged,
        )


def _parse_citation_json(raw: str) -> List[Dict]:
    text = raw.strip()
    # strip code fences
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and "citations" in data:
        items = data["citations"]
    elif isinstance(data, list):
        items = data
    else:
        return []
    return [d for d in items if isinstance(d, dict)]


__all__ = [
    "CitationRecord",
    "LiteratureBundle",
    "LiteratureAgent",
    "PubMedLiteratureClient",
    "build_pubmed_query_for_context",
    "parse_pubmed_esummary",
]
