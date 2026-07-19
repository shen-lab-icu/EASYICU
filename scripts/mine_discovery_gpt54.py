#!/usr/bin/env python
"""Mine discovery research questions with the local gpt5.4 proxy (v2).

gpt5.4 sits in the EXTRACTION seat (not a keyword stub): targeted PubMed
prognostic/risk-factor ICU reviews -> gpt5.4 extracts gap-driven predictor->
outcome ideas -> map against the FULL miiv concept catalog (209 concepts) ->
outcome-blind joint feasibility probed on demand against the full export ->
gpt5.4 same-topic novelty screen -> ranked, preregistered `proposed`. Stops at
the human gate.

Two deliberate scoping choices (both methodological, not result cherry-picking):
  * literature scope is prognostic/risk-factor ICU reviews, because the only
    binary-determinable outcome the framework admits for feasibility ranking is
    mortality (`known_0_1`); diagnostic/management reviews surface non-mortality
    endpoints that are correctly non-executable.
  * available_concepts is the full export catalog, so predictors resolve to real
    concepts instead of an artificially narrow panel.

Run (proxy creds inline, never written to a controlled file):
    OPENAI_BASE_URL=http://127.0.0.1:8787/v1 OPENAI_API_KEY=*** \
    python scripts/mine_discovery_gpt54.py --model gpt5.4 --article-count 30 --top-k 15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from easyicu.research_agent.concept_catalog import load_concept_catalog  # noqa: E402
from easyicu.research_agent.data_catalog import build_available_catalog  # noqa: E402
from easyicu.research_agent.concept_catalog import (  # noqa: E402
    normalize_concept_name,
)
from easyicu.research_agent.idea_mining import (  # noqa: E402
    OutcomeDeterminability,
    run_idea_mining_dry_run,
)  # noqa: E402
from easyicu.research_agent.idea_mining_funnel import (  # noqa: E402
    LiteratureFunnelSpec,
    fetch_literature_funnel_corpus,
)
from easyicu.research_agent.discovery.idea_scope import (  # noqa: E402
    JOURNAL_PRESETS,
    LiteratureScopeSpec,
    resolve_journals,
)
from easyicu.research_agent.llm import OpenAIClient  # noqa: E402

import run_idea_mining_s6_validation_harness as H  # noqa: E402

EXPORT = Path(
    os.environ.get(
        "EASYICU_DISCOVERY_EXPORT",
        "/Volumes/外置硬盘/easyicu_fullexport_miiv_20260610",
    )
)

# Prognostic / risk-factor ICU review scope -> predictor->mortality ideas.
PROGNOSTIC_QUERY = (
    '("Crit Care"[Journal] OR "Intensive Care Med"[Journal] '
    'OR "Lancet Respir Med"[Journal] OR "Am J Respir Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal]) '
    "AND (review[Publication Type] OR editorial[Publication Type]) "
    "AND (mortality[Title/Abstract] OR prognosis[Title/Abstract] "
    'OR "risk factor"[Title/Abstract] OR "risk factors"[Title/Abstract] '
    "OR outcome[Title/Abstract] OR predictor[Title/Abstract] "
    "OR prognostic[Title/Abstract]) "
    'AND (ICU[Title/Abstract] OR "intensive care"[Title/Abstract] '
    'OR "critically ill"[Title/Abstract])'
)


def _build_column_index() -> dict[str, str]:
    """Map every export column to its file, PLUS a canonical alias for each so a
    resolved concept name finds the column even when the export uses the raw name
    (e.g. the resolver canonicalizes ``aki`` -> ``kdigo_aki`` but the renal module
    exports the column as ``aki``). Without the canonical alias, AKI-outcome ideas
    silently mis-probe as "absent"."""
    idx: dict[str, str] = {}
    for f in sorted(glob.glob(str(EXPORT / "*.parquet"))):
        try:
            names = pq.read_schema(f).names
        except Exception:
            continue
        if "stay_id" not in names:
            continue
        for c in names:
            idx.setdefault(c, f)  # first file that carries the column
            canon = normalize_concept_name(c)
            if canon and canon != c:
                idx.setdefault(canon, f)  # e.g. 'kdigo_aki' -> the 'aki' file
    return idx


def _present_stays(col_index: dict[str, str], concept: str) -> Optional[set]:
    """stay_ids with >=1 non-null observation of `concept`, or None if absent.

    Resolves the concept to its actual export column, accepting the resolver's
    canonical form (``kdigo_aki``) for a raw export column (``aki``)."""
    f = col_index.get(concept) or col_index.get(normalize_concept_name(concept))
    if f is None:
        return None
    # find the actual column physically present in that file
    names = pq.read_schema(f).names
    col = concept if concept in names else next(
        (c for c in names if normalize_concept_name(c) == normalize_concept_name(concept)),
        None,
    )
    if col is None:
        return None
    df = pd.read_parquet(f, columns=["stay_id", col])
    present = df.loc[df[col].notna(), "stay_id"].dropna().unique()
    return set(present.tolist())


def make_export_feasibility_probe(
    col_index: dict[str, str], denominator: int, all_stays: set
):
    """On-demand joint feasibility over the full export (covers any concept)."""

    cache: dict[str, Optional[set]] = {}

    def _stays(concept: str) -> Optional[set]:
        if concept not in cache:
            cache[concept] = _present_stays(col_index, concept)
        return cache[concept]

    def probe(
        *,
        concepts: Sequence[str],
        database: str,
        data_path: Any,
        cohort: Any = None,
        analytic_unit: str = "stay",
    ) -> Mapping[str, Any]:
        out: dict[str, Any] = {}
        present = {c: _stays(c) for c in concepts}
        # A real joint requires EVERY requested concept to be present. If any is
        # absent, there is no joint cohort -- n_joint MUST be 0, NOT the count
        # over the present subset (which would fake feasibility: e.g. peep ->
        # kdigo_aki reporting peep's 39403 while the outcome was missing). The
        # absent concept is still omitted from `out` so the pair is flagged
        # non-executable downstream.
        any_absent = any(s is None for s in present.values())
        if any_absent:
            n_joint = 0
        else:
            joint = set(all_stays)
            for s in present.values():
                joint &= s
            n_joint = len(joint)
        jf = (n_joint / denominator) if denominator else 0.0
        for c, s in present.items():
            if s is None:
                continue  # absent concept -> omit so the pair is flagged, not faked
            out[c] = {
                "joint_fraction_complete": jf,
                "n_joint_complete": n_joint,
                "denominator_n": denominator,
                "source": "full_export_on_demand",
                "note": f"present_n={len(s)}",
            }
        return out

    return probe


def _make_llm(model: str) -> OpenAIClient:
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL")
    if not key:
        raise SystemExit("set OPENAI_API_KEY (+ OPENAI_BASE_URL for the local proxy)")
    return OpenAIClient(model=model, api_key=key, base_url=base, request_timeout=180.0)


_NOVELTY_JUDGE_SYSTEM = (
    "You are a critical ICU evidence reviewer judging whether a proposed study is "
    "already covered by the prior-art titles supplied. Be conservative: if the "
    "prior work plausibly already answers it, say so. Return only JSON: "
    '{"verdict": "duplicate"|"crowded"|"differentiated", "rationale": "<one sentence>"}.'
)


def make_novelty_judge(llm: OpenAIClient):
    """Phase 3 LLM differentiation judge (veto-net: can only tighten novelty)."""
    from easyicu.research_agent.llm import LLMMessage  # noqa: E402

    def judge(*, idea, executable_candidate, hits, count_label):
        construct = idea.exposure_or_predictor.strip() or ", ".join(
            str(c) for c in idea.analysis_concepts if str(c).strip()
        )
        outcome = idea.outcome.strip() or "(concept-set / no single outcome)"
        titles = [h.get("title", "") for h in hits[:12] if h.get("title")]
        payload = {
            "proposed_study": {
                "analysis_family": idea.analysis_family,
                "construct_or_variables": construct,
                "outcome": outcome,
                "population": idea.population,
            },
            "count_based_label": count_label,
            "prior_art_titles": titles,
            "instruction": (
                "Decide if the proposed study is a duplicate of the prior-art "
                "titles, in a crowded-but-differentiable field, or genuinely "
                "differentiated. Judge only against the titles shown."
            ),
        }
        messages = [
            LLMMessage(role="system", content=_NOVELTY_JUDGE_SYSTEM),
            LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        raw = llm.complete(messages, max_tokens=300, temperature=0.0)
        text = raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw else "{}"
        data = json.loads(text)
        return {
            "verdict": str(data.get("verdict") or "").strip().lower(),
            "rationale": str(data.get("rationale") or "").strip(),
        }

    return judge


# Complementary scope: biomarker / physiologic predictors of ICU mortality.
# Skews toward predictors that map to structured miiv labs/vitals/scores.
BIOMARKER_QUERY = (
    '("Crit Care"[Journal] OR "Intensive Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal] OR "Chest"[Journal] '
    'OR "Shock"[Journal] OR "J Crit Care"[Journal]) '
    "AND (review[Publication Type] OR editorial[Publication Type]) "
    "AND (biomarker[Title/Abstract] OR lactate[Title/Abstract] "
    'OR "blood gas"[Title/Abstract] OR hemodynamic[Title/Abstract] '
    'OR "laboratory"[Title/Abstract] OR electrolyte[Title/Abstract] '
    'OR "organ dysfunction"[Title/Abstract]) '
    "AND (mortality[Title/Abstract] OR prognosis[Title/Abstract] "
    "OR prognostic[Title/Abstract] OR outcome[Title/Abstract]) "
    'AND (ICU[Title/Abstract] OR "intensive care"[Title/Abstract] '
    'OR "critically ill"[Title/Abstract])'
)

# Complementary scope: treatment-effect / outcome HETEROGENEITY -- ICU
# subphenotypes, endotypes, latent classes, and heterogeneity of treatment
# effect. This is the literature seam that motivates clustering / subphenotype
# discovery ideas (Finding 2) rather than single predictor->mortality links.
HETEROGENEITY_QUERY = (
    '("Crit Care"[Journal] OR "Intensive Care Med"[Journal] '
    'OR "Lancet Respir Med"[Journal] OR "Am J Respir Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal]) '
    "AND (review[Publication Type] OR editorial[Publication Type]) "
    "AND (subphenotype[Title/Abstract] OR sub-phenotype[Title/Abstract] "
    "OR phenotype[Title/Abstract] OR endotype[Title/Abstract] "
    'OR "latent class"[Title/Abstract] OR "treatment effect"[Title/Abstract] '
    "OR heterogeneity[Title/Abstract] OR cluster[Title/Abstract]) "
    'AND (ICU[Title/Abstract] OR "intensive care"[Title/Abstract] '
    'OR "critically ill"[Title/Abstract] OR sepsis[Title/Abstract] '
    "OR ARDS[Title/Abstract])"
)

_QUERY_PRESETS = {
    "prognostic": PROGNOSTIC_QUERY,
    "biomarker": BIOMARKER_QUERY,
    "heterogeneity": HETEROGENEITY_QUERY,
}

_DEFAULT_FUNNEL_JOURNAL_PRESET = "critical_care_specialty_wide"

# Run-plan filters for the broad critical-care funnel. They are deliberately
# scoped to this runner (not global prompts): the goal is to keep specialty
# journal recall broad while biasing retrieved ideas toward adult ICU variables
# and outcomes that a structured EHR export can actually test.
_ADULT_ICU_FILTER_TERMS = (
    "("
    "adult[Title/Abstract] OR adults[Title/Abstract] "
    'OR "critically ill"[Title/Abstract] '
    'OR "critical illness"[Title/Abstract] '
    'OR ICU[Title/Abstract] OR "intensive care"[Title/Abstract] '
    "OR sepsis[Title/Abstract] OR ARDS[Title/Abstract] "
    'OR "mechanical ventilation"[Title/Abstract] '
    "OR shock[Title/Abstract] OR vasopressor[Title/Abstract] "
    "OR delirium[Title/Abstract] "
    'OR "acute kidney injury"[Title/Abstract]'
    ") NOT ("
    "neonatal[Title/Abstract] OR neonate[Title/Abstract] "
    "OR newborn[Title/Abstract] OR preterm[Title/Abstract] "
    "OR infant[Title/Abstract] OR infants[Title/Abstract] "
    "OR pediatric[Title/Abstract] OR paediatric[Title/Abstract] "
    "OR child[Title/Abstract] OR children[Title/Abstract] "
    "OR adolescent[Title/Abstract] OR adolescents[Title/Abstract] "
    'OR "bronchopulmonary dysplasia"[Title/Abstract]'
    ")"
)
_EHR_ACTIONABLE_FILTER_TERMS = (
    "("
    "mortality[Title/Abstract] OR death[Title/Abstract] "
    'OR "length of stay"[Title/Abstract] '
    'OR "mechanical ventilation"[Title/Abstract] '
    "OR ventilation[Title/Abstract] OR vasopressor[Title/Abstract] "
    "OR norepinephrine[Title/Abstract] "
    'OR "mean arterial pressure"[Title/Abstract] OR MAP[Title/Abstract] '
    "OR lactate[Title/Abstract] OR oxygenation[Title/Abstract] "
    'OR "PaO2/FiO2"[Title/Abstract] OR "SpO2/FiO2"[Title/Abstract] '
    "OR SOFA[Title/Abstract] OR GCS[Title/Abstract] "
    "OR creatinine[Title/Abstract] OR bilirubin[Title/Abstract] "
    "OR platelet[Title/Abstract] OR platelets[Title/Abstract] "
    'OR "urine output"[Title/Abstract] '
    'OR "renal replacement therapy"[Title/Abstract] '
    "OR missingness[Title/Abstract] "
    'OR "measurement frequency"[Title/Abstract] '
    'OR "vital signs"[Title/Abstract] '
    "OR laboratory[Title/Abstract]"
    ") NOT ("
    'mRS[Title/Abstract] OR "modified Rankin"[Title/Abstract] '
    'OR "global longitudinal strain"[Title/Abstract] '
    "OR transcriptomic[Title/Abstract] OR genomic[Title/Abstract] "
    "OR proteomic[Title/Abstract] OR metabolomic[Title/Abstract] "
    "OR microbiome[Title/Abstract] OR cytokine[Title/Abstract] "
    'OR "quality of life"[Title/Abstract] '
    'OR "functional outcome"[Title/Abstract]'
    ")"
)

_FUNNEL_TOPIC_PRESETS = {
    "criticalcare": [
        "critical illness",
        "intensive care",
        "sepsis",
        "ARDS",
        "mechanical ventilation",
        "shock",
        "vasopressor",
        "delirium",
        "acute kidney injury",
        "organ support",
    ],
    "prognostic": [
        "mortality",
        "prognosis",
        "risk factor",
        "predictor",
        "critical illness",
    ],
    "biomarker": [
        "biomarker",
        "lactate",
        "blood gas",
        "laboratory",
        "organ dysfunction",
        "critical illness",
    ],
    "heterogeneity": [
        "subphenotype",
        "phenotype",
        "endotype",
        "treatment effect heterogeneity",
        "sepsis",
        "ARDS",
    ],
}


class HarnessPubMedSearchClient:
    """Adapter from the validation harness functions to funnel search()."""

    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []

    def search(self, query: str, *, retmax: int = 20):
        search = _pubmed_search_robust(query, retmax=retmax)
        articles = H.pubmed_fetch(search["pmids"])
        self.queries.append(
            {"query": query, "retmax": retmax, "count": search.get("count", 0)}
        )
        out = []
        for article in articles:
            from easyicu.research_agent.literature import CitationRecord  # noqa: E402

            out.append(
                CitationRecord(
                    key=f"pubmed_{article['pmid']}",
                    title=article["title"],
                    year=article["year"],
                    venue=article["journal"],
                    relevance=article["abstract"],
                    url=article["url"],
                    pmid=article["pmid"],
                )
            )
        return out


def _pubmed_search_robust(query: str, *, retmax: int) -> dict[str, Any]:
    if len(query) >= 1800:
        return _pubmed_search_post(query, retmax=retmax)
    try:
        return H.pubmed_search(query, retmax=retmax)
    except RuntimeError as exc:
        # Broad journal presets plus adult/EHR filters can exceed reliable GET
        # URL behavior in E-utilities. Retry the same ESearch request as POST
        # before surfacing the network error.
        if len(query) < 1800 and "redirect" not in str(exc).lower():
            raise
        return _pubmed_search_post(query, retmax=retmax)


def _pubmed_search_post(query: str, *, retmax: int) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{H.EUTILS}/esearch.fcgi",
        data=params,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "EasyICU-discovery/1.0",
        },
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            result = raw.get("esearchresult") or {}
            return {
                "count": int(result.get("count", 0) or 0),
                "pmids": [str(item) for item in result.get("idlist", [])],
            }
        except (HTTPError, URLError, TimeoutError) as exc:
            last = exc
            time.sleep(min(2.0 * (attempt + 1), 20.0))
    raise RuntimeError("PubMed POST JSON request failed") from last


class AbstractGapTextClient:
    """Expose PubMed abstract text to the funnel gap-excerpt extractor."""

    def fetch_gap_text(self, citation, *, route):
        return " ".join(
            part
            for part in [citation.title, citation.relevance or ""]
            if str(part or "").strip()
        )


def _fetch_materials(query: str, retmax: int):
    search = H.pubmed_search(query, retmax=retmax)
    articles = H.pubmed_fetch(search["pmids"])
    return H.build_materials(articles)


def _funnel_scope_for_preset(
    scope: str,
    *,
    topic_terms: Sequence[str] = (),
    last_n_years: int = 8,
    journal_preset: str = _DEFAULT_FUNNEL_JOURNAL_PRESET,
    journals: Sequence[str] = (),
    extra_terms: Optional[str] = None,
) -> LiteratureScopeSpec:
    topics = list(topic_terms) or list(_FUNNEL_TOPIC_PRESETS[scope])
    return LiteratureScopeSpec(
        journals=list(journals),
        journal_preset=journal_preset,
        pub_types=[],
        last_n_years=last_n_years,
        topic_terms=topics,
        extra_terms=extra_terms,
    )


def _default_filter_enabled(scope: str, requested: Optional[bool]) -> bool:
    if requested is not None:
        return bool(requested)
    return scope == "criticalcare"


def _combined_filter_terms(args) -> tuple[Optional[str], dict[str, Any]]:
    adult_icu = _default_filter_enabled(args.scope, args.funnel_adult_icu_filter)
    ehr_actionable = _default_filter_enabled(
        args.scope, args.funnel_ehr_actionable_filter
    )
    fragments: list[str] = []
    if adult_icu:
        fragments.append(_ADULT_ICU_FILTER_TERMS)
    if ehr_actionable:
        fragments.append(_EHR_ACTIONABLE_FILTER_TERMS)
    combined = " AND ".join(f"({fragment})" for fragment in fragments) or None
    return combined, {
        "adult_icu": adult_icu,
        "ehr_actionable": ehr_actionable,
        "extra_terms": combined,
    }


def _fetch_funnel_materials(args, out_dir: Path):
    filter_terms, filter_report = _combined_filter_terms(args)
    base_scope = _funnel_scope_for_preset(
        args.scope,
        topic_terms=args.funnel_topic_terms,
        last_n_years=args.funnel_last_years,
        journal_preset=args.funnel_journal_preset,
        journals=args.funnel_journal,
        extra_terms=filter_terms,
    )
    spec = LiteratureFunnelSpec(
        base_scope=base_scope,
        platform_gap_terms=list(args.funnel_platform_terms),
        max_gap_excerpt_chars=args.funnel_max_excerpt_chars,
    )
    search_client = HarnessPubMedSearchClient()
    result = fetch_literature_funnel_corpus(
        spec,
        search_client,
        text_client=AbstractGapTextClient(),
        reference_year=args.funnel_reference_year,
        retmax_per_route=args.funnel_retmax_per_route,
    )
    report = {
        "mode": "literature_funnel",
        "reference_year": args.funnel_reference_year,
        "retmax_per_route": args.funnel_retmax_per_route,
        "journal_preset": args.funnel_journal_preset,
        "journals": resolve_journals(base_scope),
        "filters": filter_report,
        "routes": [route.model_dump(mode="json") for route in result.query_routes],
        "source_searches": search_client.queries,
        "materials": {
            "n": len(result.materials),
            "by_adapter_level": dict(
                Counter(material.source_adapter_level for material in result.materials)
            ),
            "by_discovery_route": dict(
                Counter(
                    material.discovery_route or "unknown"
                    for material in result.materials
                )
            ),
            "by_source_text_role": dict(
                Counter(
                    material.source_text_role or "unknown"
                    for material in result.materials
                )
            ),
        },
    }
    (out_dir / "funnel_query_routes.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result.materials, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt5.4")
    ap.add_argument("--article-count", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--top-n-screen", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--scope",
        choices=sorted(_FUNNEL_TOPIC_PRESETS),
        default="prognostic",
        help="literature scope preset",
    )
    ap.add_argument(
        "--pubmed-query", default=None, help="override the scope query entirely"
    )
    ap.add_argument(
        "--use-funnel",
        action="store_true",
        help="use the auditable multi-route LiteratureFunnelSpec corpus instead of the legacy single query",
    )
    ap.add_argument("--funnel-retmax-per-route", type=int, default=10)
    ap.add_argument("--funnel-reference-year", type=int, default=2026)
    ap.add_argument("--funnel-last-years", type=int, default=8)
    ap.add_argument("--funnel-max-excerpt-chars", type=int, default=1200)
    ap.add_argument(
        "--funnel-journal-preset",
        choices=sorted(JOURNAL_PRESETS),
        default=_DEFAULT_FUNNEL_JOURNAL_PRESET,
        help="critical-care journal preset used by --use-funnel",
    )
    ap.add_argument(
        "--funnel-journal",
        action="append",
        default=[],
        help="extra PubMed journal title abbreviation for --use-funnel; may be repeated",
    )
    ap.add_argument(
        "--funnel-topic-terms",
        action="append",
        default=[],
        help="override funnel topic terms; may be repeated",
    )
    ap.add_argument(
        "--funnel-platform-terms",
        action="append",
        default=[],
        help="extra platform-gap terms; may be repeated",
    )
    ap.add_argument(
        "--funnel-adult-icu-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "append adult ICU inclusion/exclusion terms to every funnel route; "
            "defaults on for --scope criticalcare"
        ),
    )
    ap.add_argument(
        "--funnel-ehr-actionable-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "append structured-EHR-variable/outcome terms to every funnel route; "
            "defaults on for --scope criticalcare"
        ),
    )
    ap.add_argument(
        "--reflection-rounds",
        type=int,
        default=0,
        help="Phase 2: self-critique/refine rounds over extracted ideas (0=off)",
    )
    ap.add_argument(
        "--reflection-search",
        action="store_true",
        help="Phase 2b: retrieval-augmented reflection (inject prior-art titles)",
    )
    ap.add_argument(
        "--novelty-judge",
        action="store_true",
        help="Phase 3: LLM-reasoned novelty veto-net (can only tighten labels)",
    )
    ap.add_argument(
        "--novelty-optimize",
        type=int,
        default=0,
        metavar="ROUNDS",
        help=(
            "Gap A: SciMON-style novelty optimisation rounds -- measure a "
            "crowded idea against PubMed, revise toward a differentiated angle, "
            "re-measure, keep only if novelty improved (0=off)"
        ),
    )
    ap.add_argument(
        "--novelty-optimize-min-hits",
        type=int,
        default=5,
        help="Gap A: minimum exact prior-art hits before an idea is revised",
    )
    ap.add_argument(
        "--validate-candidates",
        action="store_true",
        help=(
            "Gap B: ResearchAgent-style multi-criteria validator panel "
            "(clarity/novelty/feasibility_fit/impact, advisory only)"
        ),
    )
    args = ap.parse_args()
    if args.use_funnel and args.pubmed_query:
        raise SystemExit("--use-funnel and --pubmed-query are mutually exclusive")
    if not args.use_funnel and args.scope not in _QUERY_PRESETS:
        raise SystemExit(f"--scope {args.scope!r} requires --use-funnel")
    query = args.pubmed_query or _QUERY_PRESETS.get(args.scope)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir or f"research_output/discovery_gpt54/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[disc] indexing export columns + denominator...")
    col_index = _build_column_index()
    if not col_index:
        raise SystemExit(f"no usable parquet (stay_id) under {EXPORT}")
    # Partial exports are minable: don't require the outcome module. Use `death`
    # for the stay universe when present (so mortality outcomes line up), else
    # fall back to the widest stay_id set among whatever modules ARE exported
    # (demographics first, then any module). Mortality ideas against a partial
    # export without `death` are correctly flagged not-available, not crashed.
    denom_file = col_index.get("death") or col_index.get("age")
    if denom_file is None:
        denom_file = next(iter(dict.fromkeys(col_index.values())))
    if "death" not in col_index:
        print("[disc] NOTE: no `death`/outcome module in this export — mining "
              "against partial export; mortality endpoints will be unavailable.")
    all_stays = set(
        pd.read_parquet(denom_file, columns=["stay_id"])["stay_id"]
        .dropna()
        .unique()
        .tolist()
    )
    denominator = len(all_stays)
    print(f"[disc] {len(col_index)} columns indexed; denominator={denominator} "
          f"stays (from {Path(denom_file).name})")

    catalog = build_available_catalog(EXPORT)
    available = list(catalog.ids())
    print(f"[disc] available concepts (full catalog): {len(available)}")

    if args.use_funnel:
        filter_terms, filter_report = _combined_filter_terms(args)
        print(
            f"[disc] scope={args.scope} fetching funnel corpus "
            f"(retmax_per_route={args.funnel_retmax_per_route})..."
        )
        if filter_terms:
            print(
                "[disc] funnel filters: "
                f"adult_icu={filter_report['adult_icu']} "
                f"ehr_actionable={filter_report['ehr_actionable']}"
            )
        materials, source_report = _fetch_funnel_materials(args, out_dir)
        print(f"[disc] materials: {len(materials)} via literature funnel")
        print(f"[disc] source mix: {source_report['materials']}")
    else:
        print(
            f"[disc] scope={args.scope} fetching <= {args.article_count} ICU reviews from PubMed..."
        )
        materials = _fetch_materials(query, args.article_count)
        print(f"[disc] materials: {len(materials)} abstracts")
        source_report = {
            "mode": "legacy_single_query",
            "query": query,
            "retmax": args.article_count,
            "materials": {"n": len(materials)},
        }
        (out_dir / "source_query.json").write_text(
            json.dumps(source_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if not materials:
        raise SystemExit("no materials retrieved; check network / query")

    extraction_llm = _make_llm(args.model)
    screener = H.OpenRouterSameTopicScreener(
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.model,
        base_url=os.environ.get("OPENAI_BASE_URL", H.OPENROUTER_BASE_URL),
        screen_cache_dir=out_dir / "screen_cache",
    )
    search_client = H.PubMedPriorArtScreenClient(
        screener=screener,
        cache_dir=out_dir / "prior_art_cache",
        top_n_screen=args.top_n_screen,
    )
    probe = make_export_feasibility_probe(col_index, denominator, all_stays)

    # Outcome determinability: use the SHARED concept catalog (127 concepts
    # scored -- binary known_0_1, continuous/ordinal non_binary_determinable,
    # organ-support interventions) rather than a mortality-only hand-roll. The
    # mortality-only dict was the dominant cause of false "db-cannot-do": every
    # non-mortality outcome that resolved to a real present concept (rrt,
    # kdigo_aki, lact, fluid_balance, sofa2 ...) fell through to determinability
    # "unknown" and was buried as a database failure. The free-text mortality
    # aliases are still merged on top so the generic-umbrella path can normalize
    # a bare "mortality"/"survival" label onto `death`.
    determinability: dict = dict(load_concept_catalog().outcome_determinability)
    determinability.update(
        {
            "death": OutcomeDeterminability(outcome="death", status="known_0_1"),
            "mortality": OutcomeDeterminability(
                outcome="mortality",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
            "in-hospital mortality": OutcomeDeterminability(
                outcome="in-hospital mortality",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
            "icu mortality": OutcomeDeterminability(
                outcome="icu mortality",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
            "death in hospital": OutcomeDeterminability(
                outcome="death in hospital",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
        }
    )

    # Extended feasibility: reconsider db-cannot-do via ICD-derivable cohort
    # (Case 1) and dictionary / cross-DB concept reachability (Case 2). Only
    # downgrades to hold (human-confirm); never promotes to executable.
    from easyicu.research_agent.idea_mining_extended_feasibility import (
        ExtendedFeasibilityIndex,
    )

    extended_index = ExtendedFeasibilityIndex.build(
        current_db="miiv", available_concepts=available
    )
    print(
        f"[disc] idea-mining dry run (model={args.model}, full catalog, on-demand feasibility)..."
    )
    result = run_idea_mining_dry_run(
        materials=materials,
        llm=extraction_llm,
        available_concepts=available,
        outcome_determinability=determinability,
        output_dir=out_dir / "dry_run",
        database="miiv",
        data_path=EXPORT,  # informational; the probe reads the export itself
        feasibility_probe=probe,
        analytic_unit="stay",
        top_k=args.top_k,
        prior_art_search_client=search_client,
        prior_art_searched_at=datetime.now(timezone.utc).isoformat(),
        prior_art_top_n=args.top_n_screen,
        untraceable_quote_policy="skip",
        reflection_rounds=args.reflection_rounds,
        reflection_search_client=search_client if args.reflection_search else None,
        novelty_judge=(
            make_novelty_judge(extraction_llm) if args.novelty_judge else None
        ),
        novelty_optimize_rounds=args.novelty_optimize,
        novelty_optimize_min_hits=args.novelty_optimize_min_hits,
        validate_candidates=args.validate_candidates,
        extended_feasibility_index=extended_index,
        cross_db_targets=["miiv", "mimic", "eicu", "aumc", "hirid", "sic"],
    )

    payload = result.model_dump(mode="json")
    yr = payload["yield_report"]
    novelty = Counter(
        r["prior_art"]["novelty_label"] for r in payload["discovery_records"]
    )
    gng = Counter(r["go_no_go"] for r in payload["discovery_records"])
    print("\n========== DISCOVERY YIELD ==========")
    print(f"  literature ideas : {yr['n_literature_ideas']}")
    print(
        f"  resolved pred/out: {yr['n_resolved_predictor']} / {yr['n_resolved_outcome']}"
    )
    print(f"  executable       : {yr['n_executable']}")
    print(f"  novelty          : {dict(novelty)}")
    print(f"  go_no_go         : {dict(gng)}")

    print("\n========== CANDIDATES (executable first) ==========")
    recs = sorted(
        payload["discovery_records"],
        key=lambda r: (
            r.get("go_no_go") != "recommend",
            r.get("feasibility_route") or "",
            r.get("candidate_topic") or "",
        ),
    )
    for i, r in enumerate(recs[: args.top_k], 1):
        feas = r.get("database_feasibility") or {}
        pa = r.get("prior_art") or {}
        print(
            f"\n[{i}] go_no_go={r.get('go_no_go')}  "
            f"route={r.get('feasibility_route')}  "
            f"novelty={pa.get('novelty_label')}  "
            f"screen={pa.get('same_topic_screen_status')}"
        )
        print(f"    topic     : {r.get('candidate_topic')}")
        print(
            f"    resolved  : {feas.get('resolved_predictor_concept')} -> "
            f"{feas.get('resolved_outcome_concept')}"
        )
        print(
            f"    joint n   : {feas.get('n_joint_complete')}/"
            f"{feas.get('denominator_n')} ({feas.get('coverage_source')})"
        )
        print(f"    evidence  : {pa.get('evidence_map_counts') or {}}")
        print(f"    next      : {r.get('feasibility_next_action')}")
        print(f"    source    : {r.get('literature_source')}")
        print(f"    quote     : {(r.get('gap_evidence_quote') or '')[:240]}")

    novelty_opt = payload.get("novelty_optimization") or []
    revised = [e for e in novelty_opt if e.get("revised")]
    if novelty_opt:
        print("\n========== NOVELTY OPTIMIZATION (Gap A) ==========")
        print(f"  ideas measured   : {len(novelty_opt)}")
        print(f"  ideas revised    : {len(revised)}")
        for e in revised[:10]:
            print(
                f"    - {e.get('citation_key')}: "
                f"{e.get('initial_exact_hits')} -> {e.get('final_exact_hits')} hits | "
                f"{e.get('initial_construct')}  =>  {e.get('final_construct')}"
            )

    validation = payload.get("candidate_validation") or []
    if validation:
        print("\n========== MULTI-CRITERIA VALIDATION (Gap B) ==========")
        for v in validation[: args.top_k]:
            print(
                f"    [{v.get('go_no_go')}] clarity={v.get('clarity')} "
                f"novelty={v.get('novelty')} feasibility_fit={v.get('feasibility_fit')} "
                f"impact={v.get('impact')} :: {str(v.get('candidate_topic'))[:80]}"
            )

    (out_dir / "discovery_console_summary.json").write_text(
        json.dumps(
            {
                "yield": yr,
                "novelty": dict(novelty),
                "go_no_go": dict(gng),
                "n_executable": yr["n_executable"],
                "novelty_optimization": novelty_opt,
                "candidate_validation": validation,
                "out_dir": str(out_dir),
                "source_report": source_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[disc] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
