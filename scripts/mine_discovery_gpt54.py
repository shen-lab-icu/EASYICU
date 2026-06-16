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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from easyicu.research_agent.concept_catalog import load_concept_catalog  # noqa: E402
from easyicu.research_agent.data_catalog import build_available_catalog  # noqa: E402
from easyicu.research_agent.idea_mining import OutcomeDeterminability, run_idea_mining_dry_run  # noqa: E402
from easyicu.research_agent.llm import OpenAIClient  # noqa: E402

import run_idea_mining_s6_validation_harness as H  # noqa: E402

EXPORT = Path("/Volumes/外置硬盘/easyicu_fullexport_miiv_20260610")

# Prognostic / risk-factor ICU review scope -> predictor->mortality ideas.
PROGNOSTIC_QUERY = (
    '("Crit Care"[Journal] OR "Intensive Care Med"[Journal] '
    'OR "Lancet Respir Med"[Journal] OR "Am J Respir Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal]) '
    'AND (review[Publication Type] OR editorial[Publication Type]) '
    'AND (mortality[Title/Abstract] OR prognosis[Title/Abstract] '
    'OR "risk factor"[Title/Abstract] OR "risk factors"[Title/Abstract] '
    'OR outcome[Title/Abstract] OR predictor[Title/Abstract] '
    'OR prognostic[Title/Abstract]) '
    'AND (ICU[Title/Abstract] OR "intensive care"[Title/Abstract] '
    'OR "critically ill"[Title/Abstract])'
)


def _build_column_index() -> dict[str, str]:
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
    return idx


def _present_stays(col_index: dict[str, str], concept: str) -> Optional[set]:
    """stay_ids with >=1 non-null observation of `concept`, or None if absent."""
    f = col_index.get(concept)
    if f is None:
        return None
    df = pd.read_parquet(f, columns=["stay_id", concept])
    present = df.loc[df[concept].notna(), "stay_id"].dropna().unique()
    return set(present.tolist())


def make_export_feasibility_probe(col_index: dict[str, str], denominator: int, all_stays: set):
    """On-demand joint feasibility over the full export (covers any concept)."""

    cache: dict[str, Optional[set]] = {}

    def _stays(concept: str) -> Optional[set]:
        if concept not in cache:
            cache[concept] = _present_stays(col_index, concept)
        return cache[concept]

    def probe(*, concepts: Sequence[str], database: str, data_path: Any,
              cohort: Any = None, analytic_unit: str = "stay") -> Mapping[str, Any]:
        out: dict[str, Any] = {}
        present = {c: _stays(c) for c in concepts}
        # joint = stays present for EVERY requested concept that exists in data
        usable = [s for s in present.values() if s is not None]
        joint = set(all_stays)
        for s in usable:
            joint &= s
        n_joint = len(joint) if usable else 0
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
    'AND (review[Publication Type] OR editorial[Publication Type]) '
    'AND (biomarker[Title/Abstract] OR lactate[Title/Abstract] '
    'OR "blood gas"[Title/Abstract] OR hemodynamic[Title/Abstract] '
    'OR "laboratory"[Title/Abstract] OR electrolyte[Title/Abstract] '
    'OR "organ dysfunction"[Title/Abstract]) '
    'AND (mortality[Title/Abstract] OR prognosis[Title/Abstract] '
    'OR prognostic[Title/Abstract] OR outcome[Title/Abstract]) '
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
    'AND (subphenotype[Title/Abstract] OR sub-phenotype[Title/Abstract] '
    'OR phenotype[Title/Abstract] OR endotype[Title/Abstract] '
    'OR "latent class"[Title/Abstract] OR "treatment effect"[Title/Abstract] '
    'OR heterogeneity[Title/Abstract] OR cluster[Title/Abstract]) '
    'AND (ICU[Title/Abstract] OR "intensive care"[Title/Abstract] '
    'OR "critically ill"[Title/Abstract] OR sepsis[Title/Abstract] '
    'OR ARDS[Title/Abstract])'
)

_QUERY_PRESETS = {
    "prognostic": PROGNOSTIC_QUERY,
    "biomarker": BIOMARKER_QUERY,
    "heterogeneity": HETEROGENEITY_QUERY,
}


def _fetch_materials(query: str, retmax: int):
    search = H.pubmed_search(query, retmax=retmax)
    articles = H.pubmed_fetch(search["pmids"])
    return H.build_materials(articles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt5.4")
    ap.add_argument("--article-count", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--top-n-screen", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--scope", choices=sorted(_QUERY_PRESETS), default="prognostic",
                    help="literature scope preset")
    ap.add_argument("--pubmed-query", default=None, help="override the scope query entirely")
    ap.add_argument("--reflection-rounds", type=int, default=0,
                    help="Phase 2: self-critique/refine rounds over extracted ideas (0=off)")
    ap.add_argument("--reflection-search", action="store_true",
                    help="Phase 2b: retrieval-augmented reflection (inject prior-art titles)")
    ap.add_argument("--novelty-judge", action="store_true",
                    help="Phase 3: LLM-reasoned novelty veto-net (can only tighten labels)")
    args = ap.parse_args()
    query = args.pubmed_query or _QUERY_PRESETS[args.scope]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir or f"research_output/discovery_gpt54/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[disc] indexing full-export columns + denominator...")
    col_index = _build_column_index()
    death_file = col_index.get("death")
    if death_file is None:
        raise SystemExit("no `death` column in export")
    all_stays = set(
        pd.read_parquet(death_file, columns=["stay_id"])["stay_id"].dropna().unique().tolist()
    )
    denominator = len(all_stays)
    print(f"[disc] {len(col_index)} columns indexed; denominator={denominator} stays")

    catalog = build_available_catalog(EXPORT)
    available = list(catalog.ids())
    print(f"[disc] available concepts (full catalog): {len(available)}")

    print(f"[disc] scope={args.scope} fetching <= {args.article_count} ICU reviews from PubMed...")
    materials = _fetch_materials(query, args.article_count)
    print(f"[disc] materials: {len(materials)} abstracts")
    if not materials:
        raise SystemExit("no materials retrieved; check network / query")

    extraction_llm = _make_llm(args.model)
    screener = H.OpenRouterSameTopicScreener(
        api_key=os.environ["OPENAI_API_KEY"], model=args.model,
        base_url=os.environ.get("OPENAI_BASE_URL", H.OPENROUTER_BASE_URL),
        screen_cache_dir=out_dir / "screen_cache",
    )
    search_client = H.PubMedPriorArtScreenClient(
        screener=screener, cache_dir=out_dir / "prior_art_cache", top_n_screen=args.top_n_screen
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
            "mortality": OutcomeDeterminability(outcome="mortality", status="known_0_1",
                                                normalized_outcome_concept="death"),
            "in-hospital mortality": OutcomeDeterminability(outcome="in-hospital mortality",
                                                            status="known_0_1", normalized_outcome_concept="death"),
            "icu mortality": OutcomeDeterminability(outcome="icu mortality", status="known_0_1",
                                                    normalized_outcome_concept="death"),
            "death in hospital": OutcomeDeterminability(outcome="death in hospital", status="known_0_1",
                                                        normalized_outcome_concept="death"),
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
    print(f"[disc] idea-mining dry run (model={args.model}, full catalog, on-demand feasibility)...")
    result = run_idea_mining_dry_run(
        materials=materials,
        llm=extraction_llm,
        available_concepts=available,
        outcome_determinability=determinability,
        output_dir=out_dir / "dry_run",
        database="miiv",
        data_path=EXPORT,            # informational; the probe reads the export itself
        feasibility_probe=probe,
        analytic_unit="stay",
        top_k=args.top_k,
        prior_art_search_client=search_client,
        prior_art_searched_at=datetime.now(timezone.utc).isoformat(),
        prior_art_top_n=args.top_n_screen,
        untraceable_quote_policy="skip",
        reflection_rounds=args.reflection_rounds,
        reflection_search_client=search_client if args.reflection_search else None,
        novelty_judge=make_novelty_judge(extraction_llm) if args.novelty_judge else None,
        extended_feasibility_index=extended_index,
    )

    payload = result.model_dump(mode="json")
    yr = payload["yield_report"]
    novelty = Counter(r["prior_art"]["novelty_label"] for r in payload["discovery_records"])
    gng = Counter(r["go_no_go"] for r in payload["discovery_records"])
    print("\n========== DISCOVERY YIELD ==========")
    print(f"  literature ideas : {yr['n_literature_ideas']}")
    print(f"  resolved pred/out: {yr['n_resolved_predictor']} / {yr['n_resolved_outcome']}")
    print(f"  executable       : {yr['n_executable']}")
    print(f"  novelty          : {dict(novelty)}")
    print(f"  go_no_go         : {dict(gng)}")

    print("\n========== CANDIDATES (executable first) ==========")
    recs = sorted(payload["discovery_records"],
                  key=lambda r: (r.get("go_no_go") != "go", -(((r.get("feasibility") or {}).get("joint_fraction_complete")) or 0)))
    for i, r in enumerate(recs[: args.top_k], 1):
        cand = r.get("candidate", r); feas = r.get("feasibility") or {}; pa = r.get("prior_art") or {}
        print(f"\n[{i}] go_no_go={r.get('go_no_go')}  novelty={pa.get('novelty_label')}  screen={pa.get('same_topic_screen_status')}")
        print(f"    predictor : {cand.get('predictor_label')} -> {cand.get('resolved_predictor_concept')}")
        print(f"    outcome   : {cand.get('outcome_label')} -> {cand.get('resolved_outcome_concept')}")
        print(f"    family    : {cand.get('analysis_family')}  window={cand.get('time_window_hint')}")
        print(f"    joint_complete: {feas.get('joint_fraction_complete')}  (n={feas.get('n_joint_complete')}/{feas.get('denominator_n')})")
        print(f"    source PMID: {cand.get('citation_key')}")
        print(f"    rationale : {(cand.get('rationale') or '')[:240]}")

    (out_dir / "discovery_console_summary.json").write_text(
        json.dumps({"yield": yr, "novelty": dict(novelty), "go_no_go": dict(gng),
                    "n_executable": yr["n_executable"], "out_dir": str(out_dir)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[disc] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
