#!/usr/bin/env python3
"""Real idea-mining discovery run (NOT a plumbing smoke).

Unlike ``run_idea_mining_s6_validation_harness.py`` — which uses the rule-based
``DeterministicGapExtractor`` — this driver wires the *real* proxy LLM
(gpt5.4) into ``run_idea_mining_dry_run`` so the discovered topics come from an
actual model reading real PubMed review/editorial abstracts, gated by the
source-quote traceability check. DB feasibility is probed against a real
MIMIC-IV wide cohort derived from ``universe_m3.parquet``.

Output: a discovery_report.md + candidate_triage_report.json under --out-dir.
Everything stops at the human gate (candidates are ``proposed``, never
``accepted``). API keys come from the environment; none are written to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT), str(REPO_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_idea_mining_s6_validation_harness as harness  # noqa: E402
from run_research_agent_bench import _make_llm  # noqa: E402

from easyicu.research_agent.concept_availability import (  # noqa: E402
    real_data_concept_feasibility,
)
from easyicu.research_agent.concept_catalog import load_concept_catalog  # noqa: E402

# Sparse binary EVENT concepts whose absence means "no event" (= 0), not
# "missing": when probed as outcomes their column covers the whole cohort.
EVENT_DEFAULT_FALSE = (
    "rrt",
    "mech_vent",
    "ecmo",
    "death",
    "circ_event",
    "circ_failure",
    "aki",
    "sep3_sofa2",
    "susp_inf",
)


def _event_aware_probe(**kwargs):
    """real_data_concept_feasibility with event-concept absence treated as 0."""
    return real_data_concept_feasibility(
        event_default_false_concepts=EVENT_DEFAULT_FALSE, **kwargs
    )


from easyicu.research_agent.discovery.idea_mining import (  # noqa: E402
    run_idea_mining_dry_run,
)
from easyicu.research_agent.discovery.idea_mining_feasibility_tier import (  # noqa: E402
    SourceItemIndex,
)
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole  # noqa: E402

DEFAULT_SOURCE_CATALOG = REPO_ROOT / "benchmark" / "source_item_catalog_miiv.json"

DEFAULT_FULL_EXPORT = Path("/Volumes/外置硬盘/easyicu_fullexport_miiv_20260610")

# One representative aggregate per universe_m3 concept. For data-layer
# feasibility (non-missing counts) the aggregate choice is immaterial — an
# aggregate is NaN iff the concept was never measured — so we pick the
# clinically conventional summary and rename to the BARE concept name the
# feasibility probe resolves against.
CONCEPT_SPECS: list[tuple[str, str, VariableRole]] = [
    ("lact", "lact_max", VariableRole.LAB),
    ("crea", "crea_max", VariableRole.LAB),
    ("bun", "bun_max", VariableRole.LAB),
    ("bili", "bili_max", VariableRole.LAB),
    ("alb", "alb_min", VariableRole.LAB),
    ("na", "na_max", VariableRole.LAB),
    ("glu", "glu_max", VariableRole.LAB),
    ("ph", "ph_min", VariableRole.LAB),
    ("plt", "plt_min", VariableRole.LAB),
    ("wbc", "wbc_max", VariableRole.LAB),
    ("inr_pt", "inr_pt_max", VariableRole.LAB),
    ("bicar", "bicar_min", VariableRole.LAB),
    ("map", "map_mean", VariableRole.VITAL),
    ("hr", "hr_mean", VariableRole.VITAL),
    ("resp", "resp_mean", VariableRole.VITAL),
    ("spo2", "spo2_mean", VariableRole.VITAL),
    ("temp", "temp_mean", VariableRole.VITAL),
    ("susp_inf", "susp_inf_max", VariableRole.INTERVENTION),
]


def build_full_wide_cohort(export_dir: Path, output_path: Path) -> dict:
    """Collapse the full module-grouped MIIV export into ONE wide cohort with
    bare concept-named columns (one aggregate per concept per stay), using
    duckdb so the ~98M long-format rows aggregate with low memory.

    The export column names already ARE the EasyICU concept keys (alb, crea,
    hr, sofa2, sep3, ...), so the feasibility probe resolves them directly.
    This exposes ~150 concepts instead of a single cohort's 18.
    """
    import duckdb
    import pyarrow.parquet as pq

    files = sorted(export_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under export dir: {export_dir}")

    con = duckdb.connect()
    # Base cohort: every stay_id seen anywhere (anchor on demographics/outcome).
    base_files = [
        f for f in files if "demographic" in f.name or "outcome" in f.name
    ] or files
    union = " UNION ".join(
        f"SELECT stay_id FROM read_parquet('{f.as_posix()}')" for f in base_files
    )
    con.execute(
        f"CREATE TABLE wide AS SELECT DISTINCT stay_id FROM ({union}) WHERE stay_id IS NOT NULL"
    )

    skip_cols = {"stay_id", "charttime", "starttime", "endtime", "time", "los_hosp"}
    concepts: list[str] = []
    for f in files:
        names = list(pq.read_schema(f).names)
        value_cols = [c for c in names if c not in skip_cols]
        if not value_cols:
            continue
        # aggregate each concept column to one value per stay (max = non-null iff
        # ever measured; choice is immaterial for data-layer feasibility)
        agg = ", ".join(f'max("{c}") AS "{c}"' for c in value_cols)
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE _m AS "
            f"SELECT stay_id, {agg} FROM read_parquet('{f.as_posix()}') "
            f"WHERE stay_id IS NOT NULL GROUP BY stay_id"
        )
        # join, de-duplicating any concept already present
        new_cols = [c for c in value_cols if c not in concepts]
        if not new_cols:
            continue
        sel = ", ".join(f'_m."{c}"' for c in new_cols)
        con.execute(
            f"CREATE OR REPLACE TABLE wide AS "
            f"SELECT wide.*, {sel} FROM wide LEFT JOIN _m USING (stay_id)"
        )
        concepts.extend(new_cols)

    if "death" in concepts:
        con.execute("UPDATE wide SET death = COALESCE(death, 0)")
    con.execute(f"COPY wide TO '{output_path.as_posix()}' (FORMAT parquet)")
    n = con.execute("SELECT count(*) FROM wide").fetchone()[0]
    nonmiss = {}
    for c in concepts:
        frac = con.execute(f'SELECT count("{c}")*1.0/count(*) FROM wide').fetchone()[0]
        nonmiss[c] = round(float(frac), 4)
    con.close()
    return {"n_stays": int(n), "concepts": concepts, "nonmissing_fraction": nonmiss}


def build_bare_wide_cohort(universe_path: Path, output_path: Path) -> dict:
    df = pd.read_parquet(universe_path)
    cols = {"stay_id": df["stay_id"]}
    if "death" in df.columns:
        cols["death"] = df["death"].fillna(0).astype(int)
    used = []
    for bare, src, _role in CONCEPT_SPECS:
        if src in df.columns:
            cols[bare] = df[src]
            used.append(bare)
    wide = pd.DataFrame(cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(output_path, index=False)
    return {
        "n_stays": int(len(wide)),
        "concepts": used,
        "nonmissing_fraction": {
            c: round(float(wide[c].notna().mean()), 4)
            for c in wide.columns
            if c not in ("stay_id",)
        },
    }


# A PubMed query deliberately scoped to the concept universe we can actually
# execute against (vitals + chemistry + blood gas + coagulation + hematology +
# suspected infection, outcome=mortality). The generic "future research in
# critical care" review query pulls nutrition / sedation / VR / ultrasound
# topics that no vitals+labs cohort can map — a scope mismatch, not a resolver
# fault. Scoping the literature to our concepts is the honest fix.
CONCEPT_SCOPED_TERM = (
    '("Intensive Care Med"[Journal] OR "Crit Care"[Journal] OR "Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal] OR "Shock"[Journal]) '
    "AND (review[Publication Type] OR editorial[Publication Type] OR "
    'systematic[Title/Abstract] OR "meta-analysis"[Publication Type]) '
    "AND ("
    'lactate[Title/Abstract] OR "lactate clearance"[Title/Abstract] OR '
    "hyperlactatemia[Title/Abstract] OR creatinine[Title/Abstract] OR "
    '"acute kidney injury"[Title/Abstract] OR bilirubin[Title/Abstract] OR '
    "albumin[Title/Abstract] OR hypoalbuminemia[Title/Abstract] OR "
    "thrombocytopenia[Title/Abstract] OR platelet[Title/Abstract] OR "
    'coagulopathy[Title/Abstract] OR "international normalized ratio"[Title/Abstract] OR '
    "leukocytosis[Title/Abstract] OR hyperglycemia[Title/Abstract] OR "
    "glucose[Title/Abstract] OR dysnatremia[Title/Abstract] OR sodium[Title/Abstract] OR "
    "acidosis[Title/Abstract] OR bicarbonate[Title/Abstract] OR "
    '"mean arterial pressure"[Title/Abstract] OR hypotension[Title/Abstract] OR '
    "tachycardia[Title/Abstract] OR hypoxemia[Title/Abstract] OR "
    'sepsis[Title/Abstract] OR "suspected infection"[Title/Abstract]) '
    "AND (mortality[Title/Abstract] OR death[Title/Abstract] OR "
    "survival[Title/Abstract] OR prognosis[Title/Abstract] OR outcome[Title/Abstract])"
)


# A query targeting PROGNOSTIC / association studies of routinely-measured ICU
# markers vs mortality. These yield ideas of the form "marker X is associated
# with mortality", where X resolves to an EasyICU lab/score/vital concept and
# the outcome resolves to `death` — i.e. questions a retrospective measurement
# cohort can actually execute (high joint completeness). This is the S2
# feasibility gate working as designed: scope discovery to what the data can
# answer, not gaming — the ideas remain real literature with real uncertainty.
PROGNOSTIC_SCOPED_TERM = (
    '("Intensive Care Med"[Journal] OR "Crit Care"[Journal] OR "Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal] OR "J Crit Care"[Journal] OR "Shock"[Journal] '
    'OR "Aust Crit Care"[Journal]) '
    "AND (prognostic[Title/Abstract] OR prognosis[Title/Abstract] OR "
    'predictor[Title/Abstract] OR "associated with mortality"[Title/Abstract] OR '
    '"predicts mortality"[Title/Abstract] OR "mortality prediction"[Title/Abstract]) '
    "AND ("
    'lactate[Title/Abstract] OR "red cell distribution width"[Title/Abstract] OR '
    'RDW[Title/Abstract] OR "anion gap"[Title/Abstract] OR albumin[Title/Abstract] OR '
    'bilirubin[Title/Abstract] OR "blood urea nitrogen"[Title/Abstract] OR '
    "creatinine[Title/Abstract] OR platelet[Title/Abstract] OR "
    '"mean platelet volume"[Title/Abstract] OR neutrophil[Title/Abstract] OR '
    "lymphocyte[Title/Abstract] OR glucose[Title/Abstract] OR sodium[Title/Abstract] OR "
    'chloride[Title/Abstract] OR "C-reactive protein"[Title/Abstract] OR '
    '"lactate dehydrogenase"[Title/Abstract] OR bicarbonate[Title/Abstract] OR '
    "SOFA[Title/Abstract] OR lymphopenia[Title/Abstract] OR hyperglycemia[Title/Abstract])"
)


# Tighter query biasing esearch toward the classic, high-coverage routine-lab
# prognostic markers that resolve cleanly to single EasyICU concepts and have
# near-universal measurement (so joint completeness is high). All are real,
# heavily-studied mortality-association topics.
LABMARKER_SCOPED_TERM = (
    '("Intensive Care Med"[Journal] OR "Crit Care"[Journal] OR "Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal] OR "J Crit Care"[Journal] OR "Shock"[Journal] '
    'OR "Aust Crit Care"[Journal]) '
    'AND (prognostic[Title/Abstract] OR "associated with mortality"[Title/Abstract] OR '
    '"predicts mortality"[Title/Abstract] OR predictor[Title/Abstract]) '
    "AND ("
    '"red cell distribution width"[Title/Abstract] OR RDW[Title/Abstract] OR '
    '"anion gap"[Title/Abstract] OR lactate[Title/Abstract] OR '
    '"serum albumin"[Title/Abstract] OR hypoalbuminemia[Title/Abstract] OR '
    'bilirubin[Title/Abstract] OR "blood urea nitrogen"[Title/Abstract] OR '
    '"lactate dehydrogenase"[Title/Abstract] OR "C-reactive protein"[Title/Abstract] OR '
    '"mean platelet volume"[Title/Abstract])'
)


# DIVERSE-OUTCOME query: deliberately NOT mortality-anchored. Targets the
# prediction/onset literature for the binary clinical outcomes EasyICU can
# determine and the cohort materializes — acute kidney injury, sepsis-3,
# circulatory failure, RRT, organ dysfunction — so discovered ideas resolve to
# outcomes OTHER than death (aki, sep3_sofa2, circ_failure, ...). The 123-outcome
# determinability catalog then admits them.
DIVERSE_OUTCOME_TERM = (
    '("Intensive Care Med"[Journal] OR "Crit Care"[Journal] OR "Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal] OR "J Crit Care"[Journal] OR "Shock"[Journal] '
    'OR "Kidney Int"[Journal] OR "Nephrol Dial Transplant"[Journal]) '
    "AND (prediction[Title/Abstract] OR predictor[Title/Abstract] OR "
    '"risk factor"[Title/Abstract] OR onset[Title/Abstract] OR '
    'development[Title/Abstract] OR "early identification"[Title/Abstract]) '
    "AND ("
    '"acute kidney injury"[Title/Abstract] OR AKI[Title/Abstract] OR '
    'sepsis[Title/Abstract] OR "septic shock"[Title/Abstract] OR '
    '"circulatory failure"[Title/Abstract] OR "organ dysfunction"[Title/Abstract] OR '
    '"renal replacement therapy"[Title/Abstract] OR "multiorgan failure"[Title/Abstract] OR '
    'coagulopathy[Title/Abstract] OR "respiratory failure"[Title/Abstract])'
)


# ROUTINE-predictor -> DIVERSE-outcome. The only way to get EXECUTABLE non-death
# instances: papers pairing a ROUTINE marker/score (already a concept, high
# coverage) with a NON-death clinical outcome EasyICU determines (AKI, sepsis,
# RRT, circulatory failure). Resolves both sides to dictionary concepts.
ROUTINE_TO_OUTCOME_TERM = (
    '("Intensive Care Med"[Journal] OR "Crit Care"[Journal] OR "Crit Care Med"[Journal] '
    'OR "Ann Intensive Care"[Journal] OR "J Crit Care"[Journal] OR "Shock"[Journal] '
    'OR "Kidney Int"[Journal] OR "J Am Soc Nephrol"[Journal]) '
    "AND ("
    'lactate[Title/Abstract] OR "anion gap"[Title/Abstract] OR SOFA[Title/Abstract] OR '
    'creatinine[Title/Abstract] OR "fluid balance"[Title/Abstract] OR '
    '"urine output"[Title/Abstract] OR "blood urea nitrogen"[Title/Abstract] OR '
    "albumin[Title/Abstract] OR chloride[Title/Abstract] OR bicarbonate[Title/Abstract] OR "
    'platelet[Title/Abstract] OR "base excess"[Title/Abstract]) '
    "AND (predict[Title/Abstract] OR predicts[Title/Abstract] OR "
    'predictor[Title/Abstract] OR "associated with"[Title/Abstract] OR '
    '"risk factor"[Title/Abstract] OR onset[Title/Abstract] OR development[Title/Abstract]) '
    "AND ("
    '"acute kidney injury"[Title/Abstract] OR AKI[Title/Abstract] OR '
    '"renal replacement"[Title/Abstract] OR sepsis[Title/Abstract] OR '
    '"septic shock"[Title/Abstract] OR "circulatory failure"[Title/Abstract] OR '
    '"organ dysfunction"[Title/Abstract])'
)


def fetch_concept_scoped_articles(
    retmax: int, *, term: str | None = None
) -> list[dict]:
    """Fetch review/editorial/meta-analysis abstracts scoped to our concepts."""
    search = harness.pubmed_search(term or CONCEPT_SCOPED_TERM, retmax=retmax)
    return harness.pubmed_fetch(search["pmids"])[:retmax]


def concept_descriptors(used: list[str]) -> list[ConceptDescriptor]:
    out = [
        ConceptDescriptor(
            name="death",
            source_concept="death",
            role=VariableRole.OUTCOME,
            dtype="int64",
        )
    ]
    role_by = {bare: role for bare, _src, role in CONCEPT_SPECS}
    for bare in used:
        out.append(
            ConceptDescriptor(
                name=bare,
                source_concept=bare,
                role=role_by.get(bare, VariableRole.LAB),
                dtype="float64",
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--universe",
        type=Path,
        default=REPO_ROOT / "research_output" / "universe_m3" / "universe_m3.parquet",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "research_output" / "_idea_mining_real_20260614",
    )
    ap.add_argument(
        "--data-source",
        choices=["full-export", "universe"],
        default="full-export",
        help="full-export = collapse the full module-grouped MIIV export to a "
        "~150-concept wide cohort (DB feasibility over the FULL concept set); "
        "universe = the narrow 18-concept universe_m3 (legacy/fallback).",
    )
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_FULL_EXPORT)
    ap.add_argument(
        "--reuse-cohort",
        type=Path,
        default=None,
        help="path to an already-built wide cohort parquet to reuse "
        "(skips the duckdb aggregation).",
    )
    ap.add_argument("--article-count", type=int, default=30)
    ap.add_argument(
        "--scope",
        choices=[
            "concept",
            "generic",
            "prognostic",
            "labmarker",
            "diverse",
            "routine2outcome",
        ],
        default="prognostic",
        help="prognostic = marker-vs-mortality association/prognostic studies "
        "(predictor=concept, outcome=death, executable); concept = broad "
        "concept-keyword reviews; generic = 'future research in critical care'.",
    )
    ap.add_argument(
        "--fulltext",
        dest="fulltext",
        action="store_true",
        default=True,
        help="prefer PMC open-access Discussion/Limitations full-text sections "
        "over the abstract for gap mining (default on); falls back to the "
        "abstract for non-OA articles.",
    )
    ap.add_argument(
        "--no-fulltext",
        dest="fulltext",
        action="store_false",
        help="abstract-only source material (legacy behavior).",
    )
    ap.add_argument(
        "--source-item-catalog",
        type=Path,
        default=DEFAULT_SOURCE_CATALOG,
        help="frozen d_labitems/d_items snapshot for T2/T3 tier triage "
        "(build with tools/build_source_item_catalog.py).",
    )
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--prior-art-top-n", type=int, default=20)
    ap.add_argument(
        "--prior-art-candidate-limit",
        type=int,
        default=12,
        help="maximum mapped, data-feasible hypotheses sent to the expensive "
        "prior-art screen (default: 12; extraction yield remains fully recorded)",
    )
    ap.add_argument(
        "--model",
        default="gpt-5.6-luna",
        help="provider model id; must match the configured OpenAI-compatible "
        "proxy exactly (default: gpt-5.6-luna)",
    )
    ap.add_argument("--request-timeout", type=float, default=600.0)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_path = out_dir / "miiv_wide_idea.parquet"
    if args.reuse_cohort is not None and args.reuse_cohort.exists():
        import pyarrow.parquet as _pq

        print(f"[1/5] reusing prebuilt wide cohort: {args.reuse_cohort}", flush=True)
        wide_path = args.reuse_cohort
        s = _pq.read_schema(wide_path)
        cols = [c for c in s.names if c != "stay_id"]
        n = _pq.read_metadata(wide_path).num_rows
        cohort_summary = {"n_stays": int(n), "concepts": cols}
    elif args.data_source == "full-export":
        print(
            f"[1/5] building FULL wide cohort from export ({args.export_dir.name}) ...",
            flush=True,
        )
        cohort_summary = build_full_wide_cohort(args.export_dir, wide_path)
    else:
        print("[1/5] building 18-concept wide cohort from universe_m3 ...", flush=True)
        cohort_summary = build_bare_wide_cohort(args.universe, wide_path)
    print(
        f"      n_stays={cohort_summary['n_stays']} "
        f"concepts={len(cohort_summary['concepts'])}",
        flush=True,
    )
    print(f"      concept set: {sorted(cohort_summary['concepts'])}", flush=True)

    print(f"[2/5] fetching real PubMed abstracts (scope={args.scope}) ...", flush=True)
    if args.scope == "routine2outcome":
        articles = fetch_concept_scoped_articles(
            args.article_count, term=ROUTINE_TO_OUTCOME_TERM
        )
    elif args.scope == "diverse":
        articles = fetch_concept_scoped_articles(
            args.article_count, term=DIVERSE_OUTCOME_TERM
        )
    elif args.scope == "labmarker":
        articles = fetch_concept_scoped_articles(
            args.article_count, term=LABMARKER_SCOPED_TERM
        )
    elif args.scope == "prognostic":
        articles = fetch_concept_scoped_articles(
            args.article_count, term=PROGNOSTIC_SCOPED_TERM
        )
    elif args.scope == "concept":
        articles = fetch_concept_scoped_articles(args.article_count)
    else:
        articles = harness.fetch_review_editorial_articles(retmax=args.article_count)
    if args.fulltext:
        materials, ft_report = harness.build_fulltext_materials(articles)
        print(
            f"      articles={len(articles)} usable_materials={len(materials)} "
            f"(PMC gap-section={ft_report['from_pmc_gap_section']} "
            f"body={ft_report['from_pmc_body']} "
            f"abstract-fallback={ft_report['from_abstract_fallback']})",
            flush=True,
        )
    else:
        materials = harness.build_materials(articles)
        print(
            f"      articles={len(articles)} usable_materials={len(materials)} "
            f"(abstract-only)",
            flush=True,
        )
    if not materials:
        raise SystemExit("no usable source materials fetched from PubMed")

    print(
        f"[3/5] building real proxy LLM ({args.model}) for idea extraction ...",
        flush=True,
    )
    llm = _make_llm(
        provider="openai", model=args.model, request_timeout=args.request_timeout
    )

    # Expose the FULL EasyICU concept dictionary (~198 concepts) so discovered
    # ideas can map to ANY derivable concept — not just the cohort's columns.
    # The feasibility probe then tests each referenced pair against the wide
    # cohort; concepts we have no data for are simply withheld from ranking.
    catalog = load_concept_catalog()
    concepts = list(catalog.available_concepts)
    concept_aliases = catalog.concept_aliases
    outcome_determinability = dict(catalog.outcome_determinability)
    # Make sure the common mortality phrasings resolve to the death outcome.
    outcome_determinability.setdefault(
        "death", {"outcome": "death", "status": "known_0_1"}
    )
    outcome_determinability.setdefault(
        "mortality", {"outcome": "death", "status": "known_0_1"}
    )
    # CALLER-SIDE outcome declaration (the framework's supported override): the
    # binary "in-use" clinical event concepts (received RRT, mechanically
    # ventilated, on ECMO) are deliberately NOT auto-passed by the catalog's
    # outcome gate because an intervention can be EITHER an exposure or an
    # outcome. For THIS exploration we want them AS outcomes (e.g. "who
    # progresses to needing RRT"), which is a legitimate research framing — so
    # we declare them here rather than globally reclassifying the dictionary.
    # They are event_default_false (absence = no event, not missing).
    for _evt in ("rrt", "mech_vent", "ecmo"):
        outcome_determinability.setdefault(
            _evt, {"outcome": _evt, "status": "known_0_1"}
        )
    print(
        f"      catalog concepts={len(concepts)} "
        f"outcomes_declared={len(outcome_determinability)}",
        flush=True,
    )

    print(
        "[4/5] PubMed prior-art client (novelty; no OpenRouter screen) ...", flush=True
    )
    cache_dir = out_dir / "priorart_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    search_client = harness.PubMedPriorArtScreenClient(
        screener=None, cache_dir=cache_dir
    )

    source_item_index = None
    if args.source_item_catalog and args.source_item_catalog.exists():
        source_item_index = SourceItemIndex.from_json(args.source_item_catalog)
        print(
            f"      source-item catalog: {len(source_item_index)} items "
            f"(T2/T3 tier triage enabled)",
            flush=True,
        )
    else:
        print(
            "      WARNING: no source-item catalog; tier triage disabled "
            "(report will not split T2/T3).",
            flush=True,
        )

    print(
        "[5/5] running idea-mining dry run (S4->S1->S3->S2, stop at gate) ...",
        flush=True,
    )
    result = run_idea_mining_dry_run(
        materials=materials,
        llm=llm,
        available_concepts=concepts,
        concept_aliases=concept_aliases,
        outcome_determinability=outcome_determinability,
        output_dir=out_dir / "dry_run",
        database="miiv",
        data_path=wide_path,
        analytic_unit="stay",
        feasibility_probe=_event_aware_probe,
        top_k=args.top_k,
        prior_art_search_client=search_client,
        prior_art_searched_at=harness._utc_now(),
        prior_art_top_n=args.prior_art_top_n,
        prior_art_candidate_limit=args.prior_art_candidate_limit,
        untraceable_quote_policy="skip",
        malformed_extraction_batch_policy="skip",
        extraction_batch_receipt_dir=out_dir / "extraction_batch_receipts",
        source_item_index=source_item_index,
    )

    print("\n==== DISCOVERY SUMMARY ====", flush=True)
    print(f"literature_ideas extracted: {len(result.literature_ideas)}")
    print(f"executable_candidates:      {len(result.executable_candidates)}")
    print(f"ranked_candidates:          {len(result.ranked_candidates)}")
    print(f"discovery_records:          {len(result.discovery_records)}")
    print(f"triage_report:  {result.triage_report_path}")
    print(f"discovery_md:   {result.discovery_report_path}")
    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(" -", w)


if __name__ == "__main__":
    main()
