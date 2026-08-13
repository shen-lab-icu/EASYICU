#!/usr/bin/env python3
"""Run the provider-free data-first Idea Mining route on a prepared cohort.

This command does not extract data and does not ask an LLM to invent topics.
It enumerates concepts already present in a prepared EasyICU wide cohort,
requires dictionary-backed availability across the harmonized public ICU
databases, then sends the surviving pairs through the standard real-data,
PubMed prior-art, discovery-ledger, and human-confirmation gates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_idea_mining_s6_validation_harness import (  # noqa: E402
    PubMedPriorArtScreenClient,
)

from easyicu.research_agent.concept_availability import (  # noqa: E402
    normalize_concept_name,
    real_data_concept_feasibility,
)
from easyicu.research_agent.concept_catalog import (  # noqa: E402
    DERIVED_CONCEPT_HINTS,
    load_concept_catalog,
)
from easyicu.research_agent.discovery.idea_mining_data_first_route import (  # noqa: E402
    run_data_first_idea_mining_dry_run,
)
from easyicu.research_agent.discovery.idea_mining_feasibility_tier import (  # noqa: E402
    SourceItemIndex,
)

DEFAULT_PREPARED_COHORT = (
    REPO_ROOT
    / "research_output"
    / "experiments"
    / "FIG5-DISC-001"
    / "triage"
    / "miiv_wide_idea.parquet"
)
DEFAULT_SOURCE_CATALOG = (
    REPO_ROOT / "benchmarks" / "catalogs" / "source_item_catalog_miiv.json"
)

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

_IDENTIFIER_COLUMNS = frozenset(
    {
        "stay_id",
        "subject_id",
        "hadm_id",
        "patient_id",
        "encounter_id",
        "row_id",
        "source_row_id",
        "database",
    }
)


def _event_aware_probe(**kwargs):
    return real_data_concept_feasibility(
        event_default_false_concepts=EVENT_DEFAULT_FALSE,
        **kwargs,
    )


def _parse_concepts(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _automatic_predictors(
    columns: list[str],
    outcomes: list[str],
    *,
    derived_only: bool,
) -> list[str]:
    """Return host-defined exposure candidates, never derived endpoints.

    The automatic route must not silently reinterpret a derived binary outcome
    (for example AKI or persistent critical illness) as an exposure merely
    because the caller selected a different outcome for this run.  Callers can
    still request such a longitudinal relationship explicitly through
    ``--predictor-concepts``; the automatic discovery pool stays conservative.
    """

    excluded = _IDENTIFIER_COLUMNS | {str(item).strip() for item in outcomes}
    catalog = load_concept_catalog(restrict_to=columns)
    return sorted(
        concept
        for concept in catalog.available_concepts
        if concept in columns
        and concept not in excluded
        and not DERIVED_CONCEPT_HINTS.get(concept, ([], False))[1]
        and (not derived_only or concept in DERIVED_CONCEPT_HINTS)
    )


def _preferred_literature_term(concept: str, aliases: list[str]) -> str:
    """Choose a query-safe clinical phrase instead of an ambiguous acronym."""

    hinted = list(DERIVED_CONCEPT_HINTS.get(concept, ([], False))[0])
    hinted_has_expansion = any(
        len(item.replace("-", " ").split()) >= 2 and not item.isupper()
        for item in hinted
    )
    source_aliases = hinted if hinted_has_expansion else aliases
    candidates: list[tuple[int, int, str]] = []
    for raw in source_aliases:
        text = " ".join(str(raw).strip().split())
        if not text or text.casefold() == concept.casefold():
            continue
        if any(token in text for token in ("=", "(", ")", "*", "/")):
            continue
        words = text.replace("-", " ").split()
        if not 2 <= len(words) <= 6:
            continue
        if text.isupper() and len(text) <= 8:
            continue
        candidates.append((len(words), len(text), text))
    if not candidates:
        return concept
    # Prefer an explicit multi-word clinical expansion; ties prefer the shorter
    # phrase so formulas/descriptions do not leak into the query.
    best_words = max(item[0] for item in candidates)
    return min(
        (item for item in candidates if item[0] == best_words),
        key=lambda item: (item[1], item[2].casefold()),
    )[2]


def _query_literature_aliases(
    concept: str, aliases: list[str], *, primary: str
) -> list[str]:
    """Return bounded catalog spellings for exact-query recall."""

    out: list[str] = []
    seen = {primary.casefold(), concept.casefold()}
    for raw in aliases:
        text = " ".join(str(raw).strip().split())
        if not text or any(token in text for token in ("=", "(", ")", "*")):
            continue
        if not 1 <= len(text.replace("-", " ").split()) <= 7:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= 8:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-cohort", type=Path, default=DEFAULT_PREPARED_COHORT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--outcome-concepts",
        default="death",
        help="Comma-separated host-declared outcomes (default: death).",
    )
    parser.add_argument(
        "--predictor-concepts",
        default=None,
        help=(
            "Optional comma-separated predictor scope. If omitted, all non-ID "
            "concept columns in the prepared cohort are screened."
        ),
    )
    parser.add_argument(
        "--candidate-scope",
        choices=["derived", "all"],
        default="derived",
        help=(
            "Automatic predictor scope. 'derived' prioritizes reusable "
            "host-defined constructs where harmonization is a platform "
            "differentiator; 'all' also screens raw measurements. Ignored when "
            "--predictor-concepts is supplied."
        ),
    )
    parser.add_argument("--database", default="miiv")
    parser.add_argument("--min-harmonized-dbs", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--prior-art-top-n", type=int, default=20)
    parser.add_argument(
        "--source-item-catalog", type=Path, default=DEFAULT_SOURCE_CATALOG
    )
    args = parser.parse_args(argv)

    cohort_path = args.prepared_cohort.resolve()
    if not cohort_path.is_file():
        raise SystemExit(f"prepared cohort not found: {cohort_path}")
    columns = list(pq.read_schema(cohort_path).names)
    outcomes = _parse_concepts(args.outcome_concepts)
    if not outcomes:
        raise SystemExit("at least one --outcome-concepts value is required")
    missing_outcomes = [outcome for outcome in outcomes if outcome not in columns]
    if missing_outcomes:
        raise SystemExit(
            "declared outcomes are absent from prepared cohort: "
            + ", ".join(missing_outcomes)
        )
    predictors = _parse_concepts(args.predictor_concepts) or _automatic_predictors(
        columns,
        outcomes,
        derived_only=args.candidate_scope == "derived",
    )
    if not predictors:
        raise SystemExit("no predictor concepts remain after host filtering")

    catalog = load_concept_catalog(restrict_to=[*predictors, *outcomes])
    literature_terms: dict[str, str] = {}
    literature_aliases: dict[str, list[str]] = {}
    for concept in [*predictors, *outcomes]:
        aliases = list(catalog.concept_aliases.get(concept, []))
        term = _preferred_literature_term(
            concept,
            aliases,
        )
        literature_terms[concept] = term
        literature_terms[normalize_concept_name(concept)] = term
        literature_aliases[concept] = _query_literature_aliases(
            concept, aliases, primary=term
        )
    literature_terms["death"] = "hospital mortality"
    source_item_index = (
        SourceItemIndex.from_json(args.source_item_catalog)
        if args.source_item_catalog.is_file()
        else None
    )
    cache_dir = args.out_dir.resolve() / "priorart_cache"
    search_client = PubMedPriorArtScreenClient(
        screener=None,
        cache_dir=cache_dir,
        top_n_screen=5,
    )
    print(
        f"[data-first] prepared cohort={cohort_path} columns={len(columns)} "
        f"predictors={len(predictors)} outcomes={outcomes}",
        flush=True,
    )
    result = run_data_first_idea_mining_dry_run(
        predictor_concepts=predictors,
        outcome_concepts=outcomes,
        available_concepts=catalog.available_concepts,
        concept_aliases=catalog.concept_aliases,
        outcome_determinability=catalog.outcome_determinability,
        output_dir=args.out_dir,
        data_path=cohort_path,
        prior_art_search_client=search_client,
        database=args.database,
        feasibility_probe=_event_aware_probe,
        min_harmonized_dbs=args.min_harmonized_dbs,
        top_k=args.top_k,
        prior_art_top_n=args.prior_art_top_n,
        source_item_index=source_item_index,
        literature_terms=literature_terms,
        literature_aliases=literature_aliases,
    )
    print(
        "[data-first] "
        f"ideas={len(result.literature_ideas)} "
        f"executable={sum(c.executable for c in result.executable_candidates)} "
        f"discovery_rows={len(result.discovery_records)}",
        flush=True,
    )
    print(f"[data-first] triage={result.triage_report_path}", flush=True)
    print(f"[data-first] report={result.discovery_report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
