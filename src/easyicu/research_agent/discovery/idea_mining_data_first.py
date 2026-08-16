"""Data-first idea generation for the harmonized cross-database layer.

The literature funnel (``idea_mining_funnel``) discovers ideas from what papers
*say* is a gap — a source everyone can read. This module is the complementary
**data-first** arm: it starts from what the platform uniquely holds — concepts
that resolve across many of the six harmonized public ICU databases — and
surfaces predictor/outcome pairs that are broadly *measurable* yet *under-
published*. Such a pair is a cross-database transportability target that only
EasyICU's harmonized layer can execute end to end.

Design constraints (mirrors the rest of idea mining):

* **No fabrication.** A pair is emitted only when both concepts genuinely
  resolve to ``full`` availability in at least ``min_harmonized_dbs`` real
  databases, judged by the dictionary-backed availability engine — never a
  guessed mapping.
* **Deterministic + injectable.** The cross-database availability function and
  the literature-count client are injected (defaulting to the real engine and
  to *no* literature screen). Ordering is fully deterministic.
* **Triage, not a novelty claim.** A low literature count is a HUMAN-REVIEW
  trigger (PubMed under-indexes some venues), never an assertion of novelty —
  identical framing to ``idea_mining_priorart``.

This module is a leaf: it must not import ``idea_mining`` (a module-boundary
test enforces that). It depends only on ``concept_availability``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..concept_availability import (
    default_public_databases,
    hypothesis_cross_database_feasibility,
    normalize_concept_name,
    normalize_database_name,
)

# Signature of the injectable availability engine: (concepts, databases) -> the
# ``hypothesis_cross_database_feasibility`` payload.
FeasibilityFn = Callable[..., Dict[str, Any]]
# A caller-provided literature counter. It owns the concept-key -> queryable-term
# mapping (which this leaf deliberately does not); returns a hit count, or None
# when the pair could not be screened.
LiteratureCounter = Callable[[str, str], Optional[int]]

# A pair whose known literature count is at or below this is flagged as an
# under-published cross-database target (a human-review trigger, not novelty).
_DEFAULT_GAP_MAX_HITS = 20
# Sort key for unscreened pairs: rank them after every screened-known count so a
# confirmed sparse gap surfaces ahead of an unknown one.
_UNSCREENED_LIT_RANK = 10**9


@dataclass(frozen=True)
class DataFirstCandidate:
    """One measurable-but-under-published cross-database pair."""

    predictor: str
    outcome: str
    harmonized_db_count: int
    harmonized_databases: tuple[str, ...]
    total_databases: int
    literature_hit_count: Optional[int]
    literature_screened: bool
    is_under_published: bool
    differentiator_note: str


def _harmonized_databases(
    predictor: str,
    outcome: str,
    databases: Sequence[str],
    feasibility_fn: FeasibilityFn,
) -> List[str]:
    """Databases where BOTH concepts resolve to full availability."""
    payload = feasibility_fn(concepts=[predictor, outcome], databases=list(databases))
    per_db = payload.get("cross_database_feasibility", {})
    harmonized = []
    for db in databases:
        canonical = normalize_database_name(str(db))
        status = per_db.get(canonical)
        if status is None:
            status = per_db.get(str(db))
        if str(status) == "full":
            harmonized.append(db)
    return harmonized


def _differentiator_note(
    harmonized: Sequence[str],
    total: int,
    literature_hit_count: Optional[int],
    is_under_published: bool,
) -> str:
    note = (
        f"measurable in {len(harmonized)}/{total} harmonized public databases "
        f"({', '.join(harmonized)})"
    )
    if is_under_published and literature_hit_count is not None:
        note += (
            f"; under-published (<= {_DEFAULT_GAP_MAX_HITS} prior-art hits; "
            f"observed {literature_hit_count}) — a cross-database transportability "
            "target the harmonized layer can execute end to end. Human prior-art "
            "review required; not a novelty claim."
        )
    elif not is_under_published and literature_hit_count is not None:
        note += f"; literature-screened ({literature_hit_count} prior-art hits)"
    else:
        note += "; literature NOT screened — human prior-art review required"
    return note


def generate_data_first_candidates(
    *,
    predictor_concepts: Sequence[str],
    outcome_concepts: Sequence[str],
    databases: Optional[Sequence[str]] = None,
    feasibility_fn: FeasibilityFn = hypothesis_cross_database_feasibility,
    literature_counter: Optional[LiteratureCounter] = None,
    min_harmonized_dbs: int = 4,
    gap_max_hits: int = _DEFAULT_GAP_MAX_HITS,
    limit: int = 25,
) -> List[DataFirstCandidate]:
    """Rank cross-database measurable predictor/outcome pairs.

    A pair qualifies only when both concepts resolve to ``full`` availability in
    at least ``min_harmonized_dbs`` of the requested (default: the six public)
    databases. Candidates are ordered by harmonization breadth first (widest
    cross-database reach = the platform's differentiator), then by ascending
    literature count so a confirmed under-published pair surfaces first.

    ``literature_counter`` is optional; without it, pairs are ranked on
    harmonization alone and flagged as not-screened. The caller owns the
    concept-key -> literature-term mapping.
    """

    dbs = list(databases) if databases else default_public_databases()
    total = len(dbs)
    seen: set[tuple[str, str]] = set()
    candidates: List[DataFirstCandidate] = []

    for raw_predictor in predictor_concepts:
        predictor = normalize_concept_name(raw_predictor)
        for raw_outcome in outcome_concepts:
            outcome = normalize_concept_name(raw_outcome)
            if not predictor or not outcome or predictor == outcome:
                continue
            key = (predictor, outcome)
            if key in seen:
                continue
            seen.add(key)

            harmonized = _harmonized_databases(predictor, outcome, dbs, feasibility_fn)
            if len(harmonized) < min_harmonized_dbs:
                continue

            screened = literature_counter is not None
            hit_count = literature_counter(predictor, outcome) if screened else None
            # A None from the counter means "could not screen this pair"; do not
            # treat an unknown as an under-published gap.
            screened_ok = screened and hit_count is not None
            is_under_published = bool(screened_ok and hit_count <= gap_max_hits)

            candidates.append(
                DataFirstCandidate(
                    predictor=predictor,
                    outcome=outcome,
                    harmonized_db_count=len(harmonized),
                    harmonized_databases=tuple(harmonized),
                    total_databases=total,
                    literature_hit_count=hit_count,
                    literature_screened=screened_ok,
                    is_under_published=is_under_published,
                    differentiator_note=_differentiator_note(
                        harmonized, total, hit_count, is_under_published
                    ),
                )
            )

    def _sort_key(cand: DataFirstCandidate) -> tuple[int, int, str, str]:
        lit_rank = (
            cand.literature_hit_count
            if cand.literature_screened and cand.literature_hit_count is not None
            else _UNSCREENED_LIT_RANK
        )
        # widest harmonization first, then most under-published, then stable.
        return (-cand.harmonized_db_count, lit_rank, cand.predictor, cand.outcome)

    candidates.sort(key=_sort_key)
    return candidates[:limit]


__all__ = [
    "DataFirstCandidate",
    "FeasibilityFn",
    "LiteratureCounter",
    "generate_data_first_candidates",
]
