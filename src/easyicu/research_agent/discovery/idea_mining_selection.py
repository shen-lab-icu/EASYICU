"""Host-owned actionability selection for expensive idea prior-art screens.

This leaf sits between concept/data feasibility and novelty search.  It never
declares novelty or scientific validity; it only decides which already mapped,
data-answerable hypotheses deserve the bounded external-search budget.
"""

from __future__ import annotations

import re
from typing import List, Literal, Mapping, Sequence, Tuple

from ..concept_availability import normalize_concept_name
from ..planning.analysis_types import normalize_analysis_family
from .hypothesis_generator import HypothesisFeasibilitySignal
from .idea_mining_priorart import _candidate_differentiators
from .idea_mining_schema import (
    ExecutableHypothesisCandidate,
    LiteratureIdeaCandidate,
)

_TREATMENT_LIKE_FAMILIES = frozenset({"treatment_response", "causal_inference"})
_TREATMENT_MIN_OBSERVED_FRACTION = 0.70
_PEDIATRIC_POPULATION = re.compile(
    r"\b(?:child|children|paediatric|pediatric|neonate|neonatal|infant|infants)\b",
    re.IGNORECASE,
)
_ADULT_POPULATION = re.compile(r"\b(?:adult|adults)\b", re.IGNORECASE)


def _population_matches_age_group(
    population: str,
    analytic_population_age_group: Literal["adult", "pediatric", "mixed"] | None,
) -> bool:
    """Reject only explicit age-group contradictions.

    An unspecified population remains eligible; the human gate still owns
    clinical applicability.  This function only prevents a clearly pediatric
    source question from consuming an adult-cohort screen (and vice versa).
    """

    if analytic_population_age_group in {None, "mixed"}:
        return True
    text = str(population or "")
    if analytic_population_age_group == "adult":
        return _PEDIATRIC_POPULATION.search(text) is None
    return _ADULT_POPULATION.search(text) is None


def _signal_is_actionable_for_screen(
    idea: LiteratureIdeaCandidate,
    signal: HypothesisFeasibilitySignal,
) -> bool:
    """Whether host-observed data can support spending novelty-search budget."""

    if signal.joint_fraction_complete <= 0.0:
        return False
    contrast = signal.predictor_contrast_fraction
    if contrast is not None and contrast <= 0.0:
        return False
    family = normalize_analysis_family(idea.analysis_family)
    # Sparse administration records do not prove untreated rows are true zeroes.
    # Until a host-owned exposure absence contract exists, a treatment/causal
    # question is screenable only when the exposure/outcome pair is broadly
    # observed. This is a search-budget gate, not an analysis authorization.
    if (
        family in _TREATMENT_LIKE_FAMILIES
        and signal.joint_fraction_complete < _TREATMENT_MIN_OBSERVED_FRACTION
    ):
        return False
    return True


def _candidate_key(
    candidate: ExecutableHypothesisCandidate,
) -> Tuple[str, str, str, str]:
    if candidate.feasibility_pair_key:
        predictor, outcome = candidate.feasibility_pair_key
    else:
        predictor = "set:" + "|".join(
            sorted(
                normalize_concept_name(item)
                for item in candidate.resolved_analysis_concepts
                if item
            )
        )
        outcome = ""
    return (
        normalize_concept_name(predictor),
        normalize_concept_name(outcome),
        normalize_concept_name(candidate.analysis_family),
        candidate.feature_derivation_status,
    )


def select_actionable_prior_art_screen(
    *,
    literature_ideas: Sequence[LiteratureIdeaCandidate],
    executable_candidates: Sequence[ExecutableHypothesisCandidate],
    feasibility_by_pair: Mapping[Tuple[str, str], HypothesisFeasibilitySignal],
    limit: int,
    analytic_population_age_group: Literal["adult", "pediatric", "mixed"] | None = None,
) -> Tuple[List[LiteratureIdeaCandidate], List[ExecutableHypothesisCandidate]]:
    """Return a bounded, deterministic, data-answerable review surface.

    Expensive novelty searches should not be spent on a phrase that failed
    host concept mapping or on a predictor/outcome pair for which prepared data
    produced no joint-feasibility signal.  Among remaining unique hypotheses,
    prefer a specific literature differentiator, observable exposure contrast,
    and higher joint completeness.  Source order is the final tie-breaker.
    """

    idea_by_id = {str(idea.literature_idea_id): idea for idea in literature_ideas}
    source_position = {
        str(idea.literature_idea_id): position
        for position, idea in enumerate(literature_ideas)
    }
    seen: set[Tuple[str, str, str, str]] = set()
    ranked: List[
        Tuple[
            Tuple[int, int, float, float, int],
            LiteratureIdeaCandidate,
            ExecutableHypothesisCandidate,
        ]
    ] = []
    for candidate in executable_candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        pair = candidate.feasibility_pair_key
        if not candidate.executable or not pair:
            continue
        normalized_pair = (
            normalize_concept_name(pair[0]),
            normalize_concept_name(pair[1]),
        )
        signal = feasibility_by_pair.get(normalized_pair)
        idea = idea_by_id.get(str(candidate.literature_idea_id))
        if signal is None or idea is None:
            continue
        if not _population_matches_age_group(
            idea.population,
            analytic_population_age_group,
        ):
            continue
        if not _signal_is_actionable_for_screen(idea, signal):
            continue
        contrast = signal.predictor_contrast_fraction
        ranked.append(
            (
                (
                    int(bool(_candidate_differentiators(idea))),
                    int(contrast is not None and contrast > 0.0),
                    float(signal.joint_fraction_complete),
                    float(contrast if contrast is not None else -1.0),
                    -source_position[str(candidate.literature_idea_id)],
                ),
                idea,
                candidate,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:limit]
    return (
        [item[1] for item in selected],
        [item[2] for item in selected],
    )


__all__ = ["select_actionable_prior_art_screen"]
