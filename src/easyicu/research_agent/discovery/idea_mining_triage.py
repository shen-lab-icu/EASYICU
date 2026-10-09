"""Owner: the triage outputs of an idea-mining dry run.

``build_yield_report`` counts how the received literature items became
executable candidates, including the items excluded before concept mapping,
and names the most frequent unresolved labels and non-executable reasons.
``build_candidate_records`` writes one triage record per candidate: its
registry status, its ranking coordinates, and the multiple-testing
denominators.  Both describe candidates; neither is a research finding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..concept_availability import normalize_concept_name
from .idea_mining_schema import (
    ExecutableHypothesisCandidate,
    IdeaMiningCandidateTriageRecord,
    IdeaMiningYieldReport,
    LiteratureIdeaCandidate,
)
from .idea_registry import CandidateNotRegisteredError, IdeaCandidateRegistry


def normalise_pair_tuple(pair: Tuple[str, str]) -> Tuple[str, str]:
    return (normalize_concept_name(pair[0]), normalize_concept_name(pair[1]))


def _top_values(values: Sequence[str], *, limit: int = 5) -> List[str]:
    counts: Dict[str, int] = {}
    for value in values:
        text = str(value or "").strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return sorted(counts, key=lambda item: (-counts[item], item))[:limit]


def build_yield_report(
    literature_ideas: Sequence[LiteratureIdeaCandidate],
    candidates: Sequence[ExecutableHypothesisCandidate],
    *,
    extraction_accounting: Optional[Mapping[str, int]] = None,
) -> IdeaMiningYieldReport:
    accounting = extraction_accounting or {}
    received = int(accounting.get("n_candidate_items_received", len(literature_ideas)))
    unresolved_predictors = [
        candidate.predictor_label
        for candidate in candidates
        if candidate.resolved_predictor_concept is None
    ]
    unresolved_outcomes = [
        candidate.outcome_label
        for candidate in candidates
        if candidate.resolved_outcome_concept is None
    ]
    reasons = [
        reason
        for candidate in candidates
        for reason in candidate.non_executable_reasons
    ]
    return IdeaMiningYieldReport(
        n_literature_ideas=len(literature_ideas),
        n_candidate_items_received=received,
        n_candidates_excluded_before_mapping=max(
            0,
            received - len(literature_ideas),
        ),
        n_dropped_untraceable=int(accounting.get("n_dropped_untraceable", 0)),
        n_dropped_invalid=int(accounting.get("n_dropped_invalid", 0)),
        n_malformed_extraction_batches=int(
            accounting.get("n_malformed_extraction_batches", 0)
        ),
        n_sources_in_malformed_batches=int(
            accounting.get("n_sources_in_malformed_batches", 0)
        ),
        n_resolved_predictor=sum(
            1 for candidate in candidates if candidate.resolved_predictor_concept
        ),
        n_resolved_outcome=sum(
            1 for candidate in candidates if candidate.resolved_outcome_concept
        ),
        n_executable=sum(1 for candidate in candidates if candidate.executable),
        n_non_executable=sum(1 for candidate in candidates if not candidate.executable),
        unresolved_predictor_labels=_top_values(unresolved_predictors),
        unresolved_outcome_labels=_top_values(unresolved_outcomes),
        top_non_executable_reasons=_top_values(reasons),
    )


def build_candidate_records(
    *,
    candidates: Sequence[ExecutableHypothesisCandidate],
    ranking_by_pair: Mapping[Tuple[str, str], Mapping[str, Any]],
    registry_ids: Mapping[str, str],
    registry: IdeaCandidateRegistry,
    hypothesis_family_id: str,
    source_snapshot_id: str,
    candidate_screening_denominator: int,
) -> List[IdeaMiningCandidateTriageRecord]:
    family_size = max(
        registry.family_size(hypothesis_family_id),
        int(candidate_screening_denominator),
    )
    executable_family_size = len(
        {
            registry_ids.get(
                candidate.executable_candidate_id,
                candidate.executable_candidate_id,
            )
            for candidate in candidates
            if candidate.executable
        }
    )
    records: List[IdeaMiningCandidateTriageRecord] = []
    for candidate in candidates:
        pair_key = (
            normalise_pair_tuple(candidate.feasibility_pair_key)
            if candidate.feasibility_pair_key
            else None
        )
        ranked = ranking_by_pair.get(pair_key) if pair_key else None
        registry_candidate_id = registry_ids.get(
            candidate.executable_candidate_id,
            candidate.executable_candidate_id,
        )
        try:
            selection_status = registry.latest_entry(
                registry_candidate_id
            ).selection_status
        except CandidateNotRegisteredError:
            selection_status = "proposed"
        records.append(
            IdeaMiningCandidateTriageRecord(
                literature_idea_id=candidate.literature_idea_id,
                executable_candidate_id=candidate.executable_candidate_id,
                registry_candidate_id=registry_candidate_id,
                hypothesis_family_id=hypothesis_family_id,
                source_snapshot_id=source_snapshot_id,
                citation_key=candidate.citation_key,
                predictor_label=candidate.predictor_label,
                outcome_label=candidate.outcome_label,
                resolved_predictor_concept=candidate.resolved_predictor_concept,
                resolved_outcome_concept=candidate.resolved_outcome_concept,
                analysis_family=candidate.analysis_family,
                resolved_analysis_concepts=list(candidate.resolved_analysis_concepts),
                feasibility_pair_key=pair_key,
                feature_derivation_status=candidate.feature_derivation_status,
                feature_derivation_requirements=list(
                    candidate.feature_derivation_requirements
                ),
                feature_derivation_note=candidate.feature_derivation_note,
                executable=candidate.executable,
                non_executable_reasons=list(candidate.non_executable_reasons),
                ranking_candidate_id=(
                    str(ranked.get("candidate_id")) if ranked else None
                ),
                priority_score=(
                    float(ranked["priority_score"])
                    if ranked and ranked.get("priority_score") is not None
                    else None
                ),
                coverage_source=(
                    str(ranked["coverage_source"])
                    if ranked and ranked.get("coverage_source") is not None
                    else None
                ),
                feasibility_note=(
                    str(ranked["feasibility_note"])
                    if ranked and ranked.get("feasibility_note") is not None
                    else None
                ),
                n_joint_complete=(
                    int(ranked["n_joint_complete"])
                    if ranked and ranked.get("n_joint_complete") is not None
                    else None
                ),
                denominator_n=(
                    int(ranked["denominator_n"])
                    if ranked and ranked.get("denominator_n") is not None
                    else None
                ),
                registry_selection_status=str(selection_status),
                multiple_testing_family_size=family_size,
                multiple_testing_executable_family_size=executable_family_size,
                multiple_testing_note=(
                    "All received candidate items, including items excluded before "
                    "mapping, remain in the conservative all-considered "
                    "preregistered denominator; "
                    "the executable denominator is reported separately. No p-values "
                    "are computed or adjusted in the S5 dry run."
                ),
                causal_audit_risk=(
                    "static_triage_marker_requires_post_analysis_causal_audit"
                ),
                causal_audit_scope=(
                    "static_triage_marker_no_per_candidate_causal_audit"
                ),
            )
        )
    return records


__all__ = [
    "build_candidate_records",
    "build_yield_report",
    "normalise_pair_tuple",
]
