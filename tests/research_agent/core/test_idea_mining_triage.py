"""An idea-mining dry run's triage outputs name candidates by normalised pairs.

The triage owner looks a candidate's ranking up by its normalised
predictor/outcome pair, and the yield report names only the most frequent
unresolved labels.  Synthetic candidates; no benchmark item.
"""

from __future__ import annotations

from typing import Optional

from easyicu.research_agent.discovery.idea_mining_schema import (
    ExecutableHypothesisCandidate,
)
from easyicu.research_agent.discovery.idea_mining_triage import (
    build_candidate_records,
    build_yield_report,
    normalise_pair_tuple,
)
from easyicu.research_agent.discovery.idea_registry import CandidateNotRegisteredError


def _candidate(
    candidate_id: str,
    predictor: str,
    outcome: str,
    *,
    resolved_predictor: Optional[str] = None,
) -> ExecutableHypothesisCandidate:
    return ExecutableHypothesisCandidate(
        executable_candidate_id=candidate_id,
        literature_idea_id=f"idea-{candidate_id}",
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        population="adult ICU patients",
        predictor_label=predictor,
        outcome_label=outcome,
        resolved_predictor_concept=resolved_predictor,
        resolved_outcome_concept="death",
        feasibility_pair_key=(predictor, outcome),
        research_question=f"Does {predictor} associate with {outcome}?",
        source_quote="future work should study this",
    )


class _UnregisteredRegistry:
    def family_size(self, _hypothesis_family_id: str) -> int:
        return 1

    def latest_entry(self, candidate_id: str):
        raise CandidateNotRegisteredError(candidate_id)


def test_a_pair_key_is_normalised_on_both_sides() -> None:
    assert normalise_pair_tuple(("Lactate Clearance", " Mortality ")) == (
        "lactate_clearance",
        "death",
    )


def test_a_candidate_record_finds_its_ranking_by_the_normalised_pair() -> None:
    candidate = _candidate(
        "exec-a", "Lactate Clearance", "Mortality", resolved_predictor="lactate"
    )
    ranking = {
        ("lactate_clearance", "death"): {
            "candidate_id": "rank-1",
            "priority_score": 0.75,
            "n_joint_complete": 120,
            "denominator_n": 150,
        }
    }

    (record,) = build_candidate_records(
        candidates=[candidate],
        ranking_by_pair=ranking,
        registry_ids={},
        registry=_UnregisteredRegistry(),
        hypothesis_family_id="family-a",
        source_snapshot_id="source-snapshot/sha256:abc123",
        candidate_screening_denominator=3,
    )

    assert record.feasibility_pair_key == ("lactate_clearance", "death")
    assert (record.ranking_candidate_id, record.priority_score) == ("rank-1", 0.75)
    assert (record.n_joint_complete, record.denominator_n) == (120, 150)
    assert record.registry_selection_status == "proposed"
    assert record.multiple_testing_family_size == 3


def test_the_yield_report_names_the_five_most_frequent_unresolved_labels() -> None:
    labels = ["b", "a", "c", "c", "d", "e", "f", "f", "f", "g"]
    candidates = [
        _candidate(f"exec-{index}", label, "death")
        for index, label in enumerate(labels)
    ]

    report = build_yield_report([], candidates)

    assert report.unresolved_predictor_labels == ["f", "c", "a", "b", "d"]
    assert report.n_resolved_predictor == 0
