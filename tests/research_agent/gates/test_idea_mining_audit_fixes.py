"""Tests for the four architecture fixes from the 2026-06-22 feasibility audit.

* Problem 1 -- derivable composites (obesity / UCR / persistent critical illness)
  are downgraded from db-cannot-do to a human-confirm hold when the primitives
  are present in the export.
* Problem 3 -- raw-table-reachable constructs (nutrition, microbiology) are
  surfaced as agent-extractable holds; truly-absent constructs stay db-cannot-do.
* Problem 2 -- a candidate that passes co-availability but is underpowered is
  demoted from recommend to a human-confirm hold.
* Problem 4 -- a candidate already registered in a prior run/user against a
  shared registry is reported as a homogenization collision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

from easyicu.research_agent.discovery.idea_mining import (
    OutcomeDeterminability,
    SourceMaterial,
    run_idea_mining_dry_run,
)
from easyicu.research_agent.discovery.idea_mining_extended_feasibility import (
    ExtendedFeasibilityIndex,
)
from easyicu.research_agent.discovery.idea_mining_priorart import (
    _MIN_JOINT_COMPLETE_FOR_RECOMMEND,
    _go_no_go_decision,
)
from easyicu.research_agent.discovery.idea_mining_schema import (
    ExecutableHypothesisCandidate,
    IdeaMiningCandidateTriageRecord,
    PriorArtAssessment,
)
from easyicu.research_agent.literature import CitationRecord
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

REPO = Path(__file__).resolve().parents[3]
_HAS_ICD = (
    REPO / "benchmarks" / "catalogs" / "icd_cohort_catalog_miiv.json"
).exists()

PRIMITIVES = ["bmi", "weight", "height", "bun", "crea", "los_icu", "hr", "sbp", "death"]


@dataclass
class _FakeIdea:
    population: str = ""
    exposure_or_predictor: str = ""
    outcome: str = ""
    analysis_concepts: List[str] = field(default_factory=list)


@pytest.fixture(scope="module")
def index() -> ExtendedFeasibilityIndex:
    # Build with an empty ICD catalog so derived/raw cases are isolated from the
    # cohort route (the ordering test re-builds with the real catalog).
    return ExtendedFeasibilityIndex.build(
        current_db="miiv",
        available_concepts=list(PRIMITIVES),
        icd_catalog_path=Path("/nonexistent-icd-catalog.json"),
    )


# --- Problem 1: derivable composites -----------------------------------------


def test_derived_obesity_downgrades_to_hold(index) -> None:
    verdict = index.reconsider(
        idea=_FakeIdea(
            population="adult ICU patients", exposure_or_predictor="obesity"
        ),
        candidate=None,
    )
    assert verdict is not None
    assert verdict.decision == "hold"
    assert verdict.case == "derived_concept"
    assert verdict.metadata["construct"] == "obesity"
    assert "bmi" in verdict.metadata["needs"]
    assert verdict.metadata["requires_human_confirm"] is True


def test_derived_ucr_downgrades_to_hold(index) -> None:
    verdict = index.reconsider(
        idea=_FakeIdea(exposure_or_predictor="urea-to-creatinine ratio elevation"),
        candidate=None,
    )
    assert verdict is not None and verdict.case == "derived_concept"
    assert set(verdict.metadata["needs"]) == {"bun", "crea"}


def test_derived_persistent_critical_illness(index) -> None:
    verdict = index.reconsider(
        idea=_FakeIdea(outcome="persistent critical illness"), candidate=None
    )
    assert verdict is not None and verdict.case == "derived_concept"
    assert verdict.metadata["needs"] == ["los_icu"]


def test_derived_not_fired_when_primitive_absent() -> None:
    # No bmi/weight/height in the export -> obesity is NOT derivable here.
    bare = ExtendedFeasibilityIndex.build(
        current_db="miiv",
        available_concepts=["lact", "map"],
        icd_catalog_path=Path("/nonexistent-icd-catalog.json"),
    )
    assert bare.propose_derived_construct("obesity") is None
    assert (
        bare.reconsider(idea=_FakeIdea(exposure_or_predictor="obesity"), candidate=None)
        is None
    )


# --- Problem 3: raw-table reachability ---------------------------------------


def test_raw_nutrition_downgrades_to_hold(index) -> None:
    verdict = index.reconsider(
        idea=_FakeIdea(exposure_or_predictor="excessive dietary protein intake"),
        candidate=None,
    )
    assert verdict is not None
    assert verdict.case == "raw_extraction"
    assert "ingredientevents" in verdict.metadata["raw_table"]
    assert "coding agent" in verdict.reason
    # Do not confuse dietary protein intake with the serum total-protein lab.
    assert verdict.metadata["construct"] == "nutrition_intake"


def test_raw_microbiology_downgrades_to_hold(index) -> None:
    verdict = index.reconsider(
        idea=_FakeIdea(exposure_or_predictor="positive blood culture organism"),
        candidate=None,
    )
    assert verdict is not None and verdict.case == "raw_extraction"
    assert verdict.metadata["raw_table"] == "hosp.microbiologyevents"


def test_truly_absent_construct_stays_db_cannot_do(index) -> None:
    # Post-discharge cognition has no structured raw source -> no downgrade, so the
    # base db-cannot-do verdict stands (fail-closed).
    verdict = index.reconsider(
        idea=_FakeIdea(
            population="ICU survivors after discharge",
            exposure_or_predictor="low-grade inflammation",
            outcome="post-ICU cognitive impairment",
        ),
        candidate=None,
    )
    assert verdict is None


@pytest.mark.skipif(not _HAS_ICD, reason="ICD catalog absent")
def test_derived_beats_icd_cohort_route() -> None:
    # When BOTH the cohort is ICD-derivable AND a construct is derivable, the
    # derived route wins (the construct blocker is more actionable than the
    # population route).
    idx = ExtendedFeasibilityIndex.build(
        current_db="miiv", available_concepts=list(PRIMITIVES)
    )
    verdict = idx.reconsider(
        idea=_FakeIdea(
            population="traumatic brain injury patients",
            exposure_or_predictor="obesity",
        ),
        candidate=None,
    )
    assert verdict is not None and verdict.case == "derived_concept"


# --- Problem 2: data-sufficiency / power floor -------------------------------


def _recommend_ready_assessment() -> PriorArtAssessment:
    return PriorArtAssessment(
        novelty_snapshot_id="novelty-snapshot/sha256:deadbeef",
        literature_idea_id="litidea_test",
        source_snapshot_id="source-snapshot/sha256:abc123",
        searched_at="2026-06-22T00:00:00+00:00",
        predictor_literature_phrase="lactate",
        outcome_literature_phrase="mortality",
        query_records=[],
        differentiators=["first-24h trajectory"],
        has_specific_differentiator=True,
        novelty_label="sparse",
        literature_saturation_signal=0.0,
        novelty_statement="prior-art triage is not a novelty claim",
        same_topic_screen_status="top-N same-topic screened",
    )


def _executable_candidate() -> ExecutableHypothesisCandidate:
    return ExecutableHypothesisCandidate(
        executable_candidate_id="execidea_test",
        literature_idea_id="litidea_test",
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        population="adult ICU patients",
        predictor_label="lactate",
        outcome_label="mortality",
        research_question="Does lactate predict mortality?",
        source_quote="future work should study lactate and mortality",
        resolved_predictor_concept="lact",
        resolved_outcome_concept="death",
    )


def _triage(*, n_joint: int, denom: int) -> IdeaMiningCandidateTriageRecord:
    return IdeaMiningCandidateTriageRecord(
        literature_idea_id="litidea_test",
        executable_candidate_id="execidea_test",
        registry_candidate_id="regidea_test",
        hypothesis_family_id="idea-family/sha256:test",
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        predictor_label="lactate",
        outcome_label="mortality",
        resolved_predictor_concept="lact",
        resolved_outcome_concept="death",
        feasibility_pair_key=("lact", "death"),
        executable=True,
        coverage_source="pair_joint_feasibility",
        n_joint_complete=n_joint,
        denominator_n=denom,
        multiple_testing_family_size=1,
        multiple_testing_executable_family_size=1,
        multiple_testing_note="single candidate",
        causal_audit_risk="low",
        causal_audit_scope="association",
    )


def test_power_floor_demotes_recommend_to_hold() -> None:
    # Columns co-exist but only a handful of joint-complete units -> hold, not go.
    decision, reason = _go_no_go_decision(
        candidate=_executable_candidate(),
        assessment=_recommend_ready_assessment(),
        triage=_triage(n_joint=10, denom=100),
    )
    assert decision == "hold"
    assert "power floor" in reason


def test_fraction_floor_demotes_recommend_to_hold() -> None:
    # Plenty of absolute units but a tiny fraction of the cohort -> selection-bias
    # hold.
    decision, reason = _go_no_go_decision(
        candidate=_executable_candidate(),
        assessment=_recommend_ready_assessment(),
        triage=_triage(n_joint=200, denom=1_000_000),
    )
    assert decision == "hold"
    assert "fraction" in reason


def test_sufficient_power_still_recommends() -> None:
    decision, reason = _go_no_go_decision(
        candidate=_executable_candidate(),
        assessment=_recommend_ready_assessment(),
        triage=_triage(n_joint=5_000, denom=50_000),
    )
    assert decision == "recommend"
    assert _MIN_JOINT_COMPLETE_FOR_RECOMMEND <= 5_000


# --- Problem 4: registry collision (homogenization) -------------------------


def _one_idea_llm() -> ScriptedMockLLMClient:
    return ScriptedMockLLMClient(
        [
            json.dumps(
                [
                    {
                        "citation_key": "neutral_review_2026",
                        "population": "adult ICU patients",
                        "exposure_or_predictor": "lactate",
                        "outcome": "patient-centered endpoint",
                        "rationale": "The source describes this direction.",
                        "source_quote": "lactate trajectory",
                        "analysis_family": "association",
                    }
                ]
            )
        ]
    )


def _nutrition_material() -> SourceMaterial:
    return SourceMaterial(
        citation=CitationRecord(
            key="neutral_review_2026",
            title="ICU physiology review",
            year="2026",
            venue="Critical Care Review",
        ),
        source_adapter_level="user_supplied_excerpt",
        source_text="The authors note that lactate trajectory may predict outcome.",
    )


def _run_once(tmp_path: Path, registry_path: Path, tag: str):
    from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

    concepts = [
        ConceptDescriptor(
            name="lactate_first24h",
            source_concept="lact",
            derived_from_concepts=["lactate"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    def fake_probe(**kwargs):
        pair = tuple(kwargs["concepts"])
        return {
            c: {
                "joint_fraction_complete": 0.8,
                "n_joint_complete": 8000,
                "denominator_n": 10000,
                "source": "synthetic_fixture",
            }
            for c in pair
        }

    return run_idea_mining_dry_run(
        materials=[_nutrition_material()],
        llm=_one_idea_llm(),
        available_concepts=concepts,
        outcome_determinability={
            "endpoint_known": OutcomeDeterminability(
                outcome="endpoint_known", status="known_0_1"
            )
        },
        output_dir=tmp_path / tag,
        registry_path=registry_path,
        database="miiv",
        data_path=tmp_path / "cohort.parquet",
        feasibility_probe=fake_probe,
    )


def test_registry_collision_reported_on_shared_registry(tmp_path) -> None:
    registry_path = tmp_path / "shared_registry.json"

    first = _run_once(tmp_path, registry_path, "run1")
    assert first.registry_collisions == []  # fresh registry, no prior art

    second = _run_once(tmp_path, registry_path, "run2")
    assert second.registry_collisions, "second run should collide with the first"
    collision = second.registry_collisions[0]
    assert collision["predictor_label"] == "lactate"
    assert "prior run" in collision["note"]
