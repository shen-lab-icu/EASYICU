"""Tests for extended idea-mining feasibility (ICD cohort + cross-DB dict).

The ICD-cohort assertions depend on the frozen MIMIC-IV ICD catalog
(``benchmark/icd_cohort_catalog_miiv.json``); they skip if it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from easyicu.research_agent.idea_mining_extended_feasibility import (
    ExtendedFeasibilityIndex,
)

REPO = Path(__file__).resolve().parents[2]
ICD_CATALOG = REPO / "benchmark" / "icd_cohort_catalog_miiv.json"
_HAS_ICD = ICD_CATALOG.exists()


@dataclass
class _FakeIdea:
    population: str = ""
    exposure_or_predictor: str = ""
    outcome: str = ""
    analysis_concepts: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.analysis_concepts is None:
            self.analysis_concepts = []


@pytest.fixture(scope="module")
def index() -> ExtendedFeasibilityIndex:
    # a small export so cross-db "already in export" exclusion is meaningful
    return ExtendedFeasibilityIndex.build(
        current_db="miiv", available_concepts=["lact", "map", "death", "crea"]
    )


# ---- Case 2: dictionary / cross-DB reachability -----------------------
def test_cross_db_construct_resolves_to_other_db(index):
    # apixaban is a dictionary concept only for SIC, not miiv
    hit = index.resolve_construct_cross_db("apixaban")
    assert hit is not None
    assert hit.concept == "apixaban"
    assert "sic" in hit.databases
    assert hit.in_current_db is False


def test_cross_db_construct_in_export_is_not_blocking(index):
    # crea is in the supplied export -> not a blocking construct
    assert index.resolve_construct_cross_db("crea") is None


def test_cross_db_unknown_construct_returns_none(index):
    assert index.resolve_construct_cross_db("early mobilization") is None


def test_reconsider_other_db_downgrades_to_hold(index):
    idea = _FakeIdea(
        population="ICU patients", exposure_or_predictor="apixaban", outcome="mortality"
    )
    verdict = index.reconsider(idea=idea, candidate=None)
    assert verdict is not None
    assert verdict.decision == "hold"
    assert verdict.case == "other_db"
    assert "sic" in verdict.metadata.get("databases", [])


# ---- Case 1: ICD-derivable cohort ------------------------------------
@pytest.mark.skipif(not _HAS_ICD, reason="ICD cohort catalog not present")
def test_icd_cohort_confident_for_specific_disease(index):
    for pop in ["ICU patients with traumatic brain injury", "patients with obesity"]:
        p = index.propose_cohort_icd(pop)
        assert p is not None, pop
        assert p.confident is True, pop
        assert p.requires_human_confirm is True
        assert p.total_hadm > 0


@pytest.mark.skipif(not _HAS_ICD, reason="ICD cohort catalog not present")
def test_icd_cohort_broad_category_flagged_not_confident(index):
    p = index.propose_cohort_icd("ICU patients with autoimmune diseases")
    assert p is not None
    assert p.confident is False
    assert p.reliability == "needs_curation"


@pytest.mark.skipif(not _HAS_ICD, reason="ICD cohort catalog not present")
def test_icd_cohort_undercoded_flagged_unreliable(index):
    p = index.propose_cohort_icd("patients at risk for ICU-acquired weakness")
    assert p is not None
    assert p.reliability == "unreliable_undercoded"
    assert p.confident is False


@pytest.mark.skipif(not _HAS_ICD, reason="ICD cohort catalog not present")
def test_icd_cohort_no_clean_code_stays_unresolved(index):
    # "immunocompromised" is a derived state with no clean ICD family
    assert index.propose_cohort_icd("critically ill immunocompromised patients") is None


@pytest.mark.skipif(not _HAS_ICD, reason="ICD cohort catalog not present")
def test_reconsider_icd_cohort_downgrades_with_confirm(index):
    idea = _FakeIdea(
        population="ICU patients with traumatic brain injury",
        exposure_or_predictor="early mobilization",
        outcome="some-unmeasured-outcome",
    )
    verdict = index.reconsider(idea=idea, candidate=None)
    assert verdict is not None
    assert verdict.decision == "hold"
    assert verdict.case == "icd_cohort"
    assert verdict.metadata.get("requires_human_confirm") is True


# ---- fail-closed: genuinely absent stays db-cannot-do ----------------
def test_reconsider_genuinely_absent_returns_none(index):
    idea = _FakeIdea(
        population="critically ill immunocompromised patients",
        exposure_or_predictor="early mobilization",
        outcome="neuromuscular electrical stimulation response",
    )
    assert index.reconsider(idea=idea, candidate=None) is None


# ---- cross-DB transportability novelty axis (prior-art) ----------------
from easyicu.research_agent.idea_mining_priorart import (  # noqa: E402
    _cross_db_prior_art_differentiator,
)
from easyicu.research_agent.idea_mining_schema import (  # noqa: E402
    PriorArtQueryRecord,
    PriorArtSearchHit,
)

_TARGETS = ["miiv", "mimic", "eicu", "aumc", "hirid", "sic"]


def _qr(*hits):
    return PriorArtQueryRecord(query_type="broad", query="q", hit_count=len(hits),
                               top_hits=list(hits))


def _hit(title, direct=True, rationale=None):
    return PriorArtSearchHit(pmid="1", title=title, direct_same_topic=direct,
                             direct_same_topic_rationale=rationale)


def test_crossdb_diff_added_when_crowded_and_no_db_mention():
    hits = [_hit("Obesity and mortality in critically ill patients")]
    diff = _cross_db_prior_art_differentiator([_qr(*hits)], hits, _TARGETS)
    assert diff is not None and "transportability" in diff


def test_crossdb_diff_none_when_prior_art_uses_target_db():
    hits = [_hit("Obesity paradox validated in the MIMIC-IV database")]
    diff = _cross_db_prior_art_differentiator([_qr(*hits)], hits, _TARGETS)
    assert diff is None


def test_crossdb_diff_none_when_multidb_prior_art():
    hits = [_hit("External validation of the obesity paradox across multiple cohorts")]
    diff = _cross_db_prior_art_differentiator([_qr(*hits)], hits, _TARGETS)
    assert diff is None


def test_crossdb_diff_none_when_no_direct_hits_or_disabled():
    hits = [_hit("Unrelated topic", direct=False)]
    assert _cross_db_prior_art_differentiator([_qr(*hits)], [], _TARGETS) is None
    real = [_hit("Obesity and mortality")]
    assert _cross_db_prior_art_differentiator([_qr(*real)], real, None) is None
