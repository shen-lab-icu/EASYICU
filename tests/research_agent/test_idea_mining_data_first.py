"""Tests for the data-first cross-database idea generator."""

from __future__ import annotations

import importlib
import inspect

from easyicu.research_agent.discovery.idea_mining_data_first import (
    DataFirstCandidate,
    generate_data_first_candidates,
)
from easyicu.research_agent.concept_availability import normalize_concept_name

_DBS = ["aumc", "eicu", "hirid", "miiv", "mimic", "sic"]


def _fake_feasibility(pair_statuses):
    """Build an injectable feasibility_fn from {(pred, out): {db: status}}."""
    norm = {
        (normalize_concept_name(p), normalize_concept_name(o)): statuses
        for (p, o), statuses in pair_statuses.items()
    }

    def fn(*, concepts, databases):
        key = (
            normalize_concept_name(concepts[0]),
            normalize_concept_name(concepts[1]),
        )
        statuses = norm.get(key, {})
        return {
            "cross_database_feasibility": {
                db: statuses.get(db, "blocked") for db in databases
            }
        }

    return fn


def test_emits_pair_measurable_across_many_databases():
    feas = _fake_feasibility({("lactate", "mortality"): {db: "full" for db in _DBS}})
    out = generate_data_first_candidates(
        predictor_concepts=["lactate"],
        outcome_concepts=["mortality"],
        databases=_DBS,
        feasibility_fn=feas,
        min_harmonized_dbs=4,
    )
    assert len(out) == 1
    cand = out[0]
    assert isinstance(cand, DataFirstCandidate)
    assert cand.harmonized_db_count == 6
    assert cand.total_databases == 6
    assert "harmonized public databases" in cand.differentiator_note


def test_drops_pair_below_min_harmonized_threshold():
    # only 2 databases full -> below the default cross-database bar.
    feas = _fake_feasibility(
        {("lactate", "mortality"): {"aumc": "full", "eicu": "full"}}
    )
    out = generate_data_first_candidates(
        predictor_concepts=["lactate"],
        outcome_concepts=["mortality"],
        databases=_DBS,
        feasibility_fn=feas,
        min_harmonized_dbs=4,
    )
    assert out == []


def test_does_not_fabricate_pairs_with_no_availability():
    # a concept the engine cannot resolve anywhere is never emitted.
    feas = _fake_feasibility({})  # every pair -> all blocked
    out = generate_data_first_candidates(
        predictor_concepts=["imaginary_biomarker"],
        outcome_concepts=["mortality"],
        databases=_DBS,
        feasibility_fn=feas,
        min_harmonized_dbs=1,
    )
    assert out == []


def test_ranks_wider_harmonization_first_then_under_published():
    feas = _fake_feasibility(
        {
            ("wide", "mortality"): {db: "full" for db in _DBS},  # 6 DBs
            ("narrow", "mortality"): {db: "full" for db in _DBS[:4]},  # 4 DBs
        }
    )
    counts = {
        (normalize_concept_name("wide"), normalize_concept_name("mortality")): 3,
        (normalize_concept_name("narrow"), normalize_concept_name("mortality")): 2,
    }
    out = generate_data_first_candidates(
        predictor_concepts=["wide", "narrow"],
        outcome_concepts=["mortality"],
        databases=_DBS,
        feasibility_fn=feas,
        literature_counter=lambda p, o: counts[(p, o)],
        min_harmonized_dbs=4,
    )
    # wider harmonization wins regardless of the (lower) literature count.
    assert [c.predictor for c in out] == ["wide", "narrow"]


def test_under_published_ordering_within_same_harmonization():
    feas = _fake_feasibility(
        {
            ("gap", "mortality"): {db: "full" for db in _DBS},
            ("crowded", "mortality"): {db: "full" for db in _DBS},
        }
    )
    counts = {
        (normalize_concept_name("gap"), normalize_concept_name("mortality")): 2,
        (normalize_concept_name("crowded"), normalize_concept_name("mortality")): 500,
    }
    out = generate_data_first_candidates(
        predictor_concepts=["gap", "crowded"],
        outcome_concepts=["mortality"],
        databases=_DBS,
        feasibility_fn=feas,
        literature_counter=lambda p, o: counts[(p, o)],
        gap_max_hits=20,
    )
    assert [c.predictor for c in out] == ["gap", "crowded"]
    gap = out[0]
    assert gap.is_under_published is True
    assert "under-published" in gap.differentiator_note
    assert out[1].is_under_published is False


def test_unscreened_pairs_flagged_and_ranked_after_known_gaps():
    feas = _fake_feasibility(
        {
            ("known_gap", "mortality"): {db: "full" for db in _DBS},
            ("unknown", "mortality"): {db: "full" for db in _DBS},
        }
    )

    def counter(pred, out):
        if pred == normalize_concept_name("known_gap"):
            return 1
        return None  # could not screen

    out = generate_data_first_candidates(
        predictor_concepts=["known_gap", "unknown"],
        outcome_concepts=["mortality"],
        databases=_DBS,
        feasibility_fn=feas,
        literature_counter=counter,
    )
    assert [c.predictor for c in out] == ["known_gap", "unknown"]
    unknown = out[1]
    assert unknown.literature_screened is False
    assert unknown.is_under_published is False
    assert "NOT screened" in unknown.differentiator_note


def test_self_pair_is_skipped():
    feas = _fake_feasibility({("lactate", "lactate"): {db: "full" for db in _DBS}})
    out = generate_data_first_candidates(
        predictor_concepts=["lactate"],
        outcome_concepts=["lactate"],
        databases=_DBS,
        feasibility_fn=feas,
        min_harmonized_dbs=1,
    )
    assert out == []


def test_real_engine_wiring_smoke():
    # exercise the default (real dictionary-backed) availability engine; assert
    # only structural invariants so the test is stable across dictionary growth.
    out = generate_data_first_candidates(
        predictor_concepts=["lact"],
        outcome_concepts=["po2", "glu", "sofa"],
        min_harmonized_dbs=1,
        limit=10,
    )
    assert isinstance(out, list)
    for cand in out:
        assert 1 <= cand.harmonized_db_count <= cand.total_databases
        assert cand.literature_screened is False  # no counter supplied
        assert "harmonized public databases" in cand.differentiator_note


def test_module_is_a_leaf_does_not_import_idea_mining():
    src = inspect.getsource(
        importlib.import_module("easyicu.research_agent.discovery.idea_mining_data_first")
    )
    assert "import idea_mining" not in src
    assert "from .idea_mining import" not in src
