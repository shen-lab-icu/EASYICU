"""Tests for the three-tier database-feasibility classifier."""

from __future__ import annotations

import importlib
import inspect

from easyicu.research_agent.idea_mining_feasibility_tier import (
    SourceItemIndex,
    classify_feasibility_tier,
)


def _index() -> SourceItemIndex:
    return SourceItemIndex(
        [
            {
                "itemid": 50954,
                "label": "Lactate Dehydrogenase (LD)",
                "category": "Chemistry",
                "abbrev": "",
                "table": "hosp/labevents",
            },
            {
                "itemid": 50915,
                "label": "D-Dimer",
                "category": "Hematology",
                "abbrev": "",
                "table": "hosp/labevents",
            },
            {
                "itemid": 221223,
                "label": "EEG",
                "category": "Neuro",
                "abbrev": "eeg",
                "table": "icu/chartevents",
            },
        ]
    )


class _Candidate:
    def __init__(self, predictor, outcome, rp, ro, executable):
        self.predictor_label = predictor
        self.outcome_label = outcome
        self.resolved_predictor_concept = rp
        self.resolved_outcome_concept = ro
        self.executable = executable


def test_match_returns_real_source_item_for_uncurated_lab():
    idx = _index()
    hits = idx.match("lactate dehydrogenase")
    assert hits and hits[0].itemid == 50954


def test_match_returns_empty_for_construct_absent_from_source():
    idx = _index()
    assert idx.match("procalcitonin") == []
    assert idx.match("hemoperfusion") == []


def test_match_does_not_fire_on_generic_qualifier_only():
    # "quantitative" alone must not match — only the specific token "eeg" does.
    idx = _index()
    assert idx.match("quantitative metrics") == []
    assert idx.match("quantitative EEG metrics")[0].itemid == 221223


def test_abbreviation_match():
    idx = _index()
    assert idx.match("eeg")[0].itemid == 221223


def test_tier_executable_when_candidate_executes():
    idx = _index()
    cand = _Candidate("lact", "death", "lact", "death", True)
    assert classify_feasibility_tier(cand, source_index=idx).tier == "executable"


def test_tier_t1_when_resolved_but_not_executable():
    idx = _index()
    cand = _Candidate("lact", "death", "lact", "death", False)
    result = classify_feasibility_tier(cand, source_index=idx)
    assert result.tier == "T1_reextract"


def test_tier_t2_when_unresolved_but_source_measures_it():
    idx = _index()
    cand = _Candidate("lactate dehydrogenase", "death", None, "death", False)
    result = classify_feasibility_tier(cand, source_index=idx)
    assert result.tier == "T2_new_concept"
    assert result.source_item_hits and result.source_item_hits[0].itemid == 50954


def test_tier_t3_when_unresolved_and_absent_from_source():
    idx = _index()
    cand = _Candidate("hemoperfusion", "death", None, "death", False)
    result = classify_feasibility_tier(cand, source_index=idx)
    assert result.tier == "T3_not_in_db"


def test_t3_blocker_dominates_a_t2_side():
    # predictor is buildable (T2) but outcome is absent (T3): the honest overall
    # verdict is T3 — the idea cannot execute on this database regardless.
    idx = _index()
    cand = _Candidate(
        "lactate dehydrogenase", "neurological outcome", None, None, False
    )
    result = classify_feasibility_tier(cand, source_index=idx)
    assert result.tier == "T3_not_in_db"


def test_no_source_index_yields_no_classification_path():
    # classify still runs but with no index every unresolved side is T3.
    cand = _Candidate("lactate dehydrogenase", "death", None, "death", False)
    result = classify_feasibility_tier(cand, source_index=None)
    assert result.tier == "T3_not_in_db"


def test_module_is_a_leaf_does_not_import_idea_mining():
    src = inspect.getsource(
        importlib.import_module("easyicu.research_agent.idea_mining_feasibility_tier")
    )
    assert "import idea_mining" not in src
    assert "from .idea_mining import" not in src
