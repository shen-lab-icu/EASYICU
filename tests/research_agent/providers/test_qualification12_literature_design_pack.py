from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.literature import LiteratureBundle, build_preplan_literature_bundle
from easyicu.research_agent.planning.literature_design_authority import (
    LITERATURE_DESIGN_DIMENSIONS,
    validate_preplan_literature_design_authority,
)
from easyicu.research_agent.schema import ResearchContext


ROOT = Path(__file__).resolve().parents[3]
PACK_PATH = (
    ROOT
    / "benchmarks"
    / "meta_generalization"
    / "qualification12_literature_design_pack_20260825.json"
)


def _pack() -> dict:
    return json.loads(PACK_PATH.read_text(encoding="utf-8"))


def test_pack_covers_all_qualification_items_with_two_reviewed_sources() -> None:
    pack = _pack()
    assert [item["task_id"] for item in pack["items"]] == [
        f"MG{index:02d}" for index in range(1, 13)
    ]
    assert pack["selection_policy"]["published_effects_are_expected_answers"] is False
    for item in pack["items"]:
        bundle = LiteratureBundle.model_validate(item["bound_preplan_literature"])
        validate_preplan_literature_design_authority(bundle)
        assert len(bundle.citations) == 2
        assert len(bundle.design_evidence_cards) == 2
        dimensions = {
            evidence.dimension
            for card in bundle.design_evidence_cards
            for evidence in card.evidence
        }
        assert dimensions == set(LITERATURE_DESIGN_DIMENSIONS)


def test_exact_question_pack_preserves_reviewed_design_analogue_role() -> None:
    mg04 = next(item for item in _pack()["items"] if item["task_id"] == "MG04")
    seed = LiteratureBundle.model_validate(mg04["bound_preplan_literature"])
    context = ResearchContext(
        research_question=seed.research_question,
        cohort={
            "cohort_name": "Adult ICU patients without prior atrial fibrillation",
            "database": "miiv",
            "n_stays": 0,
        },
        variables=[],
    )

    rebuilt = build_preplan_literature_bundle(context, bound_seed=seed)

    decisions = {
        decision.citation_key: decision for decision in rebuilt.screening_decisions
    }
    assert decisions
    assert {decision.evidence_role for decision in decisions.values()} == {
        "design_analogue"
    }
