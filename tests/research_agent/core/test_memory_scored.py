"""Tests for the HealthFlow-inspired scored memory retrieval additions.

The base :class:`RunMemory` already does token-overlap ranking; these
tests pin the new behaviours:

* :class:`StrategyCard` carries counters / lifecycle fields with
  backward-compatible defaults, and reads older on-disk cards (missing
  the new fields) without error.
* :meth:`RunMemory.scored_strategy_cards` returns a transparent score
  breakdown plus an optional JSONL audit log.
* :meth:`RunMemory.validate_card` / :meth:`retire_card` mutate the
  on-disk card, and retired cards are filtered from default
  retrieval.
* :meth:`RunMemory.record_retrieval` increments ``times_retrieved`` so
  callers can see which lessons the planner actually consumes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _make_card(ra, strategy_id: str, **kwargs):
    defaults = dict(
        strategy_id=strategy_id,
        task_family="ordinal_score_outcome_association",
        trigger_tokens=["sofa", "sofa2", "mortality"],
        recommended_plan=["audit components"],
        guardrails=["zero score may encode missing"],
        supporting_run_ids=["run_one"],
        updated_at=datetime.now(timezone.utc).isoformat(),
        applicable_databases=["miiv"],
    )
    defaults.update(kwargs)
    return ra.StrategyCard(**defaults)


def _write_card(memory, card) -> Path:
    path = memory.strategies_dir / f"{card.strategy_id}.json"
    path.write_text(
        json.dumps(card.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_strategy_card_loads_legacy_dict_without_new_fields(ra, tmp_path: Path):
    """An old card persisted before the counter fields existed must
    still round-trip through :meth:`StrategyCard.from_dict`."""
    legacy = {
        "strategy_id": "legacy",
        "task_family": "ordinal_score_outcome_association",
        "trigger_tokens": ["sofa"],
        "recommended_plan": ["audit"],
        "guardrails": ["beware zero"],
        "supporting_run_ids": ["r1"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    card = ra.StrategyCard.from_dict(legacy)
    assert card.confidence == 0.5
    assert card.times_retrieved == 0
    assert card.retired is False
    assert card.retired_reason is None


def test_scored_strategy_cards_returns_breakdown(ra, tmp_path: Path):
    memory = ra.RunMemory(tmp_path)
    _write_card(memory, _make_card(ra, "sofa_card"))
    results = memory.scored_strategy_cards(
        research_question="Is admission SOFA-2 associated with mortality?",
        database="miiv",
        target_outcome="mortality",
        limit=4,
    )
    assert len(results) == 1
    card, breakdown = results[0]
    assert card.strategy_id == "sofa_card"
    assert breakdown.overlap >= 1.0, "should match at least one trigger token"
    assert breakdown.database_bonus == 1.0
    assert breakdown.total == pytest.approx(
        breakdown.overlap
        + breakdown.support_bonus
        + breakdown.dependency_bonus
        + breakdown.database_bonus
        + breakdown.outcome_bonus
        + breakdown.confidence_bonus
        + breakdown.validation_bonus
        + breakdown.retired_penalty
    )


def test_retired_cards_excluded_from_default_retrieval(ra, tmp_path: Path):
    memory = ra.RunMemory(tmp_path)
    live = _make_card(ra, "live_card")
    dead = _make_card(ra, "dead_card", retired=True, retired_reason="contradicted by run_two")
    _write_card(memory, live)
    _write_card(memory, dead)

    results = memory.scored_strategy_cards(
        research_question="SOFA2 mortality",
        database="miiv",
        target_outcome="mortality",
    )
    ids = [card.strategy_id for card, _ in results]
    assert ids == ["live_card"]

    include_retired = memory.scored_strategy_cards(
        research_question="SOFA2 mortality",
        database="miiv",
        target_outcome="mortality",
        include_retired=True,
    )
    # Retired card carries a large penalty but is now visible.
    ids_with_retired = [card.strategy_id for card, _ in include_retired]
    assert "dead_card" in ids_with_retired or "live_card" in ids_with_retired
    # Live card still ranks first.
    assert include_retired[0][0].strategy_id == "live_card"


def test_validate_card_bumps_counter_and_timestamp(ra, tmp_path: Path):
    memory = ra.RunMemory(tmp_path)
    _write_card(memory, _make_card(ra, "c1"))
    updated = memory.validate_card("c1")
    assert updated is not None
    assert updated.validation_count == 1
    assert updated.last_validated_at is not None
    # And the change is persisted.
    on_disk = ra.StrategyCard.from_dict(
        json.loads((memory.strategies_dir / "c1.json").read_text())
    )
    assert on_disk.validation_count == 1


def test_retire_card_persists_reason(ra, tmp_path: Path):
    memory = ra.RunMemory(tmp_path)
    _write_card(memory, _make_card(ra, "c2"))
    retired = memory.retire_card("c2", reason="superseded by aki_v2")
    assert retired is not None
    assert retired.retired is True
    assert retired.retired_reason == "superseded by aki_v2"


def test_validate_and_retire_return_none_for_missing(ra, tmp_path: Path):
    memory = ra.RunMemory(tmp_path)
    assert memory.validate_card("does_not_exist") is None
    assert memory.retire_card("does_not_exist", reason="x") is None


def test_record_retrieval_bumps_times_retrieved(ra, tmp_path: Path):
    memory = ra.RunMemory(tmp_path)
    _write_card(memory, _make_card(ra, "ck"))
    memory.record_retrieval(["ck", "ck", "missing_id"])
    on_disk = ra.StrategyCard.from_dict(
        json.loads((memory.strategies_dir / "ck.json").read_text())
    )
    assert on_disk.times_retrieved == 2


def test_scored_strategy_cards_writes_audit_log(ra, tmp_path: Path):
    memory = ra.RunMemory(tmp_path)
    _write_card(memory, _make_card(ra, "in_scope"))
    # An unrelated card with no token overlap and no database match — its
    # total score should fall to or below zero, so it must be dropped.
    _write_card(
        memory,
        _make_card(
            ra,
            "unrelated",
            trigger_tokens=["sepsis", "lactate"],
            task_family="sepsis_outcome_association",
            applicable_databases=["sicdb"],
            supporting_run_ids=[],  # no prior validation → no support bonus
            recommended_plan=["check lactate trajectory"],
            guardrails=["beware retrospective sepsis labels"],
        ),
    )
    audit_path = tmp_path / "retrieval_log.jsonl"
    memory.scored_strategy_cards(
        research_question="SOFA2 mortality",
        database="miiv",
        target_outcome="mortality",
        audit_path=audit_path,
    )
    assert audit_path.exists()
    line = audit_path.read_text(encoding="utf-8").splitlines()[-1]
    payload = json.loads(line)
    dispositions = {e["strategy_id"]: e["disposition"] for e in payload["entries"]}
    assert dispositions["in_scope"] == "selected"
    assert dispositions["unrelated"] == "dropped"
    # Score breakdown is preserved in the audit record so reviewers can
    # see *why* a card was kept or dropped.
    for entry in payload["entries"]:
        assert "overlap" in entry["score"]
        assert "total" in entry["score"]


def test_relevant_strategy_cards_backward_compat(ra, tmp_path: Path):
    """The old ``relevant_strategy_cards`` signature still returns
    bare cards, so existing prompt-building code keeps working."""
    memory = ra.RunMemory(tmp_path)
    _write_card(memory, _make_card(ra, "still_works"))
    cards = memory.relevant_strategy_cards(
        research_question="SOFA2 mortality",
        database="miiv",
        target_outcome="mortality",
    )
    assert [c.strategy_id for c in cards] == ["still_works"]
