from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.manuscript_sentence_context import (
    contextual_sentence_deletion,
    has_dependent_opener,
)
from easyicu.research_agent.reporting.manuscript_post import (
    _apply_writer_evidence_repair_decisions,
)
from easyicu.research_agent.reporting.writer_repair_decision import WriterRepairDecision


def test_deleted_definition_drops_only_its_leading_dependent_chain():
    target = "The outcome was a binary endpoint."
    dependent = "It represented hospital death. This measure used the admission record."
    tail = "Age was obtained from the source."
    text = f"### Variables\n\n{target} {dependent} {tail}\n"
    actual, receipt = _apply_writer_evidence_repair_decisions(
        text,
        missing_sentences=[target],
        decisions=[WriterRepairDecision.drop(0)],
    )
    assert target not in actual
    assert "It represented" not in actual
    assert "This measure" not in actual
    assert tail in actual
    assert receipt[0]["dependent_context_drops"] == [
        "It represented hospital death.",
        "This measure used the admission record.",
    ]


@pytest.mark.parametrize("separator", ["\n\n", "\n", "\n### Next\n"])
def test_deletion_never_crosses_paragraph_or_hard_line(separator):
    target = "Rejected definition."
    text = target + separator + "It represents a different paragraph."
    result = contextual_sentence_deletion(text, 0, len(target))
    assert result.end == len(target)
    assert not result.dependent_sentences


def test_surviving_antecedent_keeps_dependent_sentence():
    target = "Rejected detail."
    text = "The outcome was death. " + target + " It used the admission record."
    start = text.index(target)
    result = contextual_sentence_deletion(text, start, start + len(target))
    assert result.end == start + len(target)
    assert not result.dependent_sentences


@pytest.mark.parametrize(
    "text",
    [
        "This study described patients.",
        "This analysis was descriptive.",
        "The outcome was hospital death.",
        "Age was recorded.",
    ],
)
def test_self_contained_openers_are_not_anaphoric(text):
    assert not has_dependent_opener(text)


def test_incomplete_dependent_sentence_is_not_guessed():
    text = "Rejected definition. It represented"
    assert contextual_sentence_deletion(text, 0, 20).end == 20


def test_citation_repair_retains_antecedent_and_dependent_sentence():
    target = "The outcome was hospital death."
    text = target + " It used the admission record."
    actual, receipt = _apply_writer_evidence_repair_decisions(
        text,
        missing_sentences=[target],
        decisions=[WriterRepairDecision.cite(0, ["source"])],
        allowed_evidence_ids=["source"],
    )
    assert target[:-1] in actual
    assert "It used the admission record." in actual
    assert "dependent_context_drops" not in receipt[0]


def test_a_later_flagged_dependent_is_accounted_for_without_fallback():
    target = "Rejected definition."
    dependent = "It represented hospital death."
    actual, receipt = _apply_writer_evidence_repair_decisions(
        target + " " + dependent,
        missing_sentences=[target, dependent],
        decisions=[
            WriterRepairDecision.drop(0),
            WriterRepairDecision.cite(1, ["source"]),
        ],
        allowed_evidence_ids=["source"],
    )
    assert not actual.strip()
    assert receipt[1]["reason_code"] == "writer_dependent_context_already_removed"
    assert receipt[1]["action"] == "drop"


def test_already_removed_dependent_cannot_hide_a_foreign_citation():
    target = "Rejected definition."
    dependent = "It represented hospital death."
    with pytest.raises(ValueError, match="registered allowed"):
        _apply_writer_evidence_repair_decisions(
            target + " " + dependent,
            missing_sentences=[target, dependent],
            decisions=[
                WriterRepairDecision.drop(0),
                WriterRepairDecision.cite(1, ["foreign"]),
            ],
            allowed_evidence_ids=["source"],
        )
