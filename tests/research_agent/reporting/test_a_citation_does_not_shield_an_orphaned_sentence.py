"""A citation after the period does not shield a newly orphaned sentence.

When a host gate deletes a Variables sentence, the same-paragraph sentences
that depend on it ("It ...", "This representation ...") go with it. Writers
often put the citation after the period, and sentence splitting gives that
citation to the next sentence, so the dependent sentence started with
audit-only markup and survived. Readers never see the markup: the paragraph
opened with "This representation ...", and the final quality audit rejected
Methods (MANUSCRIPT_VARIABLE_DEFINITION_CONTEXT_MISSING), which costs a Writer
section repair. Citation markup is now neither an antecedent nor a separator.

Synthetic Methods prose only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.manuscript_claim_policy import (
    _split_sentences,
    filter_evidence_bound_scaffold,
)
from easyicu.research_agent.contracts.manuscript_sentence_context import (
    contextual_sentence_deletion,
    has_dependent_opener,
)
from easyicu.research_agent.reporting.manuscript_post import (
    _apply_writer_evidence_repair_decisions,
    _writer_repair_target_span,
)
from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality
from easyicu.research_agent.reporting.manuscript_repair_pass import ManuscriptRepairPass
from easyicu.research_agent.reporting.writer_repair_decision import WriterRepairDecision

OPENER = "The admission severity score was taken from the first ICU day."
DEPENDENT = "This representation was analyzed as recorded and was not reinterpreted."
INDEPENDENT = "Age was recorded at ICU admission {evidence:research_context}."
TOKEN = "{evidence:research_context}"
LINK = '[research_context](evidence/research_context.json "sha256=' + "a" * 64 + '")'
CODE = "MANUSCRIPT_VARIABLE_DEFINITION_CONTEXT_MISSING"


def _variables(paragraph: str) -> str:
    return f"## Methods\n\n### Variables\n\n{paragraph}\n"


def _codes(manuscript: str) -> set[str]:
    return {finding.code for finding in audit_manuscript_quality(manuscript).findings}


def _delete_opener(text: str):
    start = text.index(OPENER)
    return contextual_sentence_deletion(text, start, start + len(OPENER))


def test_sentence_splitting_gives_a_trailing_citation_to_the_next_sentence():
    assert _split_sentences(f"{OPENER} {TOKEN} {DEPENDENT}") == [OPENER, f"{TOKEN} {DEPENDENT}"]


def test_the_orphan_is_what_the_final_audit_rejects():
    assert CODE in _codes(_variables(f"{TOKEN} {DEPENDENT} {INDEPENDENT}"))
    assert CODE not in _codes(_variables(INDEPENDENT))


@pytest.mark.parametrize("citation", [TOKEN, LINK])
def test_a_citation_after_the_period_does_not_shield_the_dependent_sentence(citation):
    text = _variables(f"{OPENER} {citation} {DEPENDENT} {INDEPENDENT}")

    deletion = _delete_opener(text)

    assert deletion.dependent_sentences == (f"{citation} {DEPENDENT}",)
    assert text[: deletion.start] + text[deletion.end :] == _variables(INDEPENDENT)


def test_a_period_inside_a_link_label_does_not_end_the_dependent_sentence():
    link = '[Fig. 1 source](evidence/figure_source.json "sha256=' + "b" * 64 + '")'
    text = _variables(f"{OPENER} {link} {DEPENDENT} {INDEPENDENT}")

    assert _delete_opener(text).dependent_sentences == (f"{link} {DEPENDENT}",)


def test_markup_before_the_deleted_opener_supplies_no_antecedent():
    text = _variables(f"{TOKEN} {OPENER} {DEPENDENT} {INDEPENDENT}")

    assert _delete_opener(text).dependent_sentences == (DEPENDENT,)


def test_an_independent_sentence_after_the_citation_is_kept():
    text = _variables(f"{OPENER} {TOKEN} Sex was recorded as charted.")

    deletion = _delete_opener(text)

    assert deletion.dependent_sentences == ()
    assert deletion.end == text.index(OPENER) + len(OPENER)


@pytest.mark.parametrize(
    ("paragraph", "expected"),
    [
        (f"{TOKEN} {DEPENDENT}", True),
        (f"{LINK} {DEPENDENT}", True),
        (f"{TOKEN} {LINK} It was recorded at admission.", True),
        (f"{TOKEN} This study used the admission record.", False),
        (f"{TOKEN} Age was recorded at admission.", False),
    ],
)
def test_the_dependent_opener_is_read_after_leading_citation_markup(paragraph, expected):
    assert has_dependent_opener(paragraph) is expected


def test_a_repair_drop_takes_the_dependent_sentence_with_it():
    text = _variables(f"{OPENER} {TOKEN} {DEPENDENT} {INDEPENDENT}")

    repaired, receipt = _apply_writer_evidence_repair_decisions(
        text, missing_sentences=[OPENER], decisions=[WriterRepairDecision.drop(0)],
    )

    assert repaired == _variables(INDEPENDENT)
    assert receipt[0]["dependent_context_drops"] == [f"{TOKEN} {DEPENDENT}"]
    assert CODE not in _codes(repaired)


def test_a_residual_strict_drop_takes_the_dependent_sentence_with_it():
    repair_pass = ManuscriptRepairPass(
        decision_provider=lambda *_args, **_kwargs: [],
        decision_applier=_apply_writer_evidence_repair_decisions,
        target_locator=_writer_repair_target_span,
    )
    text = _variables(f"{OPENER} {TOKEN} {DEPENDENT} {INDEPENDENT}")

    repaired, applied = repair_pass._deterministically_drop(text, [OPENER])

    assert repaired == _variables(INDEPENDENT)
    assert applied[0]["dependent_context_drops"] == [f"{TOKEN} {DEPENDENT}"]
    assert CODE not in _codes(repaired)


def test_the_strict_filter_takes_the_dependent_sentence_with_it():
    rejected = "The admission severity score required an increase of 2 points."
    text = _variables(f"{rejected} {TOKEN} {DEPENDENT} {INDEPENDENT}")

    filtered = filter_evidence_bound_scaffold(
        text,
        resolve_claim=lambda _ref: None,
        resolve_evidence=lambda ref: ref == "research_context",
    )

    assert rejected not in filtered.scaffold
    assert DEPENDENT not in filtered.scaffold
    assert f"{TOKEN} {DEPENDENT}" in filtered.filtered_sentences
    assert INDEPENDENT in filtered.scaffold
    assert CODE not in _codes(filtered.scaffold)
