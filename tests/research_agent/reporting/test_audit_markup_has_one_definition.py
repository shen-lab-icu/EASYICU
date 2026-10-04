"""Audit-only markup has one definition, for contextual deletion and the reader view.

When a sentence is deleted, contextual deletion also removes the dependent
sentence that follows it ("This representation ..."), reading past audit-only
markup between them.  Its markup was evidence tokens and links only, while
the reader view also strips numeric-claim footnote markers and annotation
comments.  The binder demotes an unresolved evidence token to a comment, and
a Writer often puts that token after the period.  So after the post-binding
numeric deletion, the dependent sentence began with the comment, was not
recognized, and was left orphaned.  The contracts module now defines the
markup once, and the reader view strips the markup it defines.

Generic manuscripts only; no study's values.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore
from easyicu.research_agent.contracts.manuscript_sentence_context import (
    AUDIT_MARKUP_RE,
    has_dependent_opener,
)
from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences
from easyicu.research_agent.reporting.manuscript_surface import render_reader_manuscript

MARKUP = [
    "{evidence:lactate_summary}",
    '[Lactate summary](evidence/lactate_summary__table.csv "Lactate summary")',
    "[^claim_3]",
    "<!-- evidence missing: lactate_summary -->",
    "<!-- evidence missing:\nlactate_summary -->",
]


@pytest.mark.parametrize("markup", MARKUP)
def test_a_dependent_sentence_is_recognized_behind_any_audit_markup(markup):
    assert has_dependent_opener(f"{markup} This representation used the highest value.")
    assert not has_dependent_opener(f"{markup} This study enrolled adults.")


@pytest.mark.parametrize("markup", MARKUP)
def test_the_reader_view_strips_the_markup_deletion_reads_past(markup):
    assert AUDIT_MARKUP_RE.fullmatch(markup)
    assert render_reader_manuscript(f"Lactate was recorded {markup}.\n") == "Lactate was recorded.\n"


def test_the_post_binding_numeric_deletion_takes_its_dependent_sentence(tmp_path):
    dependent = (
        "<!-- evidence missing: lactate_summary --> "
        "This representation used the highest value in the first day."
    )
    text = (
        f"## Results\n\nThe maximum lactate was 4.2 mmol/L. {dependent}\n\n"
        "The cohort is described below.\n"
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    filtered, [removed] = drop_untraceable_numeric_sentences(
        text, evidence=EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT),
    )

    assert "This representation" not in filtered
    assert removed["dependent_context_drops"] == [dependent]
    assert "The cohort is described below." in filtered
