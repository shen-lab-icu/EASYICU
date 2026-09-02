""""No manuscript" is a value, not a sentence the readiness gate greps for.

The write phase used to signal it by writing an English paragraph into
markdown, and readiness recovered the signal with
``"Manuscript scaffold not generated" in manuscript_text[:300]`` plus a regex
for phrases like ``writer failed``. Ten sites wrote the sentence, six read it
back, and the manifest-comment regex was defined verbatim in two modules — so
rewording an author-facing paragraph could silently change what a gate
concluded, and no reader could tell a deliberate pause from a blocked run.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.manuscript_state import (
    NOT_GENERATED_HEADING,
    ManuscriptState,
    is_not_generated,
    read_manuscript_state,
    render_not_generated,
)


def test_a_rendered_document_round_trips_its_state() -> None:
    state = ManuscriptState.paused("stop_after_analysis_requested")
    document = render_not_generated(state, "This run stopped after analysis.")
    assert read_manuscript_state(document) == state
    assert is_not_generated(document)


def test_the_author_facing_prose_is_not_what_the_gate_reads() -> None:
    """Rewording the paragraph must not change the recovered state."""
    state = ManuscriptState.blocked("execution_gate_did_not_pass")
    first = render_not_generated(state, "One or more steps did not complete.")
    second = render_not_generated(state, "完全不同的中文说明。")
    assert read_manuscript_state(first) == read_manuscript_state(second) == state


def test_a_pause_and_a_block_are_distinguishable() -> None:
    paused = read_manuscript_state(
        render_not_generated(ManuscriptState.paused("stop_after_analysis_requested"), "x")
    )
    blocked = read_manuscript_state(
        render_not_generated(ManuscriptState.blocked("pipeline_aborted"), "x")
    )
    assert paused.kind == "paused"
    assert blocked.kind == "blocked"
    assert not paused.generated and not blocked.generated


def test_a_real_draft_carries_no_state() -> None:
    assert read_manuscript_state("# Results\n\nMortality was 12% {evidence:a}.") is None
    assert is_not_generated("# Results\n\nreal prose") is False
    assert is_not_generated("") is False


def test_a_document_written_before_the_marker_still_reads_as_not_generated() -> None:
    """The heading alone is all such a document ever carried."""
    legacy = f"{NOT_GENERATED_HEADING}\n\nPipeline aborted: gate failed.\n"
    state = read_manuscript_state(legacy)
    assert state == ManuscriptState(kind="blocked", reason_code="unspecified")


def test_an_unreadable_marker_degrades_to_the_heading_not_to_a_draft() -> None:
    corrupt = (
        f"{NOT_GENERATED_HEADING}\n\n"
        "<!-- easyicu:manuscript-state kind=produced -->\n\nbody\n"
    )
    assert read_manuscript_state(corrupt).kind == "blocked"


def test_a_heading_far_down_the_document_is_not_the_signal() -> None:
    body = "# Results\n\n" + ("x " * 800) + NOT_GENERATED_HEADING
    assert read_manuscript_state(body) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "rewritten"}, "produced, blocked, paused, or probe"),
        ({"kind": "blocked"}, "requires a reason_code"),
        ({"kind": "paused"}, "requires a reason_code"),
        ({"kind": "produced", "reason_code": "x"}, "no reason_code"),
    ],
)
def test_an_illegal_state_cannot_be_constructed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ManuscriptState(**kwargs)


def test_a_produced_state_cannot_be_rendered_as_not_generated() -> None:
    with pytest.raises(ValueError, match="blocked or paused"):
        render_not_generated(ManuscriptState.produced(), "x")


def test_readiness_reports_the_reason_code_it_recovered() -> None:
    from easyicu.research_agent.reporting.readiness import _manuscript_text_status

    document = render_not_generated(
        ManuscriptState.blocked("execution_gate_did_not_pass"),
        "Strict fail-closed policy blocked manuscript drafting.",
    )
    status = _manuscript_text_status(document)
    assert status["manuscript_text_ready"] is False
    assert any(
        "execution_gate_did_not_pass" in error
        for error in status["manuscript_text_errors"]
    )
