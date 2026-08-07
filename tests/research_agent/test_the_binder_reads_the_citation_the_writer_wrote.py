"""The rule that resolves a repeated number has never once fired.

``_select_numeric_claim`` has a first, decisive branch: if the sentence names
its source, restrict the candidates to that evidence and its lineage. It is the
only thing that can tell one step's estimate from another's when both
registered the same value.

``_cited_evidence_ids`` fed that branch by looking for ``{evidence:<id>}``.
Measured 2026-08-01 over all 115 recorded bound manuscripts on disk: that form
appears **0** times. The writer's citations are markdown links --
``[label](evidence/<evidence_id>__<file> "sha256=...")`` -- and those appear
**541** times. So the branch has been dead in production for its whole life.

canary37 is what it cost. E1 executed end to end for the first time, every
number in the manuscript bound except one: the primary estimate, left ambiguous
across ELEVEN candidate fields (three spellings in step 06's summary, seven in
step 08's, plus the robustness panel), which blocked the manuscript. The
sentence cited step 06's summary, one clause away from the value.

Two things had to be true for the citation to be usable, and neither was:

  * the sentence window ended at the terminal period, and the citation sits
    just AFTER it -- so the window did not contain it; and
  * the window began at the previous sentence's period, so it DID contain the
    previous sentence's trailing citation, which belongs to that sentence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import NumericClaim
from easyicu.research_agent.reporting.manuscript_post import (
    _cited_evidence_ids,
    _numeric_sentence_context,
    _select_numeric_claim,
)

#: The recorded canary37 Results sentence, with its own citation after the
#: period and the PREVIOUS sentence's citation before it -- both exactly as the
#: writer emitted them.
_PREVIOUS_CITATION = (
    "[cohort_summary](evidence/table_step_artifact_bb7ddb3c781f0ae2"
    '__cohort_summary.csv "sha256=b54ae795")'
)
_OWN_CITATION = (
    "[primary_association](evidence/statistic_step_summary_b1e0b2c10d5fcb66"
    '__step_summary.json "sha256=c962ff2e")'
)
_RESULTS_LINE = (
    "The cohort denominator comprised 94,458 ICU stays. "
    + _PREVIOUS_CITATION
    + " In the primary analysis, the adjusted association estimate for the "
    "exposure and in-hospital death was 1.57, with a 95% confidence interval "
    "from 1.02 to 2.39. " + _OWN_CITATION
)
_OWN_EVIDENCE = "statistic_step_summary_b1e0b2c10d5fcb66"
_PREVIOUS_EVIDENCE = "table_step_artifact_bb7ddb3c781f0ae2"

#: The eleven candidates the run actually recorded for this one value.
_CANDIDATES = (
    ("06_primary_adjusted_mortality_association", "adjusted_effect", _OWN_EVIDENCE),
    ("06_primary_adjusted_mortality_association", "primary_estimate", _OWN_EVIDENCE),
    ("06_primary_adjusted_mortality_association", "primary_or", _OWN_EVIDENCE),
    (
        "08_missingness_robustness_replay",
        "primary_effect",
        "statistic_step_summary_5b564002597c77b8",
    ),
    (
        "08_missingness_robustness_replay",
        "primary_estimate",
        "statistic_step_summary_5b564002597c77b8",
    ),
    (
        "08_missingness_robustness_replay",
        "primary_or",
        "statistic_step_summary_5b564002597c77b8",
    ),
    (
        "08_missingness_robustness_replay",
        "robustness_panel.rows[0].point_estimate",
        "statistic_step_summary_5b564002597c77b8",
    ),
    (
        "08_missingness_robustness_replay",
        "robustness_panel.rows[1].point_estimate",
        "statistic_step_summary_5b564002597c77b8",
    ),
    (
        "08_missingness_robustness_replay",
        "robustness_rows[0].point_estimate",
        "statistic_step_summary_5b564002597c77b8",
    ),
    (
        "08_missingness_robustness_replay",
        "robustness_rows[1].point_estimate",
        "statistic_step_summary_5b564002597c77b8",
    ),
    ("robustness_panel", "primary_point_estimate", "robustness_panel"),
)
_VALUE = 1.566375890701969


def _context_for_the_estimate() -> str:
    start = _RESULTS_LINE.index("was 1.57") + len("was ")
    return _numeric_sentence_context(_RESULTS_LINE, start=start, end=start + 4)


def _recorded_candidates():
    return [
        (
            NumericClaim(
                value=_VALUE,
                canonical=_VALUE,
                evidence_id=evidence_id,
                step_id=step_id,
                source_field=field,
                tolerance=0.005,
            ),
            0.0,
        )
        for step_id, field, evidence_id in _CANDIDATES
    ]


# ---------------------------------------------------------------------------
# The citation form that actually exists
# ---------------------------------------------------------------------------


def test_a_rendered_evidence_link_is_a_citation():
    """0 of 115 recorded manuscripts use the placeholder form; 541 links do."""

    assert _cited_evidence_ids(_OWN_CITATION) == frozenset({_OWN_EVIDENCE})


def test_the_placeholder_form_still_counts():
    """Widening the reader must not drop the form it already understood."""

    assert _cited_evidence_ids("{evidence:some_record}") == frozenset({"some_record"})


def test_a_link_title_does_not_hide_the_citation():
    """The writer appends `"sha256=..."`; a space-stopping tail matches nothing.

    This is not hypothetical -- the first draft of the pattern stopped at the
    space and returned an empty set on the real manuscript.
    """

    assert '"sha256=' in _OWN_CITATION
    assert _cited_evidence_ids(_OWN_CITATION)


def test_a_non_evidence_link_is_not_a_citation():
    assert _cited_evidence_ids("[paper](https://example.org/a__b.pdf)") == frozenset()


# ---------------------------------------------------------------------------
# The window that has to contain it
# ---------------------------------------------------------------------------


def test_the_sentence_window_reaches_its_own_trailing_citation():
    """The writer puts the link after the terminal period, every time."""

    assert _OWN_EVIDENCE in _cited_evidence_ids(_context_for_the_estimate())


def test_the_window_keeps_the_citation_written_BEFORE_the_sentence_too():
    """canary39 refuted the symmetry argument on a real manuscript.

    The first version skipped the leading run of citations, reasoning that they
    belong to the previous sentence. canary39's Results paragraph reads

        [01_cohort_definition_flow](...) The source cohort comprised 94,458 ICU
        stays identified by `stay_id`. [02_table_one](...)

    -- the value's true owner is cited BEFORE it and the citation after belongs
    to the NEXT sentence. Skipping the leading run put the wrong step in scope
    and the number stayed ambiguous, blocking the manuscript again.

    Position does not identify ownership, so both neighbouring runs stay in
    scope. That is safe rather than sloppy: ``_restrict_to_cited_evidence`` only
    NARROWS the candidate set, so an extra citation costs recall and cannot by
    itself produce a wrong bind -- which the two negative tests below still
    prove.
    """

    leading = (
        "[flow](evidence/statistic_step_summary_5586117158e82833"
        '__step_summary.json "sha256=1b3e7ef1")'
        " The source cohort comprised 94,458 ICU stays. "
        "[t1](evidence/statistic_step_summary_39586ec6da69c0c2"
        '__step_summary.json "sha256=e656301c")'
    )
    start = leading.index("comprised 94,458") + len("comprised ")
    cited = _cited_evidence_ids(
        _numeric_sentence_context(leading, start=start, end=start + 6)
    )

    assert "statistic_step_summary_5586117158e82833" in cited

    # and the after-the-period case from canary37 still works
    assert _OWN_EVIDENCE in _cited_evidence_ids(_context_for_the_estimate())


def test_the_window_never_swallows_the_next_sentences_prose():
    """Only links are absorbed; the first non-link stops the walk.

    The next sentence carries its OWN citation, which is what makes this
    distinguish a non-greedy link pattern from a greedy one. A greedy tail
    runs from the first citation to the LAST `)` on the line and drags every
    word in between with it -- prose, claims and all. Mutation found exactly
    that: without the second citation here, greedy and non-greedy behave the
    same and the test proves nothing. It is also not hypothetical: the greedy
    draft of this pattern was written first and swallowed the whole line.
    """

    text = (
        _RESULTS_LINE
        + " The complete-case analysis included 1,000 records. "
        + "[replay](evidence/statistic_step_summary_5b564002597c77b8"
        '__step_summary.json "sha256=db33f381")'
    )
    start = text.index("was 1.57") + len("was ")
    context = _numeric_sentence_context(text, start=start, end=start + 4)

    assert "complete-case analysis" not in context
    assert "statistic_step_summary_5b564002597c77b8" not in _cited_evidence_ids(context)


# ---------------------------------------------------------------------------
# The value that blocked a manuscript
# ---------------------------------------------------------------------------


def test_the_primary_estimate_binds_to_the_step_its_sentence_cites():
    """canary37's exact eleven candidates, its exact sentence."""

    claim, ambiguous = _select_numeric_claim(
        candidates=_recorded_candidates(),
        context=_context_for_the_estimate(),
        previous_step_id=None,
        lineage={evidence: frozenset({evidence}) for _, _, evidence in _CANDIDATES},
    )

    assert ambiguous is False
    assert claim is not None
    assert claim.step_id == "06_primary_adjusted_mortality_association"
    assert claim.evidence_id == _OWN_EVIDENCE


def test_without_the_citation_the_same_value_stays_ambiguous():
    """The guard still fails closed; it did not become a permissive default.

    Same eleven candidates, same prose, no citation -- the host must refuse to
    guess which step's estimate the sentence means.
    """

    uncited = _RESULTS_LINE.replace(_OWN_CITATION, "").replace(_PREVIOUS_CITATION, "")
    start = uncited.index("was 1.57") + len("was ")

    claim, ambiguous = _select_numeric_claim(
        candidates=_recorded_candidates(),
        context=_numeric_sentence_context(uncited, start=start, end=start + 4),
        previous_step_id=None,
        lineage={evidence: frozenset({evidence}) for _, _, evidence in _CANDIDATES},
    )

    assert claim is None
    assert ambiguous is True


def test_citing_a_step_that_owns_none_of_the_candidates_still_blocks():
    """Reading more citations must not let a foreign step's number bind."""

    foreign = _RESULTS_LINE.replace(
        _OWN_EVIDENCE, "statistic_step_summary_ffffffffffffffff"
    )
    start = foreign.index("was 1.57") + len("was ")

    claim, ambiguous = _select_numeric_claim(
        candidates=_recorded_candidates(),
        context=_numeric_sentence_context(foreign, start=start, end=start + 4),
        previous_step_id=None,
        lineage={evidence: frozenset({evidence}) for _, _, evidence in _CANDIDATES},
    )

    assert claim is None
    assert ambiguous is True


# ---------------------------------------------------------------------------
# Reachability, on the recorded corpus
# ---------------------------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_recorded_manuscripts_use_only_the_form_that_was_unreadable():
    """The measurement that makes this a defect rather than a preference."""

    placeholder = re.compile(r"\{evidence:[^}]+\}")
    rendered = re.compile(r"\]\(evidence/[^)\s/]+?__[^)\n]*\)")
    placeholders = links = 0
    for path in _CORPUS.glob("*/*/*/*/manuscript_scaffold_bound.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        placeholders += len(placeholder.findall(text))
        links += len(rendered.findall(text))

    assert (
        placeholders == 0
    ), "the placeholder form reappeared; re-measure before trusting either reader"
    assert links > 0, "no recorded manuscript cites anything the binder can read"


def test_the_citation_walk_cannot_spin_on_a_zero_width_pattern():
    """A `while True` that trusts a regex to advance is one edit from a hang.

    Found by mutation: swapping the citation pattern for one that can match
    the empty string hung this suite instead of failing it. Production's
    pattern requires a bracketed label so it cannot match empty, but a writer
    phase that hangs is worse than one that binds nothing.
    """

    import re

    from easyicu.research_agent.reporting import manuscript_post as module

    original = module._TRAILING_CITATION_RE
    module._TRAILING_CITATION_RE = re.compile(r"\s*")
    try:
        assert module._extend_through_trailing_citations("abc", 1) == 1
    finally:
        module._TRAILING_CITATION_RE = original
