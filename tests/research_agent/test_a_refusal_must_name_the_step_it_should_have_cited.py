"""A miscitation must be told apart from a tie and from an orphan number.

MEASURED (e1 sepsis 10/10 steps and e3 KDIGO 11/11 steps, both manuscripts
written, both blocked by their own numeric audit): every remaining marker on
both manuscripts was the cohort size 94,458, and all four were the same
failure -- a sentence citing a step that registered no such value:

    e1  cited 00_probe   owned by 02_table_one_by_sepsis3_status, 03_..., 04_...
    e3  cited table_one  owned by 02_table_one_by_kdigo_stage,     03_..., 04_...

Both refusals were correct: ``00_probe``'s two evidence files contain no 94458
at all. What was missing is that nothing said so. Three unrelated failures --
nobody owns the number, several owners tie for it, and the sentence cited the
wrong owner -- all surfaced as "Manuscript numeric claims disagree with
registered step_summary values", which names no sentence, no citation and no
owner. Only the third can name its own fix, and it is the one that was
happening.
"""

from __future__ import annotations

import re

from easyicu.research_agent.authority.evidence_store import NumericClaim
from easyicu.research_agent.reporting.manuscript_post import _miscitation_detail

COHORT_EVIDENCE = "statistic_step_summary_cohort0001"
AUDIT_EVIDENCE = "statistic_step_summary_audit0002"
PROBE_EVIDENCE = "statistic_probe_summary_3febb018"

LINEAGE = {
    "00_probe": frozenset({PROBE_EVIDENCE}),
    "02_table_one_by_sepsis3_status": frozenset({COHORT_EVIDENCE}),
    "04_measurement_missingness_audit": frozenset({AUDIT_EVIDENCE}),
    PROBE_EVIDENCE: frozenset({PROBE_EVIDENCE}),
    COHORT_EVIDENCE: frozenset({COHORT_EVIDENCE}),
    AUDIT_EVIDENCE: frozenset({AUDIT_EVIDENCE}),
}


def _claim(evidence_id: str, step_id: str, field: str) -> NumericClaim:
    return NumericClaim(
        value="94,458",
        canonical=94458.0,
        evidence_id=evidence_id,
        step_id=step_id,
        source_field=field,
        tolerance=0.0,
    )


def _candidates():
    return [
        (_claim(COHORT_EVIDENCE, "02_table_one_by_sepsis3_status", "cohort_n"), 0.0),
        (
            _claim(AUDIT_EVIDENCE, "04_measurement_missingness_audit", "n_total"),
            0.0,
        ),
    ]


# Verbatim from e1's pre-binding manuscript.
MISCITING = (
    "The operational denominator comprised 94,458 ICU stays represented in the "
    "supplied cohort definition {evidence:00_probe}."
)


def test_a_miscitation_names_what_was_cited_and_who_owns_the_value() -> None:
    detail = _miscitation_detail(_candidates(), context=MISCITING, lineage=LINEAGE)
    assert detail is not None
    assert detail["cited"] == ["00_probe"]
    assert detail["owned_by"] == [
        "02_table_one_by_sepsis3_status",
        "04_measurement_missingness_audit",
    ]


def test_a_correctly_cited_value_is_not_a_miscitation() -> None:
    context = (
        "The cohort comprised 94,458 ICU stays "
        "{evidence:02_table_one_by_sepsis3_status}."
    )
    assert _miscitation_detail(_candidates(), context=context, lineage=LINEAGE) is None


def test_an_uncited_sentence_is_not_a_miscitation() -> None:
    """A tie with nothing cited is a different failure and must stay one."""

    context = "The operational denominator comprised 94,458 ICU stays."
    assert _miscitation_detail(_candidates(), context=context, lineage=LINEAGE) is None


def test_an_unresolvable_placeholder_is_not_a_miscitation() -> None:
    """It never scoped anything, so it cannot have caused the refusal."""

    context = "The cohort comprised 94,458 stays {evidence:step_that_never_ran}."
    assert _miscitation_detail(_candidates(), context=context, lineage=LINEAGE) is None


def test_a_value_nobody_registered_is_not_a_miscitation() -> None:
    assert _miscitation_detail([], context=MISCITING, lineage=LINEAGE) is None


def test_the_marker_carries_both_sides_of_the_fix() -> None:
    """The emitted comment must be readable without the evidence store."""

    from easyicu.research_agent.reporting import manuscript_post

    source = manuscript_post.bind_numeric_values.__code__.co_consts
    rendered = " ".join(str(item) for item in source if isinstance(item, str))
    assert "MISCITED:" in rendered
    assert ":cited=[" in rendered and ":owned_by=[" in rendered
    # The three refusals must stay distinguishable in the output, not collapse
    # back into one marker.
    assert "AMBIGUOUS:" in rendered and "UNTRACED:" in rendered


def test_the_three_refusals_use_three_distinct_markers() -> None:
    markers = {"MISCITED", "AMBIGUOUS", "UNTRACED"}
    from easyicu.research_agent.reporting import manuscript_post

    text = "".join(
        str(item)
        for item in manuscript_post.bind_numeric_values.__code__.co_consts
        if isinstance(item, str)
    )
    found = {m for m in markers if re.search(rf"<!-- {m}:", text)}
    assert found == markers, f"missing distinct refusal markers: {markers - found}"
