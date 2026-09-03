"""A citation the sentence did not make must not untrace its numbers.

MEASURED (e3 KDIGO, ``run_20260805T122839_4d85cc``, 11/11 steps, manuscript
written, blocked by its own numeric audit): the Results sentence carrying the
primary estimate reads, in the pre-binding manuscript, exactly

    {evidence:03_stage_stratified_mortality_distribution} In the adjusted
    primary analysis, stage 3 AKI versus non-stage-3 AKI was associated with
    higher odds of in-hospital death (odds ratio, 6.48; 95% CI, 6.02-6.97).

The leading citation terminates the PREVIOUS sentence. This sentence names no
source of its own. All three of its numbers were nevertheless scoped to step
03's lineage -- which registered none of them -- and the empty result was read
as "the sentence names its source and no candidate belongs to it", so they went
out ambiguous and the manuscript was blocked.

The sentence window keeps citations on both sides deliberately: the writer
emits them before a sentence as readily as after, so position alone does not
settle ownership, and for NARROWING an extra citation only costs recall. For
REFUSING it does not: at this gate a lost bind is a blocked manuscript.

Measured over 40 recorded pre-binding manuscripts, 1,627 numbers sit in a
sentence context carrying a citation: 1,284 (79%) cite within their own prose
and are unaffected; 343 (21%) carry only an inherited one.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import NumericClaim
from easyicu.research_agent.reporting.manuscript_post import (
    _select_numeric_claim,
    _sentence_cites_within_its_own_prose,
)

# Verbatim from the measured run's manuscript_scaffold.md.
INHERITED = (
    " {evidence:03_stage_stratified_mortality_distribution} In the adjusted "
    "primary analysis, stage 3 AKI versus non-stage-3 AKI was associated with "
    "higher odds of in-hospital death (odds ratio, 6.48; 95% CI, 6.02-6.97)."
)
# The same sentence, with the citation where the writer usually puts it.
OWN = (
    "In the adjusted primary analysis, stage 3 AKI versus non-stage-3 AKI was "
    "associated with higher odds of in-hospital death (odds ratio, 6.48; 95% "
    "CI, 6.02-6.97). {evidence:07_primary_adjusted_mortality_association}"
)

# Field names and value taken from the run's registered step summaries.
PRIMARY_EVIDENCE = "statistic_step_summary_c1a0b17b4c1c0001"
REPLAY_EVIDENCE = "statistic_step_summary_d69fe4ff30309fde"
FOREIGN_EVIDENCE = "statistic_step_summary_d2351eab7c058b2d"


def _claim(evidence_id: str, step_id: str, field: str) -> NumericClaim:
    return NumericClaim(
        value="6.48",
        canonical=6.4812345,
        evidence_id=evidence_id,
        step_id=step_id,
        source_field=field,
        tolerance=0.01,
    )


def _candidates():
    return [
        (_claim(PRIMARY_EVIDENCE, "07_primary_adjusted_mortality_association", f), 0.0)
        for f in ("primary_or", "primary_estimate", "adjusted_effect")
    ] + [
        (_claim(REPLAY_EVIDENCE, "10_complete_case_robustness_replay", f), 0.0)
        for f in ("primary_or", "primary_effect")
    ]


# The cited step registered none of these numbers, so its lineage covers none
# of the candidates -- exactly the measured situation.
LINEAGE = {
    "03_stage_stratified_mortality_distribution": frozenset({FOREIGN_EVIDENCE}),
    "07_primary_adjusted_mortality_association": frozenset({PRIMARY_EVIDENCE}),
    FOREIGN_EVIDENCE: frozenset({FOREIGN_EVIDENCE}),
    PRIMARY_EVIDENCE: frozenset({PRIMARY_EVIDENCE}),
    REPLAY_EVIDENCE: frozenset({REPLAY_EVIDENCE}),
}


def test_the_predicate_separates_an_inherited_citation_from_an_owned_one() -> None:
    assert _sentence_cites_within_its_own_prose(INHERITED) is False
    assert _sentence_cites_within_its_own_prose(OWN) is True


def test_an_inherited_citation_does_not_untrace_the_number() -> None:
    claim, ambiguous = _select_numeric_claim(
        candidates=_candidates(),
        context=INHERITED,
        previous_step_id=None,
        lineage=LINEAGE,
    )
    # The measured failure: this returned (None, True) and the manuscript was
    # blocked on a sentence that had cited nothing.
    assert claim is not None, "an inherited citation vetoed a number again"
    assert ambiguous is False


def test_the_sentences_own_citation_still_vetoes_a_foreign_number() -> None:
    """The guard this change must not weaken.

    A sentence that DOES name its source, over candidates none of which belong
    to it, still refuses -- otherwise a real, registered, correctly-hashed
    number binds to the wrong step.
    """

    foreign_context = (
        "In the adjusted primary analysis the odds ratio was 6.48. "
        "{evidence:03_stage_stratified_mortality_distribution}"
    )
    assert _sentence_cites_within_its_own_prose(foreign_context) is True
    claim, ambiguous = _select_numeric_claim(
        candidates=_candidates(),
        context=foreign_context,
        previous_step_id=None,
        lineage=LINEAGE,
    )
    assert claim is None
    assert ambiguous is True


def test_an_inherited_citation_that_scopes_still_reaches_the_cited_step() -> None:
    """The withdrawn veto must not cost the narrowing that already worked.

    NOT a test that narrowing is load-bearing. Two attempts to write one both
    survived deleting `candidates = scoped`: any citation strong enough to
    scope also puts its step or evidence id into the sentence context, where
    `_contextual_source_score` picks the same claim on its own. Rather than
    contrive a case, this asserts the observable contract -- an inherited
    citation covering a candidate still lands on it -- and records that the
    narrowing branch is not independently demonstrated here.
    """

    context = (
        " {evidence:07_primary_adjusted_mortality_association} The reported "
        "value was 6.48."
    )
    assert _sentence_cites_within_its_own_prose(context) is False
    claim, ambiguous = _select_numeric_claim(
        candidates=_candidates(),
        context=context,
        previous_step_id=None,
        lineage=LINEAGE,
    )
    assert claim is not None and ambiguous is False
    assert claim.step_id == "07_primary_adjusted_mortality_association"


@pytest.mark.parametrize(
    "context",
    [
        "The odds ratio was 6.48.",
        " {evidence:03_x} {evidence:04_y} The odds ratio was 6.48.",
    ],
)
def test_a_sentence_with_no_citation_of_its_own_is_never_vetoed(context: str) -> None:
    assert _sentence_cites_within_its_own_prose(context) is False
