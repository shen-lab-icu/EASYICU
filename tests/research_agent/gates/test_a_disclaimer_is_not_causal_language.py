"""A disclaimer of causality is not causal language.

The causal-language scan flagged every sentence that contained "causal", so a
manuscript that correctly called its estimate an association "rather than a
causal effect" drew a major clinician comment.  The web readiness page shows
that comment as an open major scientific revision.  A negation or contrast cue
in the same clause, a few words before the causal word, now marks a
disclaimer; a causal claim is still found, also next to a disclaimer.

Generic sentences only; no study's values.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.review.causal_audit import (
    EffectLabel,
    scan_manuscript_for_causal_language,
)

ASSOCIATIONAL = EffectLabel(
    evidence_id="primary_estimate",
    artefact_path="primary.csv",
    estimand="hazard_ratio",
    label="associational",
    rationale="default",
)
OVERCLAIMED = EffectLabel(
    evidence_id="primary_estimate",
    artefact_path="primary.csv",
    estimand="risk_difference",
    label="causal_overclaimed",
    rationale="missing supports",
    identification_strategy="iptw",
    missing_supports=["dag"],
)

DISCLAIMERS = (
    "The estimate describes the association, not causal interpretation {evidence:primary_estimate}.",
    "We report associations rather than a causal effect {evidence:primary_estimate}.",
    "This observational design cannot support causal inference.",
    "The hazard ratio should not be interpreted as causal {evidence:primary_estimate}.",
    "The estimate was summarised without interpreting the exposure as a causal factor.",
    "Causal inference is not possible from routinely collected records.",
    "Causal conclusions cannot be drawn from this cohort.",
    "These findings do not establish that the exposure causes death.",
    "This non-causal estimate summarises the cohort {evidence:primary_estimate}.",
    "The model doesn't imply that ventilation causes death.",
)
CLAIMS = (
    "Sepsis causes acute kidney injury.",
    "The exposure caused a higher mortality {evidence:primary_estimate}.",
    "This is a causal effect of early vasopressors {evidence:primary_estimate}.",
    "Mortality was attributable to delayed antibiotics.",
    "Not only does sepsis cause organ failure, it also prolongs ventilation.",
    "There is no doubt that hypotension caused the injury.",
    "Without doubt the effect of the drug was large {evidence:primary_estimate}.",
    "Although no trial exists, the drug causes harm.",
    "The estimate is not causal; sepsis causes kidney injury.",
    "The estimate is not causal, but the causal pathway runs through sedation.",
)


@pytest.mark.parametrize("label", [ASSOCIATIONAL, OVERCLAIMED], ids=["associational", "overclaimed"])
@pytest.mark.parametrize("sentence", DISCLAIMERS)
def test_a_disclaimer_is_not_a_hit(sentence, label):
    assert scan_manuscript_for_causal_language(bound_manuscript=sentence, effect_labels=[label]) == []


@pytest.mark.parametrize("sentence", CLAIMS)
def test_a_causal_claim_is_still_a_hit(sentence):
    hits = scan_manuscript_for_causal_language(bound_manuscript=sentence, effect_labels=[ASSOCIATIONAL])

    assert [hit.severity for hit in hits] == ["warning"]


def test_a_claim_on_an_overclaimed_effect_is_still_an_error():
    hits = scan_manuscript_for_causal_language(
        bound_manuscript="The exposure caused a higher mortality {evidence:primary_estimate}.",
        effect_labels=[OVERCLAIMED],
    )

    assert [hit.severity for hit in hits] == ["error"]


def test_a_cue_in_an_earlier_clause_does_not_disclaim_a_later_claim():
    sentence = "Without adjustment for severity, the exposure caused excess deaths {evidence:primary_estimate}."

    assert len(scan_manuscript_for_causal_language(bound_manuscript=sentence, effect_labels=[ASSOCIATIONAL])) == 1


def test_a_cue_many_words_before_does_not_disclaim_the_word():
    sentence = (
        "No single registry recorded every admission of the many older adults "
        "who were in the end shown to have caused the outbreak."
    )

    assert len(scan_manuscript_for_causal_language(bound_manuscript=sentence, effect_labels=[])) == 1
