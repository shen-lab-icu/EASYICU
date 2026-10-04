"""A disclaimer of causality is not causal language; a negated claim still is.

The causal-language scan flagged every sentence that contained "causal", so a
manuscript that correctly called its estimate an association "rather than a
causal effect" drew a major clinician comment.  The web readiness page shows
that comment as an open major scientific revision.  A negation or contrast cue
in the same clause, a few words before the bare word "causal", marks a
disclaimer.

Only that word can be disclaimed.  A verb or phrase pattern ("caused",
"attributable to", "effect of", "leads to") states a mechanism even beside a
negation: "not attributable to illness severity but to early vasopressor use"
attributes the difference to the exposure, so it is still a hit.  "and",
"but", "whereas" and "while" start a new assertion.

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
    "This non-causal estimate summarises the cohort {evidence:primary_estimate}.",
    "These associations should not be interpreted as causal.",
    "This design cannot establish a causal relationship.",
    "The hazard ratio is not a causal estimate.",
    "We estimate an association, and causal inference is not possible.",
    "A causal relationship could not be established in this cohort.",
    "The estimate is not causal but associational.",
    "We report associations instead of a causal effect.",
    "We report associations rather than interpreting them as causal.",
    "The estimate is neither causal nor generalisable.",
    # The phrase is the noun of the disclaimed word.
    "No causal effect of renal replacement therapy can be inferred from these data.",
    "These data cannot estimate the causal effect of early vasopressors.",
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


# A negation near a verb or phrase pattern does not withdraw the claim.
CLAIMS_BESIDE_A_NEGATION = (
    "The mortality difference was not attributable to illness severity but to early vasopressor use.",
    "The excess mortality is not explained by baseline severity and is attributable to the exposure.",
    "Patients with no prior dialysis had mortality attributable to the exposure.",
    "Neither age nor sex modified the effect of the exposure.",
    "The causal effect was not attenuated after adjustment.",
    "The causal effect was not significant after adjustment.",
    "Rather than being confounded the estimate is causal.",
    "The estimate is not confounded and is causal.",
    "Early vasopressors did not lead to lower mortality.",
)
# Negating the report verb leaves the reported mechanism in the sentence; the
# exempt way to say it names the word itself ("do not establish a causal
# relationship").
VERB_CLAIMS_UNDER_A_NEGATED_REPORT = (
    "These findings do not establish that the exposure causes death.",
    "The model doesn't imply that ventilation causes death.",
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


@pytest.mark.parametrize("sentence", CLAIMS_BESIDE_A_NEGATION)
def test_a_claim_beside_a_negation_is_still_a_hit(sentence):
    hits = scan_manuscript_for_causal_language(bound_manuscript=sentence, effect_labels=[ASSOCIATIONAL])

    assert [hit.severity for hit in hits] == ["warning"]


@pytest.mark.parametrize("sentence", VERB_CLAIMS_UNDER_A_NEGATED_REPORT)
def test_a_verb_claim_under_a_negated_report_is_still_a_hit(sentence):
    hits = scan_manuscript_for_causal_language(bound_manuscript=sentence, effect_labels=[ASSOCIATIONAL])

    assert [hit.severity for hit in hits] == ["warning"]


def test_a_negated_attribution_on_an_overclaimed_effect_is_an_error():
    sentence = (
        "The mortality difference was not attributable to illness severity but to early "
        "vasopressor use {evidence:primary_estimate}."
    )

    hits = scan_manuscript_for_causal_language(bound_manuscript=sentence, effect_labels=[OVERCLAIMED])

    assert [hit.severity for hit in hits] == ["error"]
