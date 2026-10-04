"""A variable name reads as a word of its sentence, said once.

Reader labels are written like headings ("Death by 28 days"), and both the
host claims and the label repair dropped them into sentences as they were:
"was associated with Death by 28 days".  A Writer's "patient age" became
"patient Patient age at baseline" when its key was expanded, and only a
repeated prefix of two or more words was collapsed.  A name inside a sentence
now drops its heading capital, keeping acronyms, mixed-case terms, title-case
names and eponyms, and a repeated one-word prefix collapses unless it is a
function word that English legitimately doubles.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.manuscript_claim_policy import (
    missing_scientific_claims_in_results,
)
from easyicu.research_agent.reporting.manuscript_labels import reader_claim_labels
from easyicu.research_agent.reporting.manuscript_quality import repair_reader_internal_phrases
from easyicu.research_agent.reporting.manuscript_surface import (
    collapse_repeated_label_prefix,
    in_sentence_label,
)


@pytest.mark.parametrize(
    ("label", "inside"),
    [
        ("Death by 28 days", "death by 28 days"),
        ("Patient age at baseline", "patient age at baseline"),
        ("Ninety-day mortality", "ninety-day mortality"),
        ("ICU length of stay", "ICU length of stay"),
        ("SOFA score", "SOFA score"),
        ("pH at admission", "pH at admission"),
        ("Sequential Organ Failure Assessment", "Sequential Organ Failure Assessment"),
        ("Charlson comorbidity index", "Charlson comorbidity index"),
        ("Black race", "Black race"),
        ("first lactate tertile", "first lactate tertile"),
    ],
)
def test_a_heading_capital_drops_inside_a_sentence(label, inside):
    assert in_sentence_label(label) == inside


def test_claim_names_are_in_sentence_names():
    labels = reader_claim_labels(None, {"mort_28d": "Death by 28 days", "sofa": "SOFA score"})

    assert labels == {"mort_28d": "death by 28 days", "sofa": "SOFA score"}


def test_an_expanded_key_reads_in_its_sentence_and_opens_one_capitalized():
    labels = {"mort_28d": "Death by 28 days", "age": "Patient age at baseline"}

    repaired, _ = repair_reader_internal_phrases(
        "mort_28d was common. Older age was associated with mort_28d.\n\n- age was recorded.",
        reader_display_labels=labels,
    )

    assert repaired == (
        "Death by 28 days was common. Older patient age at baseline was associated with "
        "death by 28 days.\n\n- Patient age at baseline was recorded."
    )


def test_a_repeated_one_word_prefix_collapses():
    repaired, repairs = repair_reader_internal_phrases(
        "The median patient age was higher.", reader_display_labels={"age": "Patient age at baseline"},
    )

    assert repaired == "The median patient age at baseline was higher."
    assert any(item["code"] == "MANUSCRIPT_REPEATED_LABEL_PREFIX_REMOVED" for item in repairs)


def test_a_doubled_function_word_is_english_not_a_repeated_prefix():
    text = "There was no difference in in-hospital death."

    assert collapse_repeated_label_prefix(text, ["In-hospital death"]) == (text, ())


class _Claim:
    claim_ref = "primary_model.association"

    def render_reader_text(self, *, include_estimate: bool = True, labels=None) -> str:
        return "death by 28 days was more frequent after exposure (odds ratio, 1.50; 95% CI, 1.10 to 2.00)."


def test_a_claim_opening_its_sentence_is_still_found_in_results():
    manuscript = (
        "## Results\n\nDeath by 28 days was more frequent after exposure "
        "(odds ratio, 1.50; 95% CI, 1.10 to 2.00).\n\n## Discussion\n\nText.\n"
    )

    assert missing_scientific_claims_in_results(manuscript, claims=[_Claim()]) == ()
