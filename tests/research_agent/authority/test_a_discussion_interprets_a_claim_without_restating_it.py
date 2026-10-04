"""The Discussion interprets a host claim; it does not repeat the Results sentence.

The Writer places a claim's token alone as the Discussion's first paragraph.
Only Conclusion regions were treated as interpretation, so the Discussion token
expanded into the full result sentence, word for word the one in Results: a
survival manuscript stated its proportional-hazards decision three times.  A
Discussion token now renders the claim's bounded interpretation, like a
Conclusion token; Results and other sections are unchanged.

A stub claim stands in for any family's claim: only the two reader forms matter.
"""

from __future__ import annotations

from easyicu.research_agent.authority.manuscript_claim_policy import (
    expand_scientific_claim_tokens,
)

RESULT = "The exposure was positively associated with the outcome (adjusted hazard ratio, 1.5; 95% CI, 1.1 to 2.0)."
INTERPRETATION = "The exposure was positively associated with the outcome, as estimated by the adjusted hazard ratio."


class _Claim:
    evidence_id = "statistic_step_summary_primary"

    def render_reader_text(self, *, include_estimate: bool = True, labels=None) -> str:
        return RESULT if include_estimate else INTERPRETATION


def _expand(scaffold: str) -> str:
    expansion = expand_scientific_claim_tokens(scaffold, resolve_claim=lambda ref: _Claim())
    assert expansion.missing_claim_refs == ()
    return expansion.scaffold


def _section(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


SCAFFOLD = "\n".join([
    "# Study",
    "",
    "## Abstract",
    "",
    "**Results:** {claim:primary.association}",
    "",
    "**Conclusions:** {claim:primary.association}",
    "",
    "## Results",
    "",
    "### Primary analysis",
    "",
    "{claim:primary.association}",
    "",
    "## Discussion",
    "",
    "{claim:primary.association}",
    "",
    "The association is interpreted cautiously.",
    "",
    "### Strengths and limitations",
    "",
    "{claim:primary.association}",
    "",
    "## Conclusion",
    "",
    "{claim:primary.association}",
    "",
])


def test_results_state_the_result_and_the_discussion_interprets_it():
    expanded = _expand(SCAFFOLD)

    assert RESULT in _section(expanded, "## Results")
    discussion = _section(expanded, "## Discussion")
    assert RESULT not in discussion
    assert discussion.count(INTERPRETATION) == 2  # its subsections are Discussion too
    conclusion = _section(expanded, "## Conclusion")
    assert INTERPRETATION in conclusion and RESULT not in conclusion


def test_the_abstract_keeps_its_result_and_conclusion_forms():
    abstract = _section(_expand(SCAFFOLD), "## Abstract")

    assert f"**Results:** {RESULT}" in abstract
    assert f"**Conclusions:** {INTERPRETATION}" in abstract


def test_a_section_after_the_discussion_states_results_again():
    scaffold = SCAFFOLD.replace(
        "## Conclusion\n", "## Supplementary results\n\n{claim:primary.association}\n\n## Conclusion\n"
    )

    assert RESULT in _section(_expand(scaffold), "## Supplementary results")
