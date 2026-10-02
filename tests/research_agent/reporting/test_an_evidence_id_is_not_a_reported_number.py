"""A step artifact's evidence id is a citation, not a reported number.

Every step-produced evidence id carries a digest
(``statistic_step_summary_3f9a0c1d2b4e5f60``).  The strict gate counted those
digits as a reported value, so a Methods sentence that cited a step artifact
was deleted as an unregistered numeric method detail, however plain its prose.
In the real runs since 9/24 that was 55 of the 112 Methods sentences the gate
counted as numeric, in every study family.  Citation identifiers are now
stripped before the result and numeric tests, as literature keys already were;
a number in the prose still needs an exact host method fact.

Generic manuscripts only; no study's values.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.manuscript_claim_policy import filter_evidence_bound_scaffold

DIGEST_ID = "statistic_step_summary_3f9a0c1d2b4e5f60"
PLAIN_ID = "model_summary"


def _no_claim(ref: str):
    return None


def _kept(scaffold: str, evidence_id: str, *, registered: bool = True) -> bool:
    result = filter_evidence_bound_scaffold(
        scaffold, resolve_claim=_no_claim, resolve_evidence=lambda ref: registered and ref == evidence_id,
    )
    return result.filtered_sentences == ()


def test_methods_prose_citing_a_step_artifact_is_kept():
    model = f"Hazard ratios were estimated with Cox proportional hazards models {{evidence:{DIGEST_ID}}}."
    software = f"The analysis code and its locked runtime are archived with the run {{evidence:{DIGEST_ID}}}."
    scaffold = (
        f"## Methods\n\n### Statistical analysis\n\n{model}\n\n"
        f"### Software and reproducibility\n\n{software}\n"
    )

    result = filter_evidence_bound_scaffold(
        scaffold, resolve_claim=_no_claim, resolve_evidence=lambda ref: ref == DIGEST_ID,
    )

    assert result.filtered_sentences == ()
    assert model in result.scaffold and software in result.scaffold


def test_a_number_in_methods_prose_still_needs_a_host_method_fact():
    numeric = f"Follow-up was capped at 90 days after the landmark {{evidence:{DIGEST_ID}}}."

    result = filter_evidence_bound_scaffold(
        f"## Methods\n\n{numeric}\n", resolve_claim=_no_claim, resolve_evidence=lambda ref: ref == DIGEST_ID,
    )

    assert result.removed_result_sentences == (numeric,)


@pytest.mark.parametrize("registered", [True, False], ids=["registered", "unregistered"])
@pytest.mark.parametrize("section", ["Methods", "Discussion"])
@pytest.mark.parametrize(
    "prose",
    [
        "Hazard ratios were estimated with Cox proportional hazards models",
        "The analysis code is archived with the run",
        "Mortality was 20%",
    ],
)
def test_a_digest_in_the_cited_id_does_not_change_the_decision(section, prose, registered):
    def scaffold(evidence_id: str) -> str:
        return f"## {section}\n\n{prose} {{evidence:{evidence_id}}}.\n"

    assert _kept(scaffold(DIGEST_ID), DIGEST_ID, registered=registered) == _kept(
        scaffold(PLAIN_ID), PLAIN_ID, registered=registered
    )


def test_a_literature_year_in_methods_is_not_a_reported_number():
    definition = "Sepsis was identified with the consensus definition [@consensus_2016]."

    result = filter_evidence_bound_scaffold(f"## Methods\n\n{definition}\n", resolve_claim=_no_claim)

    assert result.filtered_sentences == ()
    assert definition in result.scaffold
