"""Adversarial finding paraphrases independent of the policy's verb list."""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceStore,
)
from easyicu.research_agent.authority.manuscript_claim_policy import (
    filter_evidence_bound_scaffold,
)


@pytest.mark.parametrize(
    "sentence",
    [
        "The intervention caused organ injury.",
        "The intervention prevented organ injury.",
        "The intervention did not cause organ injury.",
        "Keywords: the intervention caused organ injury",
    ],
)
def test_strict_fragment_rejects_unregistered_causal_claims(tmp_path, sentence) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "The intervention abolished renal injury.",
        "The treatment spared the kidneys.",
        "The intervention did not restore organ function.",
        "These findings establish efficacy across populations.",
        "The intervention instigated organ injury.",
        "本研究证实干预导致器官损伤。",
    ],
)
@pytest.mark.parametrize(
    "prefix",
    [
        "## Results\n",
        "# Results\n### Primary outcome\n",
        "## Abstract\n**Results:**\n",
        "## Conclusion\n",
        "## Figure legends\n",
        "Figure 1. ",
        "## Results\n### ",
    ],
)
def test_finding_regions_require_authority_even_for_unrecognized_paraphrases(
    prefix, sentence
) -> None:
    result = filter_evidence_bound_scaffold(
        prefix + sentence, resolve_claim=lambda _: None
    )
    assert sentence not in result.scaffold
    assert result.unsupported_scientific_claim_sentences


@pytest.mark.parametrize(
    "suffix",
    [
        " {evidence:registered}.",
        " in 20% of patients {evidence:registered}.",
        " [@prior_2024].",
    ],
)
def test_finding_cannot_borrow_a_numeric_citation_or_literature_marker(suffix) -> None:
    sentence = "The intervention abolished organ injury" + suffix
    result = filter_evidence_bound_scaffold(
        "## Results\n" + sentence,
        resolve_claim=lambda _: None,
        resolve_evidence=lambda _: True,
    )
    assert sentence not in result.scaffold
    assert result.unsupported_scientific_claim_sentences == (sentence,)


def test_abstract_methods_and_explicit_literature_background_remain_separate() -> None:
    scaffold = (
        "## Abstract\n**Methods:**\n"
        "We fitted the prespecified regression model.\n"
        "**Results:**\nThe intervention abolished organ injury.\n"
        "## Introduction\n"
        "Prior studies reported that treatment prevented injury [@prior_2024].\n"
        "Prior studies reported harm, but our intervention prevented injury [@prior_2024].\n"
    )
    result = filter_evidence_bound_scaffold(scaffold, resolve_claim=lambda _: None)
    assert "We fitted the prespecified regression model." in result.scaffold
    assert "Prior studies reported that treatment prevented injury" in result.scaffold
    assert "our intervention" not in result.scaffold
    assert "abolished" not in result.scaffold


def test_literature_background_cannot_be_presented_as_our_results() -> None:
    sentence = "Prior studies reported that treatment prevented injury [@prior_2024]."
    result = filter_evidence_bound_scaffold(
        "## Results\n" + sentence, resolve_claim=lambda _: None
    )
    assert sentence not in result.scaffold


def test_abstract_subheadings_preserve_methods_without_releasing_results() -> None:
    scaffold = (
        "## Abstract\n### Background\nClinical uncertainty motivated this study.\n"
        "### Methods\nWe fitted the prespecified regression model.\n"
        "### Results\nThe intervention abolished organ injury.\n"
        "### Conclusions\nThese findings establish efficacy across populations.\n"
        "## Introduction\nClinical uncertainty motivated this study.\n"
    )
    result = filter_evidence_bound_scaffold(scaffold, resolve_claim=lambda _: None)
    assert "We fitted the prespecified regression model." in result.scaffold
    assert result.scaffold.count("Clinical uncertainty motivated this study.") == 2
    assert "abolished" not in result.scaffold
    assert "establish efficacy" not in result.scaffold


def test_unknown_subheading_does_not_exit_the_findings_boundary() -> None:
    result = filter_evidence_bound_scaffold(
        "## Results\n### Unexpected observations\nThe treatment spared kidneys.",
        resolve_claim=lambda _: None,
    )
    assert "Unexpected observations" not in result.scaffold
    assert "spared" not in result.scaffold


@pytest.mark.parametrize("heading", ["## Results ##", "## **Results**", "## 3. Results:"])
def test_markdown_heading_decoration_does_not_release_findings(heading: str) -> None:
    result = filter_evidence_bound_scaffold(
        f"{heading}\nThe intervention abolished organ injury.",
        resolve_claim=lambda _: None,
    )
    assert "abolished" not in result.scaffold


def test_caption_labels_remain_structural_but_asserted_caption_headings_do_not() -> None:
    result = filter_evidence_bound_scaffold(
        "## Figure legends\n### Figure 1\n"
        "### Figure 2. The intervention abolished organ injury\n",
        resolve_claim=lambda _: None,
    )
    assert "### Figure 1" in result.scaffold
    assert "abolished" not in result.scaffold


def test_registered_neutral_numeric_facts_and_organization_remain_usable() -> None:
    scaffold = (
        "## Results\n### Cohort characteristics\n"
        "This study describes baseline characteristics.\n"
        "The cohort comprised 100 stays {evidence:summary}.\n"
        "Median age was 65 years {evidence:summary}.\n"
        "Mortality was 20% {evidence:summary}.\n"
        "See Table 1 {evidence:table_one}.\n"
    )
    result = filter_evidence_bound_scaffold(
        scaffold,
        resolve_claim=lambda _: None,
        resolve_evidence=lambda ref: ref in {"summary", "table_one"},
    )
    assert result.scaffold == scaffold
    assert not result.filtered_sentences


def test_neutral_survival_metrics_and_validation_caveat_preserve_the_owner_gate() -> (
    None
):
    scaffold = (
        "## Results\n### Primary association\n"
        "Exposed RMST was 25 days {evidence:survival}.\n"
        "Comparator RMST was 23 days {evidence:survival}.\n"
        "The RMST difference was 2 days {evidence:survival}.\n"
        "## Conclusion\nIndependent validation is required [@prior_2024].\n"
    )
    result = filter_evidence_bound_scaffold(
        scaffold,
        resolve_claim=lambda _: None,
        resolve_evidence=lambda ref: ref == "survival",
    )
    assert not result.filtered_sentences
    assert result.scaffold == scaffold
    uncited = filter_evidence_bound_scaffold(
        scaffold,
        resolve_claim=lambda _: None,
        resolve_evidence=lambda _: False,
    )
    assert len(uncited.removed_result_sentences) == 3
