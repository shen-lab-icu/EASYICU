from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.manuscript_claim_policy import (
    expand_scientific_claim_tokens,
    filter_evidence_bound_scaffold,
)
from easyicu.research_agent.authority.scientific_claims import ScientificClaim


def _claim() -> ScientificClaim:
    return ScientificClaim(
        claim_id="adjusted_association",
        claim_type="association",
        exposure="Lactate",
        outcome="hospital mortality",
        direction="positive",
        estimand="adjusted odds ratio",
        population="the primary cohort",
        analysis_role="primary",
        status="supported",
        adjusted_for=["age", "sex"],
        step_id="04_association",
        evidence_id="04_association_summary",
    )


def _resolver(ref: str):
    claim = _claim()
    return claim if ref == claim.claim_ref else None


def test_policy_accepts_only_a_complete_known_claim_token() -> None:
    result = filter_evidence_bound_scaffold(
        "{claim:04_association.adjusted_association}\n"
        "Lactate was higher {claim:04_association.adjusted_association}.",
        resolve_claim=_resolver,
    )

    assert result.scaffold == "{claim:04_association.adjusted_association}\n"
    assert result.filtered_sentences == (
        "Lactate was higher {claim:04_association.adjusted_association}.",
    )
    assert result.unsupported_scientific_claim_sentences == result.filtered_sentences


def test_policy_preserves_metadata_but_filters_uncited_numeric_result() -> None:
    result = filter_evidence_bound_scaffold(
        "Data availability: generated scripts are available.\n"
        "Mortality was 20%.",
        resolve_claim=_resolver,
    )

    assert "Data availability" in result.scaffold
    assert "Mortality was 20%" not in result.scaffold
    assert result.removed_result_sentences == ("Mortality was 20%.",)


def test_policy_keeps_registered_numeric_fact_for_later_provenance_binding() -> None:
    sentence = "Mortality was 20% {evidence:outcome_rate}."

    result = filter_evidence_bound_scaffold(
        sentence,
        resolve_claim=_resolver,
        resolve_evidence=lambda ref: ref == "outcome_rate",
    )

    assert result.scaffold == sentence + "\n"
    assert result.filtered_sentences == ()


def test_policy_rejects_numeric_fact_with_unregistered_evidence() -> None:
    sentence = "Mortality was 20% {evidence:unregistered}."

    result = filter_evidence_bound_scaffold(
        sentence,
        resolve_claim=_resolver,
        resolve_evidence=lambda _ref: False,
    )

    assert result.scaffold == "\n"
    assert result.removed_result_sentences == (sentence,)


def test_registered_evidence_does_not_bypass_qualitative_claim_authority() -> None:
    sentence = "Patients had higher mortality {evidence:outcome_rate}."

    result = filter_evidence_bound_scaffold(
        sentence,
        resolve_claim=_resolver,
        resolve_evidence=lambda ref: ref == "outcome_rate",
    )

    assert result.scaffold == "\n"
    assert result.unsupported_scientific_claim_sentences == (sentence,)


def test_exact_literature_key_keeps_only_nonnumeric_background_context() -> None:
    context = "Sepsis-3 mortality remains clinically important [@paper_2024]."
    numeric = "Prior mortality was 20% [@paper_2024]."

    kept = filter_evidence_bound_scaffold(context, resolve_claim=_resolver)
    blocked = filter_evidence_bound_scaffold(numeric, resolve_claim=_resolver)

    assert kept.scaffold == context + "\n"
    assert blocked.scaffold == "\n"


@pytest.mark.parametrize(
    "sentence",
    [
        "Patients experienced excess mortality {evidence:unrelated_log}.",
        "Patients had higher mortality {evidence:unrelated_log}.",
    ],
)
def test_strict_policy_rejects_assertions_with_arbitrary_evidence_tokens(
    tmp_path, sentence
) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)

    with pytest.raises(EvidenceEnforcementError) as blocked:
        store.enforce_evidence_bound_scaffold(sentence)

    detail = blocked.value.detail
    assert sentence in (
        detail["removed_sentences"]
        + detail["unsupported_scientific_claim_sentences"]
    )


def test_policy_preserves_provenance_metadata_with_evidence_token() -> None:
    sentence = (
        "Data availability: The reproducibility envelope and SHA-256 digest are "
        "available {evidence:run_manifest}."
    )

    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert result.scaffold == sentence + "\n"
    assert result.filtered_sentences == ()


def test_policy_ignores_numeric_literature_key_but_not_numeric_result() -> None:
    context = (
        "The declared clinical framework used an exact run-bound source "
        "[@paper_2024]."
    )
    result = filter_evidence_bound_scaffold(context, resolve_claim=_resolver)
    blocked = filter_evidence_bound_scaffold(
        "Mortality was 20% [@paper_2024].",
        resolve_claim=_resolver,
    )

    assert result.scaffold == context + "\n"
    assert blocked.scaffold == "\n"
    assert blocked.removed_result_sentences == (
        "Mortality was 20% [@paper_2024].",
    )


def test_policy_preserves_scientific_noun_phrases_in_keyword_metadata() -> None:
    sentence = "Keywords: survival benefit, protective factors, ICU mortality"

    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert result.scaffold == sentence + "\n"
    assert result.filtered_sentences == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "Treatment conferred a survival benefit.",
        "The exposure was protective against mortality.",
        "The intervention was harmful.",
        "Treated patients fared better.",
        "The exposure adversely affected outcomes.",
        "Treatment conferred benefit.",
        "The exposure was linked to harm.",
        "The findings suggested a benefit.",
    ],
)
def test_policy_rejects_common_qualitative_scientific_assertions(sentence) -> None:
    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert sentence not in result.scaffold
    assert result.unsupported_scientific_claim_sentences == (sentence,)


def test_policy_applies_qualitative_claim_gate_to_markdown_headings() -> None:
    result = filter_evidence_bound_scaffold(
        "## Treatment conferred a survival benefit",
        resolve_claim=_resolver,
    )

    assert "Treatment conferred" not in result.scaffold
    assert result.unsupported_scientific_claim_sentences == (
        "Treatment conferred a survival benefit",
    )


def test_policy_preserves_versioned_clinical_term_in_structural_title() -> None:
    title = "# Retrospective ICU Study of Sepsis-3 and In-Hospital Mortality"

    result = filter_evidence_bound_scaffold(title, resolve_claim=_resolver)

    assert result.scaffold == title + "\n"
    assert result.filtered_sentences == ()


def test_versioned_term_does_not_hide_assertive_numeric_title() -> None:
    title = "# Sepsis-3 mortality was 20%"

    result = filter_evidence_bound_scaffold(title, resolve_claim=_resolver)

    assert result.scaffold == "\n"
    assert result.filtered_sentences


@pytest.mark.parametrize(
    "heading",
    [
        "## Treatment yielded better outcomes",
        "## The intervention was beneficial",
    ],
)
def test_policy_rejects_equivalent_unsupported_result_headings(heading) -> None:
    result = filter_evidence_bound_scaffold(heading, resolve_claim=_resolver)

    assert result.scaffold == "\n"
    assert result.filtered_sentences


@pytest.mark.parametrize(
    "sentence",
    [
        "Data availability: Scripts are available, and treatment conferred a survival benefit.",
        "Data availability: Patients experienced excess mortality.",
        "Generated scripts are available while exposed patients fared worse.",
        "Funding: The exposure was harmful to patients.",
    ],
)
def test_metadata_forms_cannot_hide_scientific_clauses(tmp_path, sentence) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)

    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "See {Claim:04_association.adjusted_association}.",
        "See {claim:04_association.adjusted_association.",
        "See {claim:adjusted_association}.",
        "See {{claim:04_association.adjusted_association}}.",
        "See {Evidence:run_manifest}.",
        "See {evidence:run_manifest.",
        "See {evidence:run_manifest}}.",
        "See {evidence:}.",
    ],
)
def test_strict_policy_rejects_malformed_authority_placeholders(
    tmp_path, sentence
) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)

    with pytest.raises(EvidenceEnforcementError) as blocked:
        store.enforce_evidence_bound_scaffold(sentence)

    assert blocked.value.detail["unsupported_scientific_claim_sentences"] == [
        sentence
    ]


def test_claim_expansion_preserves_markdown_prefix_and_binds_evidence() -> None:
    result = expand_scientific_claim_tokens(
        "> {claim:04_association.adjusted_association}",
        resolve_claim=_resolver,
    )

    assert result.scaffold.startswith("> After adjustment for age, sex")
    assert "{evidence:04_association_summary}" in result.scaffold
    assert result.missing_claim_refs == ()
    assert result.malformed_sentences == ()


def test_claim_expansion_reports_missing_and_malformed_tokens() -> None:
    result = expand_scientific_claim_tokens(
        "{claim:99_missing.adjusted_association}\n"
        "Prose {claim:04_association.adjusted_association}.",
        resolve_claim=_resolver,
    )

    assert result.missing_claim_refs == ("99_missing.adjusted_association",)
    assert result.malformed_sentences == (
        "Prose {claim:04_association.adjusted_association}.",
    )
