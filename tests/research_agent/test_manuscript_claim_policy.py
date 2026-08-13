from __future__ import annotations

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
