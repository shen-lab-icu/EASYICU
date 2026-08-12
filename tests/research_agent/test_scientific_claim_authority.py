from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_scientific_claim_is_exposed_through_public_contracts(ra) -> None:
    from easyicu.research_agent.authority.scientific_claims import ScientificClaim
    from easyicu.research_agent.contracts.runtime import (
        ScientificClaim as RuntimeScientificClaim,
    )

    assert ra.ScientificClaim is ScientificClaim
    assert RuntimeScientificClaim is ScientificClaim


def _store_with_association_claim(ra, tmp_path: Path):
    summary = {
        "interpretation_class": "adjusted_association",
        "exposure": "Lactate",
        "outcome": "hospital mortality",
        "effect_scale": "odds_ratio",
        "primary_estimate_interval": [1.10, 1.80],
        "analysis_set": "primary_cohort",
        "analysis_role": "primary",
        "adjustment_covariates": ["age", "sex"],
    }
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    source = tmp_path / "source_step_summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    record = store.register_file(
        kind="statistic",
        description="Typed association summary",
        source_path=source,
        evidence_id="04_association_summary",
        produced_by_step="04_association",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="04_association",
        evidence_id=record.evidence_id,
        summary=summary,
    )
    return store


def test_qualitative_result_cannot_launder_support_through_an_evidence_id(
    ra, tmp_path: Path
) -> None:
    store = _store_with_association_claim(ra, tmp_path)

    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        store.enforce_evidence_bound_scaffold(
            "SOFA was positively associated with hospital mortality "
            "{evidence:04_association_summary}."
        )

    assert "scientific claim" in str(exc_info.value).lower()
    assert exc_info.value.detail["unsupported_scientific_claim_sentences"]


@pytest.mark.parametrize(
    "sentence",
    [
        "Mortality was lower after adjustment {evidence:04_association_summary}.",
        "The groups were similar {evidence:04_association_summary}.",
        "The pattern was consistent with severity {evidence:04_association_summary}.",
        "The finding may reflect residual confounding {evidence:04_association_summary}.",
    ],
)
def test_other_qualitative_assertions_cannot_borrow_a_valid_evidence_id(
    ra, tmp_path: Path, sentence: str
) -> None:
    store = _store_with_association_claim(ra, tmp_path)

    with pytest.raises(ra.EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(sentence)


def test_writer_can_only_select_a_host_rendered_scientific_claim(
    ra, tmp_path: Path
) -> None:
    store = _store_with_association_claim(ra, tmp_path)
    token = "{claim:04_association.adjusted_association}"

    filtered, removed = store.enforce_evidence_bound_scaffold(token)
    assert removed == []
    bound = store.bind_manuscript(filtered)

    assert (
        "Lactate was positively associated with hospital mortality in the "
        "primary cohort analysis set (estimand: adjusted odds ratio" in bound
    )
    assert "04_association_summary" in bound
    assert "{claim:" not in bound


def test_scientific_claim_token_cannot_be_attached_to_unrelated_writer_prose(
    ra, tmp_path: Path
) -> None:
    store = _store_with_association_claim(ra, tmp_path)

    with pytest.raises(ra.EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            "SOFA was positively associated with hospital mortality "
            "{claim:04_association.adjusted_association}."
        )


def test_invalid_scientific_claim_contract_fails_step_publication(
    ra, tmp_path: Path
) -> None:
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    invalid_summary = {
        "interpretation_class": "adjusted_association",
        "exposure": "Lactate",
        "outcome": "mortality",
        "effect_scale": "odds_ratio",
        "primary_estimate_interval": [1.10, 1.80],
        "analysis_set": "primary_cohort",
        "analysis_role": "invented_role",
        "adjustment_covariates": ["age", "sex"],
    }
    source = tmp_path / "invalid_step_summary.json"
    source.write_text(json.dumps(invalid_summary), encoding="utf-8")
    store.register_file(
        kind="statistic",
        description="Invalid typed claim",
        source_path=source,
        evidence_id="invalid_summary",
        produced_by_step="04_association",
        generation_mode="deterministic_standard",
    )

    with pytest.raises(ValueError, match="scientific_claims"):
        store.register_step_summary_numerics(
            step_id="04_association",
            evidence_id="invalid_summary",
            summary=invalid_summary,
        )


def test_scientific_claim_authority_survives_store_reload(ra, tmp_path: Path) -> None:
    _store_with_association_claim(ra, tmp_path)

    reopened = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    claims = reopened.scientific_claims()

    assert [claim.claim_ref for claim in claims] == [
        "04_association.adjusted_association"
    ]
    assert "Lactate was positively associated" in reopened.bind_manuscript(
        claims[0].placeholder
    )


def test_scientific_claim_metadata_cannot_drift_from_registered_summary(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.authority.evidence_store import (
        EvidenceAuthorityIntegrityError,
    )

    store = _store_with_association_claim(ra, tmp_path)
    record = store.get("04_association_summary")
    assert record is not None
    record.metadata["scientific_claims"][0]["exposure"] = "SOFA"

    with pytest.raises(
        EvidenceAuthorityIntegrityError,
        match="differs from host derivation",
    ):
        store.scientific_claims()


def test_writer_digest_exposes_only_claim_tokens_for_qualitative_results(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.reporting.writer_evidence import (
        _render_writer_evidence_digest_v2,
    )

    store = _store_with_association_claim(ra, tmp_path)
    records = [
        {
            "step_id": "04_association",
            "status": "ok",
            "evidence_ids": ["04_association_summary"],
            "step_summary": {},
        }
    ]

    digest = _render_writer_evidence_digest_v2(records, evidence=store)

    assert "## host-authorized scientific claims" in digest
    assert "{claim:04_association.adjusted_association}" in digest
    assert "must use the exact placeholder" in digest


def test_runner_cannot_self_issue_scientific_claim_authority(ra, tmp_path: Path) -> None:
    summary = {
        "scientific_claims": [
            {
                "claim_id": "invented",
                "claim_type": "association",
                "exposure": "SOFA",
                "outcome": "mortality",
                "direction": "positive",
                "estimand": "odds ratio",
                "population": "the primary cohort",
                "analysis_role": "primary",
                "status": "supported",
            }
        ]
    }
    source = tmp_path / "self_issued.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    record = store.register_file(
        kind="statistic",
        description="Self-issued claim",
        source_path=source,
        evidence_id="self_issued",
        produced_by_step="04_association",
        generation_mode="llm",
    )

    with pytest.raises(ValueError, match="host-derived"):
        store.register_step_summary_numerics(
            step_id="04_association",
            evidence_id=record.evidence_id,
            summary=summary,
        )


def test_llm_generated_summary_cannot_receive_scientific_claim_authority(
    ra, tmp_path: Path
) -> None:
    summary = {
        "interpretation_class": "adjusted_association",
        "exposure": "Lactate",
        "outcome": "mortality",
        "effect_scale": "odds_ratio",
        "primary_estimate_interval": [1.10, 1.80],
        "analysis_set": "primary_cohort",
        "analysis_role": "primary",
        "adjustment_covariates": [],
    }
    source = tmp_path / "llm_summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    record = store.register_file(
        kind="statistic",
        description="LLM summary",
        source_path=source,
        evidence_id="llm_summary",
        produced_by_step="04_association",
        generation_mode="llm",
    )

    store.register_step_summary_numerics(
        step_id="04_association",
        evidence_id=record.evidence_id,
        summary=summary,
    )

    assert store.scientific_claims() == []
    assert any(
        claim.source_field == "primary_estimate_interval[0]"
        for claim in store.numeric_claims()
    )


@pytest.mark.parametrize(
    ("interval", "expected_direction"),
    [
        ([1.01, 1.50], "positive"),
        ([0.40, 0.99], "negative"),
        ([0.80, 1.20], "no_clear_association"),
    ],
)
def test_host_derives_association_direction_from_interval_against_null(
    interval: list[float], expected_direction: str
) -> None:
    from easyicu.research_agent.authority.scientific_claims import (
        derive_scientific_claim_drafts,
    )

    claims = derive_scientific_claim_drafts(
        {
            "interpretation_class": "adjusted_association",
            "exposure": "Lactate",
            "outcome": "mortality",
            "effect_scale": "odds_ratio",
            "primary_estimate_interval": interval,
            "analysis_set": "primary_cohort",
            "analysis_role": "primary",
            "adjustment_covariates": [],
        }
    )

    assert [claim.direction for claim in claims] == [expected_direction]
