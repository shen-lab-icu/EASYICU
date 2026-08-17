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


def _descriptive_distribution_summary() -> dict:
    ceiling = "descriptive_unadjusted_not_causal"

    def absolute_risk(
        level_index: int,
        level: int,
        events: int,
        estimate: float,
        low: float,
        high: float,
    ) -> dict:
        return {
            "level_index": level_index,
            "level": level,
            "events": events,
            "denominator": 100,
            "estimate_pct": estimate,
            "standard_error_pct": 2.0,
            "ci_low_pct": low,
            "ci_high_pct": high,
            "confidence_level": 0.95,
            "interval_method": "patient_cluster_robust_wald",
            "covariance": "cluster_robust",
            "cluster_count": 80,
        }

    return {
        "interpretation_class": "exposure_outcome_distribution",
        "interpretation_ceiling": ceiling,
        "analysis_role": "primary",
        "analysis_set": "bound_typed_cohort",
        "cohort_n": 200,
        "exposure": "early_lactate_elevation",
        "outcome": "hospital_mortality",
        "descriptive_estimates": {
            "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
            "analysis_role": "primary",
            "analysis_set": "bound_typed_cohort",
            "interpretation_ceiling": ceiling,
            "dependence": {
                "schema_version": "easyicu.planned_dependence/1",
                "variance_estimator": "cluster_robust",
                "cluster_unit": "patient",
                "group_source": "patient_stay_id",
                "group_derivation": "prefix_before_delimiter",
                "delimiter": ":s",
            },
            "exposure_prevalence": [],
            "outcome_absolute_risks": [
                absolute_risk(0, 0, 10, 10.0, 6.080072, 13.919928),
                absolute_risk(1, 1, 30, 30.0, 26.080072, 33.919928),
            ],
            "risk_difference": {
                "reference_level_index": 0,
                "reference_level": 0,
                "comparison_level_index": 1,
                "comparison_level": 1,
                "direction": "comparison_minus_reference",
                "n": 200,
                "estimate_pct": 20.0,
                "standard_error_pct": 5.0,
                "ci_low_pct": 10.20018,
                "ci_high_pct": 29.79982,
                "confidence_level": 0.95,
                "interval_method": "linear_probability_wald",
                "covariance": "cluster_robust",
                "cluster_count": 80,
                "interpretation_ceiling": ceiling,
            },
        },
    }


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


def test_host_derives_only_descriptive_absolute_risks_and_risk_difference() -> None:
    from easyicu.research_agent.authority.scientific_claims import (
        bind_scientific_claim_drafts,
        derive_scientific_claim_drafts,
    )

    drafts = derive_scientific_claim_drafts(_descriptive_distribution_summary())

    assert [claim.claim_type for claim in drafts] == [
        "descriptive_absolute_risk",
        "descriptive_absolute_risk",
        "descriptive_risk_difference",
    ]
    assert [claim.claim_id for claim in drafts] == [
        "observed_absolute_risk_level_0",
        "observed_absolute_risk_level_1",
        "prespecified_unadjusted_risk_difference",
    ]
    claims = bind_scientific_claim_drafts(
        [draft.model_dump(mode="json") for draft in drafts],
        step_id="02_describe",
        evidence_id="02_summary",
    )
    rendered = " ".join(
        claim.render_text()
        for claim in claims
    ).lower()
    assert "observed absolute risk was 10 percent" in rendered
    assert (
        "risk difference (comparison minus reference) was 20 percentage points"
        in rendered
    )
    assert "descriptive, unadjusted, noncausal" in rendered
    assert all(word not in rendered for word in ("associated", "independent", "caused"))


def test_host_derives_counts_only_claims_without_inventing_intervals() -> None:
    from easyicu.research_agent.authority.scientific_claims import (
        derive_scientific_claim_drafts,
    )

    summary = _descriptive_distribution_summary()
    estimates = summary["descriptive_estimates"]
    estimates["dependence"] = None
    estimates["risk_difference"] = None
    for risk in estimates["outcome_absolute_risks"]:
        risk.update(
            {
                "standard_error_pct": None,
                "ci_low_pct": None,
                "ci_high_pct": None,
                "confidence_level": None,
                "interval_method": "none_counts_only",
                "covariance": "none_counts_only",
                "cluster_count": None,
            }
        )

    drafts = derive_scientific_claim_drafts(summary)

    assert [claim.claim_type for claim in drafts] == [
        "descriptive_absolute_risk",
        "descriptive_absolute_risk",
    ]
    assert "10/100 (10 percent; counts only, no confidence interval)" in (
        drafts[0].estimand
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("standard_error_pct", 0.0, "must be null for counts-only"),
        ("ci_low_pct", 1.0, "must be null for counts-only"),
        ("covariance", "binomial_independent", "counts-only authority drifted"),
    ],
)
def test_counts_only_claims_reject_interval_laundering(
    field: str, value: object, message: str
) -> None:
    from easyicu.research_agent.authority.scientific_claims import (
        derive_scientific_claim_drafts,
    )

    summary = _descriptive_distribution_summary()
    estimates = summary["descriptive_estimates"]
    estimates["dependence"] = None
    estimates["risk_difference"] = None
    for risk in estimates["outcome_absolute_risks"]:
        risk.update(
            {
                "standard_error_pct": None,
                "ci_low_pct": None,
                "ci_high_pct": None,
                "confidence_level": None,
                "interval_method": "none_counts_only",
                "covariance": "none_counts_only",
                "cluster_count": None,
            }
        )
    estimates["outcome_absolute_risks"][0][field] = value

    with pytest.raises(ValueError, match=message):
        derive_scientific_claim_drafts(summary)


def test_descriptive_claim_compiler_fails_closed_on_ceiling_or_level_drift() -> None:
    from easyicu.research_agent.authority.scientific_claims import (
        derive_scientific_claim_drafts,
    )

    wrong_ceiling = _descriptive_distribution_summary()
    wrong_ceiling["interpretation_ceiling"] = "association"
    with pytest.raises(ValueError, match="exact descriptive interpretation ceiling"):
        derive_scientific_claim_drafts(wrong_ceiling)

    wrong_level = _descriptive_distribution_summary()
    wrong_level["descriptive_estimates"]["risk_difference"][
        "comparison_level"
    ] = "1"
    with pytest.raises(ValueError, match="typed level values drifted"):
        derive_scientific_claim_drafts(wrong_level)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"estimate_pct": 12.0, "ci_low_pct": 2.0, "ci_high_pct": 22.0}, "comparison minus reference"),
        ({"n": 199}, "two absolute-risk denominators"),
        ({"cluster_count": 79}, "covariance authority"),
        ({"ci_high_pct": 35.0}, "confidence interval arithmetic"),
    ],
)
def test_descriptive_claim_compiler_rederives_the_risk_difference(
    updates: dict, message: str
) -> None:
    from easyicu.research_agent.authority.scientific_claims import (
        derive_scientific_claim_drafts,
    )

    summary = _descriptive_distribution_summary()
    summary["descriptive_estimates"]["risk_difference"].update(updates)

    with pytest.raises(ValueError, match=message):
        derive_scientific_claim_drafts(summary)


def test_descriptive_claim_names_the_exposure_observed_analysis_set() -> None:
    from easyicu.research_agent.authority.scientific_claims import (
        derive_scientific_claim_drafts,
    )

    summary = _descriptive_distribution_summary()
    analysis_set = "exposure_observed_rows_within_bound_typed_cohort"
    summary["analysis_set"] = analysis_set
    summary["descriptive_estimates"]["analysis_set"] = analysis_set

    drafts = derive_scientific_claim_drafts(summary)

    assert {draft.population for draft in drafts} == {
        "rows with observed exposure in the bound typed cohort"
    }


def test_descriptive_claims_are_digest_bound_and_writer_selectable(
    ra, tmp_path: Path
) -> None:
    summary = _descriptive_distribution_summary()
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    source = tmp_path / "descriptive_step_summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    record = store.register_file(
        kind="statistic",
        description="Typed descriptive summary",
        source_path=source,
        evidence_id="02_descriptive_summary",
        produced_by_step="02_describe",
        generation_mode="deterministic_standard",
    )

    store.register_step_summary_numerics(
        step_id="02_describe",
        evidence_id=record.evidence_id,
        summary=summary,
    )
    claims = store.scientific_claims()
    assert claims[-1].claim_ref == (
        "02_describe.prespecified_unadjusted_risk_difference"
    )
    manuscript = store.bind_manuscript(claims[-1].placeholder)
    assert "prespecified unadjusted risk difference" in manuscript
    assert "descriptive, unadjusted, noncausal" in manuscript
    assert "02_descriptive_summary" in manuscript


def test_descriptive_claim_sentence_binds_its_exact_outcome_denominator(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    summary = _descriptive_distribution_summary()
    estimates = summary["descriptive_estimates"]
    estimates["dependence"] = None
    estimates["risk_difference"] = None
    denominators = (60461, 33997)
    events = (4986, 4480)
    percentages = (8.246638, 13.177633)
    estimates["exposure_prevalence"] = [
        {
            "level_index": index,
            "level": index,
            "n": denominator,
            "denominator": sum(denominators),
            "estimate_pct": denominator / sum(denominators) * 100,
            "standard_error_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
            "confidence_level": None,
            "interval_method": "none_counts_only",
            "covariance": "none_counts_only",
            "cluster_count": None,
        }
        for index, denominator in enumerate(denominators)
    ]
    for risk, denominator, event_n, estimate_pct in zip(
        estimates["outcome_absolute_risks"],
        denominators,
        events,
        percentages,
        strict=True,
    ):
        risk.update(
            {
                "events": event_n,
                "denominator": denominator,
                "estimate_pct": estimate_pct,
                "standard_error_pct": None,
                "ci_low_pct": None,
                "ci_high_pct": None,
                "confidence_level": None,
                "interval_method": "none_counts_only",
                "covariance": "none_counts_only",
                "cluster_count": None,
            }
        )
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    source = tmp_path / "descriptive_step_summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    record = store.register_file(
        kind="statistic",
        description="Typed descriptive summary",
        source_path=source,
        evidence_id="02_descriptive_summary",
        produced_by_step="02_describe",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="02_describe",
        evidence_id=record.evidence_id,
        summary=summary,
    )
    claim = next(
        item
        for item in store.scientific_claims()
        if item.claim_id == "observed_absolute_risk_level_0"
    )
    manuscript = store.bind_manuscript(claim.placeholder)

    bound, binding_map, untraced = bind_numeric_values(
        manuscript,
        evidence=store,
        enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
    )

    assert untraced == []
    assert "observed absolute risk" in bound
    assert any(
        numeric.source_field
        == "descriptive_estimates.outcome_absolute_risks[0].denominator"
        for numeric in binding_map.values()
    )
