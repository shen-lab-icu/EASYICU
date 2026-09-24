"""A prespecified sensitivity refit reaches Results through one host claim."""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.research_agent.authority.manuscript_claim_policy import (
    place_scientific_claim_tokens_in_results,
)
from easyicu.research_agent.authority.scientific_claims import (
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
    scientific_claim_compilation_requested,
)


def _envelope(**overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": "easyicu.binary_sensitivity_reporting/1",
        "analysis_id": "age_restricted_cubic_spline",
        "strategy": "functional_form",
        "exposure": "aki_stage_strict",
        "outcome": "death",
        "analysis_set": "source_aware",
        "adjustment_covariates": ["age", "sex"],
        "covariate": "age",
        "effect_scale": "odds_ratio",
        "estimate": 1.2048899098108372,
        "lower": 0.5158911369841489,
        "upper": 2.814081480931846,
        "n": 775,
        "events": 84,
        **overrides,
    }


def _summary(**overrides: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "interpretation_class": "prespecified_sensitivity",
        "reportable_sensitivity_results": _envelope(**overrides),
    }


def _claim(**overrides: Any):
    [draft] = derive_scientific_claim_drafts(_summary(**overrides))
    [claim] = bind_scientific_claim_drafts(
        [draft.model_dump(mode="json")],
        step_id="age_functional_form",
        evidence_id="age_functional_form_summary",
    )
    return claim


def test_c_functional_form_refit_reads_as_an_adjusted_sensitivity_estimate() -> None:
    claim = _claim()

    assert claim.claim_ref == "age_functional_form.sensitivity_age_restricted_cubic_spline"
    assert (claim.claim_type, claim.analysis_role, claim.direction) == (
        "association", "sensitivity", "no_clear_association",
    )
    assert claim.render_reader_text() == (
        "After adjustment for age and sex, aki stage strict showed no clear association "
        "with death in the analysis set with age modelled by a restricted cubic "
        "spline (adjusted odds ratio, 1.205; 95% CI, 0.516 to 2.814)."
    )


def test_c_first_stay_refit_names_its_restriction() -> None:
    claim = _claim(
        analysis_id="first_icu_stay_only",
        strategy="first_stay",
        analysis_set="complete_case",
        covariate=None,
        estimate=1.5,
        lower=1.1,
        upper=2.0,
    )

    assert claim.claim_id == "sensitivity_first_icu_stay_only"
    assert claim.direction == "positive"
    assert claim.render_reader_text(include_estimate=False).endswith(
        "was positively associated with death in the complete case analysis set "
        "restricted to the first ICU stay of each patient (adjusted odds ratio)."
    )


def test_c_interval_below_one_is_a_negative_association() -> None:
    assert _claim(estimate=0.6, lower=0.4, upper=0.9).direction == "negative"


def test_c_a_refit_without_the_envelope_claims_nothing() -> None:
    legacy = {"status": "ok", "interpretation_class": "prespecified_sensitivity"}

    assert not scientific_claim_compilation_requested(legacy)
    assert derive_scientific_claim_drafts(legacy) == []


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"estimate": 3.0}, "interval must contain its estimate"),
        ({"events": 776}, "events exceed the analysed records"),
        ({"covariate": None}, "names the covariate it reshaped"),
        ({"strategy": "first_stay"}, "names the covariate it reshaped"),
        ({"upper": float("inf")}, "finite number"),
        ({"n": 775.0}, "valid integer"),
        ({"schema_version": "easyicu.binary_sensitivity_reporting/2"}, "schema_version"),
        ({"analysis_set": "primary_cohort"}, "analysis_set"),
        ({"p_value": 0.64}, "Extra inputs are not permitted"),
    ],
)
def test_c_an_incoherent_envelope_fails_closed(
    overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        derive_scientific_claim_drafts(_summary(**overrides))


def test_c_the_claim_is_placed_under_the_sensitivity_subsection() -> None:
    claim = _claim()
    scaffold = (
        "## Results\n\n"
        "### Primary association\n\n"
        "Primary text.\n\n"
        "### Sensitivity and subgroup analyses\n\n"
        "Sensitivity text.\n\n"
        "## Discussion\n\n"
        "Discussion text.\n"
    )

    placement = place_scientific_claim_tokens_in_results(scaffold, claims=[claim])

    assert placement.inserted_claim_refs == (claim.claim_ref,)
    primary, sensitivity = placement.scaffold.split("### Sensitivity", 1)
    assert claim.placeholder not in primary
    assert claim.placeholder in sensitivity.split("## Discussion", 1)[0]


def _executor_summary() -> dict[str, Any]:
    row = {
        "analysis_id": "age_restricted_cubic_spline",
        "strategy": "functional_form",
        "covariate": "age",
        "basis": "restricted_cubic_spline",
        "n_stays": 775,
        "n_deaths": 84,
        "odds_ratio": 1.2048899098108372,
        "ci_low": 0.5158911369841489,
        "ci_high": 2.814081480931846,
        "effect_measure": "odds_ratio",
        "primary_or_linear_covariate": 1.2000296869591074,
        "log_or_delta_vs_linear": 0.004041905995564948,
        "nonlinearity_wald_statistic": 0.7315957874687744,
        "nonlinearity_df": 1,
        "nonlinearity_p_value": 0.39236642022801993,
        "variance_estimator": "cluster_robust",
        "cluster_count": 669,
        "fit_status": "fitted",
        "note": "",
    }
    return {
        **_summary(),
        "deterministic_standard_analysis": "association_binary_sensitivity",
        "analysis_family": "association",
        "sensitivity_strategy": "functional_form",
        "sensitivity_spec_id": "age_restricted_cubic_spline",
        "analysis_rows": [row],
        "primary_reference": {
            "odds_ratio": 1.2000296869591074,
            "ci_low": 0.5142404796646424,
            "ci_high": 2.800384852087692,
            "n": 775,
        },
    }


def test_c_the_expanded_sensitivity_sentence_survives_strict_numeric_binding(
    tmp_path,
) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.manuscript_post import (
        bind_numeric_values,
        drop_untraceable_numeric_sentences,
    )

    summary = _executor_summary()
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    record = store.register_json(
        kind="statistic", description="Binary sensitivity refit",
        payload=summary, filename="summary.json",
        evidence_id="age_functional_form_summary",
        produced_by_step="age_functional_form",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="age_functional_form", evidence_id=record.evidence_id, summary=summary,
    )
    [claim] = store.scientific_claims()
    assert claim.analysis_role == "sensitivity"
    records = [{
        "step_id": "age_functional_form", "status": "ok",
        "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": record.evidence_id,
        "evidence_ids": [record.evidence_id],
    }]

    bound = store.bind_manuscript(
        "## Results\n\n### Sensitivity analyses\n\n" + claim.placeholder,
        per_step_records=records,
    )
    filtered, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=records,
    )
    _, bindings, untraced = bind_numeric_values(
        bound, evidence=store, per_step_records=records,
    )

    assert "restricted cubic spline (adjusted odds ratio, 1.205" in bound
    assert removed == [] and filtered == bound
    assert not untraced
    assert len(bindings) == 3


@pytest.mark.parametrize(
    ("overrides", "reader_ending"),
    [
        (
            dict(
                analysis_id="sofa_spline", exposure="lactate_max",
                outcome="icu_readmission", covariate="sofa",
                adjustment_covariates=["age", "sofa"],
                estimate=1.31, lower=1.05, upper=1.63, n=1200, events=150,
            ),
            "lactate max was positively associated with icu readmission in the "
            "analysis set with sofa modelled by a restricted cubic spline "
            "(adjusted odds ratio, 1.310; 95% CI, 1.050 to 1.630).",
        ),
        (
            dict(
                analysis_id="first_stay_only", strategy="first_stay",
                exposure="vasopressor_any", outcome="mort_28d", covariate=None,
                analysis_set="complete_case", adjustment_covariates=["age"],
                estimate=0.72, lower=0.55, upper=0.94, n=900, events=200,
            ),
            "vasopressor any was negatively associated with mort 28d in the "
            "complete case analysis set restricted to the first ICU stay of each "
            "patient (adjusted odds ratio, 0.720; 95% CI, 0.550 to 0.940).",
        ),
    ],
    ids=["lactate_readmission_spline", "vasopressor_mortality_first_stay"],
)
def test_c_any_binary_refit_reads_and_places_as_a_sensitivity_claim(
    overrides: dict[str, Any], reader_ending: str
) -> None:
    claim = _claim(**overrides)
    scaffold = (
        "## Results\n\n### Primary model\n\nPrimary text.\n\n"
        "### Robustness analyses\n\nRobustness text.\n\n## Discussion\n"
    )

    placement = place_scientific_claim_tokens_in_results(scaffold, claims=[claim])

    assert claim.render_reader_text().endswith(reader_ending)
    primary, robustness = placement.scaffold.split("### Robustness", 1)
    assert claim.placeholder not in primary
    assert claim.placeholder in robustness.split("## Discussion", 1)[0]
