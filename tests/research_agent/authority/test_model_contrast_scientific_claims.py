from __future__ import annotations

import copy
import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.scientific_claims import derive_scientific_claim_drafts
from easyicu.research_agent.audits.model_contrast_reporting import model_contrast_reporting_findings


def _summary(exposure="oxygen_index", outcome="icu_readmission"):
    return {
        "status": "ok", "analysis_family": "robustness_sensitivity",
        "authority_kind": "signed_landmark_spline_robustness",
        "primary_effect_is_nonlinear_curve_summary": False,
        "runtime_projection_sha256": "a" * 64, "complete_case_n": 800,
        "primary_or": 2.0, "primary_ci_low": 1.8, "primary_ci_high": 2.2,
        "input_bindings": [{"evidence_id": "points", "sha256": "b" * 64, "loaded": True},
                           {"evidence_id": "linear", "sha256": "c" * 64, "loaded": True}],
        "reportable_model_contrasts": {
            "schema_version": "easyicu.model_contrast_reporting/1",
            "execution_owner": "landmark_spline_robustness_executor_v1",
            "interpretation": "descriptive_prognostic_association_not_causal",
            "runtime_projection_sha256": "a" * 64,
            "exposure": exposure, "outcome": outcome, "exposure_unit": "mmHg",
            "landmark_hours": 24,
            "population_rule": "alive_and_under_observation_at_landmark_with_valid_exposure",
            "n": 800, "events": 70, "adjustment_columns": ["age", "admission_score"],
            "confidence_level": 0.95, "interval_method": "wald_log_odds",
            "variance_estimator": "patient_cluster_robust",
            "contrasts": [
                {"kind": "spline_point", "source_evidence_id": "points", "value": 1,
                 "reference": 2, "estimate": 0.8, "lower": 0.7, "upper": 0.9},
                {"kind": "spline_point", "source_evidence_id": "points", "value": 5,
                 "reference": 2, "estimate": 2.0, "lower": 1.8, "upper": 2.2},
                {"kind": "linear_increment", "source_evidence_id": "linear", "value": 1,
                 "estimate": 1.0, "lower": 0.95, "upper": 1.1},
            ],
        },
        # Another risk set must not enter primary claims by recursive scalar selection.
        "variable_opportunity_sensitivity": {"ci_low": 8.0, "ci_high": 9.0, "n": 900},
    }


@pytest.mark.parametrize("exposure,outcome", [
    ("oxygen_index", "icu_readmission"), ("creatinine", "ventilation"),
    ("platelet_count", "hospital_mortality"),
])
def test_bounded_claims_preserve_coordinates_population_adjustment_and_roles(exposure, outcome):
    summary = _summary(exposure, outcome)
    before = copy.deepcopy(summary)
    claims = derive_scientific_claim_drafts(summary)
    assert model_contrast_reporting_findings(
        step_record={"deterministic_standard_analysis": "signed_landmark_spline_robustness"},
        step_summary=summary,
    ) == []
    assert summary == before
    assert [c.direction for c in claims] == ["negative", "positive", "no_clear_association"]
    assert [c.analysis_role for c in claims] == ["primary", "primary", "sensitivity"]
    assert [c.interval_upper for c in claims] == [0.9, 2.2, 1.1]
    for c in claims:
        assert exposure in c.exposure and c.outcome == outcome
        assert c.adjusted_for == ["age", "admission_score"]
        assert "800 complete-case records" in c.population and "24-hour landmark" in c.population
        assert "noncausal" in c.estimand
    assert "1 versus 2 mmHg" in claims[0].exposure
    assert "not a summary of the nonlinear curve" in claims[1].estimand
    assert "linear functional-form sensitivity" in claims[2].estimand


@pytest.mark.parametrize("path,value", [
    (("status",), "failed"), (("authority_kind",), "generated"),
    (("complete_case_n",), 900), (("primary_ci_high",), 9.0),
    (("primary_effect_is_nonlinear_curve_summary",), True),
    (("input_bindings", 0, "loaded"), False),
    (("reportable_model_contrasts", "runtime_projection_sha256"), "d" * 64),
    (("reportable_model_contrasts", "outcome"), ""),
    (("reportable_model_contrasts", "n"), True),
    (("reportable_model_contrasts", "events"), 801),
    (("reportable_model_contrasts", "confidence_level"), 0.9),
    (("reportable_model_contrasts", "interpretation"), "causal"),
    (("reportable_model_contrasts", "adjustment_columns"), ["age", "age"]),
    (("reportable_model_contrasts", "contrasts", 0, "reference"), None),
    (("reportable_model_contrasts", "contrasts", 0, "lower"), float("nan")),
    (("reportable_model_contrasts", "contrasts", 0, "estimate"), True),
    (("reportable_model_contrasts", "contrasts", 0, "upper"), 0.5),
    (("reportable_model_contrasts", "contrasts", 1, "reference"), 3),
    (("reportable_model_contrasts", "contrasts", 2, "source_evidence_id"), "unconsumed"),
])
def test_inconsistent_model_reporting_fails_closed(path, value):
    summary = _summary()
    node = summary
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        derive_scientific_claim_drafts(summary)


def test_legacy_projection_does_not_gain_authority_by_method_label():
    summary = _summary()
    del summary["reportable_model_contrasts"]
    assert derive_scientific_claim_drafts(summary) == []


@pytest.mark.parametrize("generation_mode", ["deterministic_standard", "llm"])
def test_model_claim_registration_is_sealed_and_reader_retains_comparison(tmp_path, generation_mode):
    summary = _summary()
    source = tmp_path / "source.json"
    source.write_text(json.dumps(summary))
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    record = store.register_file(kind="statistic", description="model projection", source_path=source,
                                 evidence_id="projection", produced_by_step="robustness",
                                 generation_mode=generation_mode)
    if generation_mode != "deterministic_standard":
        store.register_step_summary_numerics(step_id="robustness", evidence_id=record.evidence_id, summary=summary)
        assert store.scientific_claims() == []
        with pytest.raises(ValueError):
            store.register_step_summary_scientific_claims(step_id="robustness", evidence_id=record.evidence_id, summary=summary)
        return
    for _ in range(2):
        store.register_step_summary_numerics(step_id="robustness", evidence_id=record.evidence_id, summary=summary)
    claims = store.scientific_claims()
    assert len(claims) == 3
    store.enforce_evidence_bound_scaffold("## Results\n\n" + "\n\n".join(c.placeholder for c in claims))
    reader = claims[1].render_reader_text()
    for required in ("5 versus 2 mmHg", "icu readmission", "800 complete-case", "24-hour landmark",
                     "adjustment for age, admission score", "not a summary of the nonlinear curve"):
        assert required in reader
    assert "900" not in reader
    forged = copy.deepcopy(summary)
    forged["reportable_model_contrasts"]["outcome"] = "another_outcome"
    with pytest.raises(ValueError, match="registered summary bytes"):
        store.register_step_summary_numerics(step_id="robustness", evidence_id=record.evidence_id, summary=forged)


@pytest.mark.parametrize("legacy_numeric_registry", [False, True])
def test_every_authorized_model_contrast_survives_strict_numeric_binding(
    tmp_path, legacy_numeric_registry,
):
    from easyicu.research_agent.reporting.manuscript_post import (
        bind_numeric_values, drop_untraceable_numeric_sentences,
    )

    summary = _summary()
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    record = store.register_json(
        kind="statistic", description="Native model reporting projection",
        payload=summary, filename="summary.json", evidence_id="projection",
        produced_by_step="model", generation_mode="deterministic_standard",
    )
    if legacy_numeric_registry:
        # A resumed store already contains these exact values, but no OR/CI
        # identity. Re-registration must repair metadata without changing bytes.
        for i, contrast in enumerate(summary["reportable_model_contrasts"]["contrasts"]):
            for field in ("estimate", "lower", "upper"):
                store.register_numeric_claim(
                    step_id="model", evidence_id=record.evidence_id,
                    source_field=f"reportable_model_contrasts.contrasts[{i}].{field}",
                    value=str(contrast[field]), canonical=contrast[field],
                )
    before = copy.deepcopy(summary)
    store.register_step_summary_numerics(
        step_id="model", evidence_id=record.evidence_id, summary=summary,
    )
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    records = [{
        "step_id": "model", "status": "ok", "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": record.evidence_id,
        "evidence_ids": [record.evidence_id],
    }]
    scaffold = "## Results\n\n" + "\n\n".join(
        claim.placeholder for claim in store.scientific_claims()
    )
    bound = store.bind_manuscript(scaffold, per_step_records=records)
    filtered, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=records,
    )
    assert removed == []
    assert filtered == bound
    _, bindings, untraced = bind_numeric_values(
        bound, evidence=store, per_step_records=records,
    )
    assert not untraced
    assert {claim.estimand.value for claim in bindings.values() if claim.effect_scale} == {
        "point_estimate", "confidence_interval_lower", "confidence_interval_upper",
    }
    assert summary == before
    assert record.sha256 == store.get(record.evidence_id).sha256


def test_llm_model_projection_does_not_acquire_native_numeric_identity(tmp_path):
    summary = _summary()
    store = EvidenceStore(tmp_path)
    record = store.register_json(
        kind="statistic", description="Unvalidated generated projection",
        payload=summary, filename="summary.json", evidence_id="projection",
        produced_by_step="model", generation_mode="llm",
    )
    registered = store.register_step_summary_numerics(
        step_id="model", evidence_id=record.evidence_id, summary=summary,
    )
    assert not store.scientific_claims()
    assert all(
        claim.effect_scale is None for claim in registered
        if claim.source_field.startswith("reportable_model_contrasts.contrasts")
    )
