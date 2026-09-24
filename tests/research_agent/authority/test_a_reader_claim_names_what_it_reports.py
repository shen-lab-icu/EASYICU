"""A reader-facing claim names the contrast it reports and reads its numbers plainly.

An exposure with three or more levels has one odds ratio per contrast, so "the"
odds ratio must say which contrast it is.  Numbers read at one precision per
kind, a covariate list reads as a list, and a counts-only frequency reads as a
frequency.  The study here is a lactate tertile and ICU readmission.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.claim_coordinates import (
    contrast_exposure_coordinate,
)
from easyicu.research_agent.authority.scientific_claims import (
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.authority.sensitivity_scientific_claims import (
    derive_sensitivity_claim_payloads,
)

pd = pytest.importorskip("pandas")


def _association_summary(**overrides: Any) -> dict[str, Any]:
    summary = {
        "status": "ok",
        "interpretation_class": "adjusted_association",
        "exposure": "lactate_tertile",
        "outcome": "icu_readmission",
        "effect_scale": "odds_ratio",
        "analysis_role": "primary",
        "analysis_set": "complete_case",
        "adjustment_covariates": ["age", "sex", "sofa"],
        "primary_estimate": 1.8604174,
        "primary_estimate_interval": [1.2217491, 2.8331002],
        "primary_contrast": {"exposure_level": "3", "reference_level": "1"},
    }
    summary.update(overrides)
    # The executor also publishes the estimate under its typed names.
    summary.setdefault("adjusted_effect", summary["primary_estimate"])
    return summary


def _bound(summary: dict[str, Any]):
    [claim] = bind_scientific_claim_drafts(
        [draft.model_dump(mode="json") for draft in derive_scientific_claim_drafts(summary)],
        step_id="readmission_model",
        evidence_id="readmission_model_summary",
    )
    return claim


def _tertiles(n: int = 540):
    import numpy as np

    rng = np.random.default_rng(20260924)
    tertile = rng.integers(1, 4, size=n).astype(float)
    age = rng.normal(62, 14, size=n)
    logit = -2.2 + 0.45 * (tertile - 1) + 0.01 * (age - 62)
    readmitted = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"lactate_tertile": tertile, "icu_readmission": readmitted, "age": age})


def _fit(tmp_path, monkeypatch, *, levels: list[str], coding: str, primary: str | None):
    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        run_adjusted_association_from_env,
    )

    frame = _tertiles()
    if coding == "binary":
        frame["lactate_tertile"] = (frame["lactate_tertile"] == 3).astype(float)
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path))
    return run_adjusted_association_from_env(
        requirement_id="readmission_by_tertile",
        exposure="lactate_tertile",
        outcome="icu_readmission",
        covariates=["age"],
        estimator_kind="logistic",
        analysis_set="complete_case",
        analysis_role="primary",
        method_family="logistic_regression",
        model_terms=[
            {
                "name": "lactate_tertile", "role": "exposure", "coding": coding,
                "levels": levels, "reference_level": levels[0],
                "transform": "treatment_contrast",
            },
            {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
        ],
        primary_contrast_level=primary,
        frame=frame,
        emit_step_summary=False,
    )


def test_an_exposure_with_several_levels_publishes_its_primary_contrast(
    tmp_path, monkeypatch
) -> None:
    summary = _fit(tmp_path, monkeypatch, levels=["1", "2", "3"], coding="categorical", primary="3")

    assert summary["primary_contrast"] == {"exposure_level": "3", "reference_level": "1"}
    [claim] = derive_scientific_claim_drafts(summary)
    assert claim.exposure == "lactate_tertile=3 versus lactate_tertile=1"


def test_a_binary_exposure_keeps_its_implicit_contrast(tmp_path, monkeypatch) -> None:
    summary = _fit(tmp_path / "binary", monkeypatch, levels=["0", "1"], coding="binary", primary=None)

    assert "primary_contrast" not in summary
    [claim] = derive_scientific_claim_drafts(summary)
    assert claim.exposure == "lactate_tertile"


def test_the_reader_sentence_names_the_contrast_at_one_precision() -> None:
    claim = _bound(_association_summary())

    assert claim.render_reader_text() == (
        "After adjustment for age, sex, and sofa, lactate tertile=3 versus lactate "
        "tertile=1 was positively associated with icu readmission in the complete "
        "case analysis set (adjusted odds ratio, 1.860; 95% CI, 1.222 to 2.833)."
    )
    assert claim.render_reader_text(include_estimate=False).endswith(
        "in the complete case analysis set (adjusted odds ratio)."
    )


def test_a_small_estimate_keeps_two_significant_figures() -> None:
    claim = _bound(_association_summary(
        effect_scale="coefficient", primary_estimate=0.00312,
        primary_estimate_interval=[0.00047, 0.0058], primary_contrast=None,
        adjustment_covariates=["age", "sofa"],
    ))

    assert "After adjustment for age and sofa, lactate tertile " in claim.render_reader_text()
    assert "0.0031; 95% CI, 0.00047 to 0.0058)" in claim.render_reader_text()


@pytest.mark.parametrize(
    "contrast",
    [
        {"exposure_level": "3", "reference_level": "3"},
        {"exposure_level": "3"},
        "3 vs 1",
    ],
)
def test_a_contrast_without_two_distinct_levels_is_refused(contrast: Any) -> None:
    with pytest.raises(ValueError, match="primary_contrast"):
        derive_scientific_claim_drafts(_association_summary(primary_contrast=contrast))


def test_a_named_level_stays_a_quoted_coordinate() -> None:
    assert contrast_exposure_coordinate("admission_type", "emergency", "elective") == (
        'admission_type="emergency" versus admission_type="elective"'
    )
    assert contrast_exposure_coordinate("lactate_tertile", "03", "1") == (
        'lactate_tertile="03" versus lactate_tertile=1'
    )


def _sensitivity_summary(**overrides: Any) -> dict[str, Any]:
    envelope = {
        "schema_version": "easyicu.binary_sensitivity_reporting/1",
        "analysis_id": "first_stay_only",
        "strategy": "first_stay",
        "exposure": "lactate_tertile",
        "outcome": "icu_readmission",
        "analysis_set": "complete_case",
        "adjustment_covariates": ["age"],
        "covariate": None,
        "effect_scale": "odds_ratio",
        "estimate": 1.7433,
        "lower": 1.0912,
        "upper": 2.7851,
        "n": 480,
        "events": 61,
        "exposure_level": "3",
        "reference_level": "1",
    }
    envelope.update(overrides)
    return {"reportable_sensitivity_results": envelope}


def test_a_sensitivity_refit_names_the_same_contrast() -> None:
    [payload] = derive_sensitivity_claim_payloads(_sensitivity_summary())

    assert payload["exposure"] == "lactate_tertile=3 versus lactate_tertile=1"
    [legacy] = derive_sensitivity_claim_payloads(
        _sensitivity_summary(exposure_level=None, reference_level=None)
    )
    assert legacy["exposure"] == "lactate_tertile"
    with pytest.raises(ValidationError, match="two distinct levels"):
        derive_sensitivity_claim_payloads(_sensitivity_summary(reference_level=None))


def test_a_counts_only_frequency_reads_as_a_frequency() -> None:
    [claim] = bind_scientific_claim_drafts(
        [{
            "schema_version": "easyicu.scientific_claim/2",
            "claim_id": "observed_outcome_frequency",
            "claim_type": "descriptive_absolute_risk",
            "exposure": "primary model complete-case records",
            "outcome": "icu_readmission",
            "direction": "descriptive_only",
            "estimand": (
                "observed outcome frequency was 12.5% (18 events among 144 records; "
                "counts only, no confidence interval)"
            ),
            "population": "the primary model complete-case records",
            "analysis_role": "auxiliary",
            "status": "supported",
            "adjusted_for": [],
        }],
        step_id="readmission_frequency",
        evidence_id="readmission_frequency_summary",
    )

    assert claim.render_reader_text() == (
        "The observed frequency of icu readmission was 12.50% (18 of 144 records) "
        "in the primary model's complete-case records; this was a descriptive, "
        "unadjusted, noncausal frequency."
    )


def test_the_contrast_sentence_survives_strict_numeric_binding(tmp_path) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.manuscript_post import (
        bind_numeric_values,
        drop_untraceable_numeric_sentences,
    )

    summary = _association_summary()
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    record = store.register_json(
        kind="statistic", description="Adjusted association",
        payload=summary, filename="summary.json",
        evidence_id="readmission_model_summary", produced_by_step="readmission_model",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="readmission_model", evidence_id=record.evidence_id, summary=summary,
    )
    [claim] = store.scientific_claims()
    records = [{
        "step_id": "readmission_model", "status": "ok",
        "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": record.evidence_id,
        "evidence_ids": [record.evidence_id],
    }]

    bound = store.bind_manuscript(
        "## Results\n\n### Primary association\n\n" + claim.placeholder,
        per_step_records=records,
    )
    filtered, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=records,
    )
    _, bindings, untraced = bind_numeric_values(
        bound, evidence=store, per_step_records=records,
    )

    assert "lactate tertile=3 versus lactate tertile=1" in bound
    assert removed == [] and filtered == bound
    assert not untraced
    assert {round(float(item.canonical), 4) for item in bindings.values()} >= {
        1.8604, 1.2217, 2.8331,
    }
