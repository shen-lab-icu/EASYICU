"""Synthetic transport: fixed coefficients, different outcome prevalence."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from easyicu.research_agent.contracts.prediction_validation import (
    ExternalValidationError,
    ExternalValidationSpec,
    FrozenLogisticModel,
    FrozenPredictor,
)
from easyicu.research_agent.methods.prediction_validation import (
    ValidationSource,
    _read_after_freeze,
    run_external_validation,
)


def synthetic_transport(recalibration="intercept_only"):
    # Ten replicated covariate strata give a stable, nonseparated slope.
    x = np.repeat(np.linspace(-2, 2, 10), 100)
    y = np.concatenate([
        np.r_[np.ones(round(100 * expit(0.9 + value))),
              np.zeros(100 - round(100 * expit(0.9 + value))) ]
        for value in np.linspace(-2, 2, 10)
    ]).astype(int)
    frame = pd.DataFrame({"subject_id": np.arange(len(x)), "hr": 80 + 10*x, "death": y})
    model = FrozenLogisticModel(
        development_database="miiv", outcome_concept="death", outcome_meaning="hospital_death",
        intercept=0, development_source_sha256="a" * 64,
        predictors=(FrozenPredictor(
            concept="hr", meaning="bpm;mean;0-24h", coefficient=1,
            median=80, center=80, scale=10, development_mean=80,
            development_sd=10, development_missing_fraction=0,
        ),),
    )
    payload = model.model_dump_json().encode()
    spec = ExternalValidationSpec(
        development_database="miiv", validation_databases=("eicu",),
        frozen_model_sha256=hashlib.sha256(payload).hexdigest(),
        predictor_concepts=("hr",), outcome_concept="death", recalibration=recalibration,
    )
    source = ValidationSource(
        database="eicu", columns=("subject_id", "hr", "death"),
        meanings={"hr": "bpm;mean;0-24h", "death": "hospital_death"},
        read=lambda: (frame.copy(), "b" * 64),
    )
    return payload, spec, source


def test_an_unfrozen_read_is_refused_before_the_reader_is_called():
    _, spec, source = synthetic_transport()
    reads = []
    source.read = lambda: (reads.append(True), "b" * 64)
    with pytest.raises(ExternalValidationError, match="model_not_frozen_before_validation_read"):
        _read_after_freeze([], spec, source)
    assert reads == []


def test_fixed_preprocessing_and_coefficients_are_applied_without_refitting():
    payload, spec, source = synthetic_transport("none")
    frame, digest = source.read()
    frame.loc[0, "hr"] = np.nan
    source.read = lambda: (frame, digest)
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    expected = expit((frame["hr"].fillna(80).to_numpy() - 80) / 10)
    y = frame["death"].to_numpy()
    assert result["databases"][0]["transport"]["brier_score"] == pytest.approx(np.mean((y - expected)**2))
    assert result["databases"][0]["recalibration"] is None
    decision = result["databases"][0]["decision_curve"][0]
    positive = expected >= decision["threshold"]
    assert decision["net_benefit_model"] == pytest.approx(
        np.mean(positive & (y == 1)) - np.mean(positive & (y == 0)) * 0.05 / 0.95
    )


def test_a_fitted_development_pipeline_transports_without_a_second_fit(monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(117)
    development = pd.DataFrame({"hr": rng.normal(80, 10, 240)})
    y = rng.binomial(1, expit((development["hr"].to_numpy() - 80) / 10))
    development.loc[::19, "hr"] = np.nan
    fitted = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=200))
    fitted.fit(development, y)
    imputer, scaler, estimator = fitted.steps[0][1], fitted.steps[1][1], fitted.steps[2][1]
    model = FrozenLogisticModel(
        development_database="miiv", development_source_sha256=hashlib.sha256(development.to_csv(index=False).encode()).hexdigest(),
        outcome_concept="death", outcome_meaning="hospital_death", intercept=estimator.intercept_[0],
        predictors=(FrozenPredictor(
            concept="hr", meaning="bpm;mean;0-24h", coefficient=estimator.coef_[0, 0],
            median=imputer.statistics_[0], center=scaler.mean_[0], scale=scaler.scale_[0],
            development_mean=development["hr"].mean(), development_sd=development["hr"].std(),
            development_missing_fraction=development["hr"].isna().mean(),
        ),),
    )
    payload = model.model_dump_json().encode()
    spec = ExternalValidationSpec(development_database="miiv", validation_databases=("eicu",),
                                  frozen_model_sha256=hashlib.sha256(payload).hexdigest(),
                                  predictor_concepts=("hr",), outcome_concept="death")
    seen = {}

    def read_validation():
        target = pd.DataFrame({"subject_id": np.arange(10_000, 10_180), "hr": rng.normal(90, 12, 180)})
        target["death"] = rng.binomial(1, expit(0.9 + (target["hr"].to_numpy() - 80) / 10))
        target.loc[::11, "hr"] = np.nan
        seen["target"] = target
        return target, hashlib.sha256(target.to_csv(index=False).encode()).hexdigest()

    def forbidden_refit(*args, **kwargs):
        pytest.fail("external validation refitted the development pipeline")

    monkeypatch.setattr(type(fitted), "fit", forbidden_refit)
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(ValidationSource(
        database="eicu", columns=("subject_id", "hr", "death"),
        meanings={"hr": "bpm;mean;0-24h", "death": "hospital_death"}, read=read_validation,
    ),))
    target = seen["target"]
    expected = fitted.predict_proba(target[["hr"]])[:, 1]
    assert result["databases"][0]["transport"]["brier_score"] == pytest.approx(
        np.mean((target["death"].to_numpy() - expected)**2), abs=1e-12,
    )


def test_transport_is_untouched_and_intercept_recalibration_is_separate():
    payload, spec, source = synthetic_transport()
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    row = result["databases"][0]
    transport = row["transport"]
    assert transport["calibration_in_the_large"] == pytest.approx(0.9, abs=0.03)
    assert transport["calibration_slope"] == pytest.approx(1, abs=0.03)
    assert transport["auroc_ci_low"] < transport["auroc"] < transport["auroc_ci_high"]
    assert transport["auroc_ci_method"] == "delong_logit"
    updated = row["recalibration"]
    assert updated["evaluation_scope"] == "apparent_on_recalibration_cohort"
    assert updated["intercept"] == pytest.approx(0.9, abs=0.03)
    assert updated["metrics"]["calibration_in_the_large"] == pytest.approx(0, abs=1e-8)
    assert updated["metrics"]["brier_score"] < transport["brier_score"]
    assert updated["metrics"]["auroc"] == pytest.approx(transport["auroc"])
    assert len(row["flexible_calibration"]) == 41
    assert all(0 <= p["observed"] <= 1 for p in row["flexible_calibration"])
    assert np.mean([p["observed"] - p["predicted"] for p in row["flexible_calibration"]]) > 0.1
    assert row["decision_curve"][0]["net_benefit_none"] == 0
    assert row["case_mix"][0]["standardized_difference"] == pytest.approx(0, abs=1e-12)
    assert result["provenance"][0]["event"] == "model_frozen"
    assert result["provenance"][1]["event"] == "validation_read"
    assert result["provenance"][1]["frozen_model_sha256"] == spec.frozen_model_sha256
    assert result["authority_scope"] == "analysis_only"
    assert not any(key in json.dumps(result) for key in ("subject_id", "probabilities"))


def test_digest_mismatch_refuses_before_validation_is_read():
    payload, spec, source = synthetic_transport()
    reads = []
    source.read = lambda: reads.append(True)
    with pytest.raises(ExternalValidationError) as error:
        run_external_validation(model_bytes=payload + b" ", spec=spec, sources=(source,))
    assert error.value.reason_code == "frozen_model_digest_mismatch"
    assert reads == []


def test_model_roster_cannot_differ_from_the_reviewed_predictors():
    payload, spec, source = synthetic_transport()
    reads = []
    source.read = lambda: reads.append(True)
    with pytest.raises(ExternalValidationError, match="frozen_model_plan_mismatch"):
        run_external_validation(model_bytes=payload, spec=spec.model_copy(update={"predictor_concepts": ("lact",)}),
                                sources=(source,))
    assert reads == []


@pytest.mark.parametrize("field,value", [
    ("scale", 0), ("coefficient", np.inf), ("median", np.nan),
    ("development_sd", -1), ("development_missing_fraction", 1),
])
def test_frozen_preprocessing_and_case_mix_state_is_validated(field, value):
    payload, _, _ = synthetic_transport()
    predictor = FrozenLogisticModel.model_validate_json(payload).predictors[0]
    with pytest.raises(ValueError):
        FrozenPredictor.model_validate({**predictor.model_dump(), field: value})


def test_intercept_only_recalibration_uses_an_offset_when_the_slope_is_wrong():
    payload, spec, source = synthetic_transport()
    model = json.loads(payload)
    model["predictors"][0]["coefficient"] = 0.5
    payload = json.dumps(model).encode()
    spec = spec.model_copy(update={"frozen_model_sha256": hashlib.sha256(payload).hexdigest()})
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    row = result["databases"][0]
    assert row["transport"]["calibration_slope"] == pytest.approx(2, abs=0.04)
    assert row["recalibration"]["slope"] == 1
    assert row["recalibration"]["metrics"]["mean_predicted_probability"] == pytest.approx(
        row["transport"]["event_rate"], abs=1e-10,
    )


def test_missing_predictor_refuses_before_read_without_dropping_it():
    payload, spec, source = synthetic_transport()
    source.columns = ("subject_id", "death")
    reads = []
    source.read = lambda: reads.append(True)
    with pytest.raises(ExternalValidationError) as error:
        run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    assert error.value.reason_code == "predictor_unavailable"
    assert error.value.detail["concepts"] == ["hr"]
    assert reads == []


@pytest.mark.parametrize("defect,reason", [
    ("outcome", "outcome_unavailable"), ("meaning", "concept_meaning_unavailable"),
])
def test_incomplete_outcome_or_meaning_refuses_before_read(defect, reason):
    payload, spec, source = synthetic_transport()
    if defect == "outcome":
        source.columns = ("subject_id", "hr")
    else:
        source.meanings.pop("hr")
    reads = []
    source.read = lambda: reads.append(True)
    with pytest.raises(ExternalValidationError, match=reason):
        run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    assert reads == []


def test_case_mix_compares_observed_means_and_pooled_variances():
    payload, spec, source = synthetic_transport()
    frame, digest = source.read()
    frame["hr"] += 10
    source.read = lambda: (frame, digest)
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    comparison = result["databases"][0]["case_mix"][0]
    expected = 10 / np.sqrt((100 + frame["hr"].std(ddof=1)**2) / 2)
    assert comparison["standardized_difference"] == pytest.approx(expected)
    assert comparison["validation_missing_fraction"] == 0


def test_case_mix_does_not_estimate_variance_from_one_observed_predictor():
    payload, spec, source = synthetic_transport("none")
    frame, digest = source.read()
    frame["hr"] = np.nan
    frame.loc[0, "hr"] = 80
    source.read = lambda: (frame, digest)
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    comparison = result["databases"][0]["case_mix"][0]
    assert comparison["validation_sd"] is None
    assert comparison["standardized_difference"] is None
    assert comparison["status"] == "not_estimable_insufficient_observations"


def test_changed_outcome_meaning_needs_exact_reviewed_confirmation():
    payload, spec, source = synthetic_transport()
    source.meanings["death"] = "icu_death"
    with pytest.raises(ExternalValidationError) as error:
        run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    assert error.value.reason_code == "meaning_change_requires_confirmation"
    confirmed = ExternalValidationSpec.model_validate({
        **spec.model_dump(), "confirmed_changes": [{
            "database": "eicu", "concept": "death", "development_meaning": "hospital_death",
            "validation_meaning": "icu_death", "confirmation": "Compare ICU mortality as a changed endpoint.",
        }],
    })
    result = run_external_validation(model_bytes=payload, spec=confirmed, sources=(source,))
    assert result["databases"][0]["changed_estimand"] is True
    assert result["databases"][0]["meaning_changes"][0]["validation_meaning"] == "icu_death"


def test_logistic_recalibration_is_optional_and_does_not_change_transport():
    payload, spec, source = synthetic_transport("logistic")
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    row = result["databases"][0]
    assert row["recalibration"]["slope"] == pytest.approx(1, abs=0.03)
    assert row["recalibration"]["metrics"]["calibration_slope"] == pytest.approx(1, abs=1e-6)
    assert row["transport"]["calibration_in_the_large"] > 0.8


@pytest.mark.parametrize("defect,reason", [
    ("duplicate_subject", "independent_subjects_required"),
    ("missing_column", "predictor_unavailable"),
    ("all_missing", "predictor_unavailable"),
    ("single_class", "outcome_invalid"),
    ("infinite", "predictor_invalid"),
])
def test_invalid_validation_rows_fail_closed(defect, reason):
    payload, spec, source = synthetic_transport()
    frame, digest = source.read()
    if defect == "duplicate_subject":
        frame.loc[1, "subject_id"] = frame.loc[0, "subject_id"]
    elif defect == "missing_column":
        frame = frame.drop(columns="hr")
    elif defect == "all_missing":
        frame["hr"] = np.nan
    elif defect == "single_class":
        frame["death"] = 0
    else:
        frame.loc[0, "hr"] = np.inf
    source.read = lambda: (frame, digest)
    with pytest.raises(ExternalValidationError) as error:
        run_external_validation(model_bytes=payload, spec=spec, sources=(source,))
    assert error.value.reason_code == reason
