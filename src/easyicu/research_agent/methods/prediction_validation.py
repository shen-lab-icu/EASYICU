"""Binary prediction metrics, internal validation and frozen model transport.

The probability-table evaluator remains non-training. Enhanced bootstrap
accepts a full-pipeline refit callback from the executor, and score recalibration
fits only the development rows supplied by that owner. The external route
applies a fixed numeric logistic model and labels optional recalibration as
apparent performance on its fitting cohort. No function chooses a cohort,
outcome, predictor roster, or split.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pydantic import ValidationError
from patsy import build_design_matrices, dmatrix
from ..canonical_json import canonical_sha256
from ..contracts.closed_levels import typed_level_key
from ..contracts.prediction_subgroup import (
    PredictionSubgroupError, PredictionSubgroupFinding, PredictionSubgroupPerformance,
    PredictionSubgroupResult, PredictionSubgroupSpec, prediction_subgroup_spec_sha256,
)
from ..contracts.prediction_validation import (
    ExternalValidationError, ExternalValidationSpec, FrozenLogisticModel,
)
from .decision_curve import net_benefit_curve
from .delong_auc import delong_auc_ci

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
    PerfectSeparationWarning,
)
from scipy.special import expit, logit
from scipy.optimize import brentq

from ..contracts.prediction_validation import (
    CalibrationStatus,
    PredictionCalibrationBin,
    PredictionThresholdMetric,
    PredictionValidationError,
    PredictionValidationReason,
    PredictionValidationResult,
    PredictionValidationSpec,
    PredictionValidationSummary,
    prediction_validation_spec_sha256,
)


def _raise(
    reason_code: PredictionValidationReason,
    message: str,
    **detail: Any,
) -> None:
    raise PredictionValidationError(reason_code, message, **detail)


def _missing_identity_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(
        lambda value: isinstance(value, str) and not value.strip()
    )


def _evaluation_arrays(
    frame: pd.DataFrame,
    spec: PredictionValidationSpec,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = {
        spec.unit_id_column,
        spec.subject_id_column,
        spec.split_column,
        spec.outcome_column,
        spec.probability_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        _raise(
            PredictionValidationReason.MISSING_COLUMNS,
            "prediction table is missing declared columns",
            missing_columns=missing,
        )
    if frame.empty:
        _raise(
            PredictionValidationReason.EMPTY_INPUT,
            "prediction table contains no rows",
        )

    for column in (spec.unit_id_column, spec.subject_id_column):
        missing_identity = _missing_identity_mask(frame[column])
        if bool(missing_identity.any()):
            _raise(
                PredictionValidationReason.IDENTITY_MISSING,
                "prediction table contains a missing identity",
                column=column,
                row_count=int(missing_identity.sum()),
            )
    duplicate_units = frame[spec.unit_id_column].duplicated(keep=False)
    if bool(duplicate_units.any()):
        _raise(
            PredictionValidationReason.DUPLICATE_UNIT,
            "unit identifiers must be globally unique",
            column=spec.unit_id_column,
            row_count=int(duplicate_units.sum()),
        )

    raw_splits = frame[spec.split_column]
    invalid_splits = raw_splits.isna() | raw_splits.map(
        lambda value: not str(value).strip() if value is not None else True
    )
    if bool(invalid_splits.any()):
        _raise(
            PredictionValidationReason.SPLIT_MISSING,
            "every row requires a non-empty split label",
            column=spec.split_column,
            row_count=int(invalid_splits.sum()),
        )
    splits = raw_splits.map(lambda value: str(value).strip())
    subject_split_count = splits.groupby(
        frame[spec.subject_id_column], sort=False
    ).nunique()
    leaking_subjects = subject_split_count[subject_split_count > 1]
    if not leaking_subjects.empty:
        _raise(
            PredictionValidationReason.SUBJECT_SPLIT_LEAKAGE,
            "a subject occurs in more than one split",
            subject_count=int(len(leaking_subjects)),
        )

    evaluation_mask = splits.eq(spec.evaluation_split)
    if not bool(evaluation_mask.any()):
        _raise(
            PredictionValidationReason.EVALUATION_SPLIT_MISSING,
            "the declared evaluation split contains no rows",
            evaluation_split=spec.evaluation_split,
            available_splits=sorted(set(splits)),
        )
    evaluation = frame.loc[evaluation_mask].copy()
    repeated_subjects = evaluation[spec.subject_id_column].duplicated(keep=False)
    if spec.analysis_unit == "subject" and bool(repeated_subjects.any()):
        _raise(
            PredictionValidationReason.SUBJECT_UNIT_NOT_UNIQUE,
            "subject-level evaluation requires exactly one row per subject",
            row_count=int(repeated_subjects.sum()),
        )

    raw_outcomes = evaluation[spec.outcome_column]
    if pd.api.types.is_bool_dtype(
        raw_outcomes.dtype
    ) or not pd.api.types.is_numeric_dtype(raw_outcomes.dtype):
        _raise(
            PredictionValidationReason.OUTCOME_INVALID,
            "evaluation outcomes must use a numeric, non-boolean dtype",
            column=spec.outcome_column,
            observed_dtype=str(raw_outcomes.dtype),
        )
    outcomes = pd.to_numeric(raw_outcomes, errors="coerce")
    invalid_outcomes = outcomes.isna() | ~outcomes.isin((0, 1))
    if bool(invalid_outcomes.any()):
        _raise(
            PredictionValidationReason.OUTCOME_INVALID,
            "evaluation outcomes must be observed binary values",
            column=spec.outcome_column,
            row_count=int(invalid_outcomes.sum()),
        )
    raw_probabilities = evaluation[spec.probability_column]
    if pd.api.types.is_bool_dtype(
        raw_probabilities.dtype
    ) or not pd.api.types.is_numeric_dtype(raw_probabilities.dtype):
        _raise(
            PredictionValidationReason.PROBABILITY_INVALID,
            "evaluation probabilities must use a numeric, non-boolean dtype",
            column=spec.probability_column,
            observed_dtype=str(raw_probabilities.dtype),
        )
    probabilities = pd.to_numeric(raw_probabilities, errors="coerce")
    invalid_probabilities = (
        probabilities.isna()
        | ~np.isfinite(probabilities)
        | probabilities.lt(0.0)
        | probabilities.gt(1.0)
    )
    if bool(invalid_probabilities.any()):
        _raise(
            PredictionValidationReason.PROBABILITY_INVALID,
            "evaluation probabilities must be finite values in [0, 1]",
            column=spec.probability_column,
            row_count=int(invalid_probabilities.sum()),
        )

    outcome_values = outcomes.astype(int).to_numpy()
    probability_values = probabilities.astype(float).to_numpy()
    if len(np.unique(outcome_values)) != 2:
        _raise(
            PredictionValidationReason.SINGLE_CLASS,
            "evaluation requires both outcome classes",
            observed_classes=sorted(set(outcome_values.tolist())),
        )
    return evaluation, outcome_values, probability_values


def _perfectly_separated(outcomes: np.ndarray, logits: np.ndarray) -> bool:
    event_logits = logits[outcomes == 1]
    non_event_logits = logits[outcomes == 0]
    return bool(
        np.min(event_logits) > np.max(non_event_logits)
        or np.max(event_logits) < np.min(non_event_logits)
    )


def _calibration_model(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    epsilon: float,
) -> tuple[CalibrationStatus, float | None, float | None, int]:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    clipped_n = int(np.count_nonzero(clipped != probabilities))
    logits = np.log(clipped / (1.0 - clipped))
    if np.unique(logits).size < 2:
        return "not_estimable_constant_probability", None, None, clipped_n
    if _perfectly_separated(outcomes, logits):
        return "not_estimable_perfect_separation", None, None, clipped_n

    design = sm.add_constant(logits, has_constant="add")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PerfectSeparationWarning)
            fitted = sm.GLM(
                outcomes,
                design,
                family=sm.families.Binomial(),
            ).fit(maxiter=100, tol=1e-10)
    except (PerfectSeparationError, np.linalg.LinAlgError):
        return "not_estimable_nonconvergence", None, None, clipped_n
    if any(issubclass(item.category, PerfectSeparationWarning) for item in caught):
        return "not_estimable_perfect_separation", None, None, clipped_n
    parameters = np.asarray(fitted.params, dtype=float)
    if (
        not bool(fitted.converged)
        or parameters.shape != (2,)
        or not np.isfinite(parameters).all()
    ):
        return "not_estimable_nonconvergence", None, None, clipped_n
    return "estimated", float(parameters[0]), float(parameters[1]), clipped_n


def _calibration_bins(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    requested_bins: int,
) -> tuple[PredictionCalibrationBin, ...]:
    quantile_count = min(int(requested_bins), len(probabilities))
    edges = np.unique(
        np.quantile(
            probabilities,
            np.linspace(0.0, 1.0, quantile_count + 1),
            method="linear",
        )
    )
    labels = (
        np.zeros(len(probabilities), dtype=int)
        if len(edges) == 1
        else np.searchsorted(edges[1:-1], probabilities, side="right")
    )
    rows: list[PredictionCalibrationBin] = []
    for output_index, label in enumerate(sorted(set(labels.tolist())), start=1):
        selected = labels == label
        selected_outcomes = outcomes[selected]
        selected_probabilities = probabilities[selected]
        rows.append(
            PredictionCalibrationBin(
                bin_index=output_index,
                n=int(selected.sum()),
                event_n=int(selected_outcomes.sum()),
                mean_predicted_probability=float(selected_probabilities.mean()),
                observed_event_rate=float(selected_outcomes.mean()),
                minimum_predicted_probability=float(selected_probabilities.min()),
                maximum_predicted_probability=float(selected_probabilities.max()),
            )
        )
    return tuple(rows)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _threshold_metrics(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: tuple[float, ...],
) -> tuple[PredictionThresholdMetric, ...]:
    rows: list[PredictionThresholdMetric] = []
    for threshold in thresholds:
        positive = probabilities >= threshold
        negative = ~positive
        events = outcomes == 1
        non_events = ~events
        true_positive = int(np.count_nonzero(positive & events))
        false_positive = int(np.count_nonzero(positive & non_events))
        true_negative = int(np.count_nonzero(negative & non_events))
        false_negative = int(np.count_nonzero(negative & events))
        rows.append(
            PredictionThresholdMetric(
                threshold=threshold,
                n=int(len(outcomes)),
                predicted_positive_n=int(np.count_nonzero(positive)),
                true_positive_n=true_positive,
                false_positive_n=false_positive,
                true_negative_n=true_negative,
                false_negative_n=false_negative,
                sensitivity=float(true_positive / (true_positive + false_negative)),
                specificity=float(true_negative / (true_negative + false_positive)),
                positive_predictive_value=_safe_ratio(
                    true_positive, true_positive + false_positive
                ),
                negative_predictive_value=_safe_ratio(
                    true_negative, true_negative + false_negative
                ),
            )
        )
    return tuple(rows)


def evaluate_binary_predictions(
    frame: pd.DataFrame,
    spec: PredictionValidationSpec | Mapping[str, Any],
) -> PredictionValidationResult:
    """Evaluate one predeclared split without training or choosing anything."""

    parsed_spec = PredictionValidationSpec.model_validate(spec)
    evaluation, outcomes, probabilities = _evaluation_arrays(frame, parsed_spec)
    calibration_status, intercept, slope, clipped_n = _calibration_model(
        outcomes,
        probabilities,
        epsilon=parsed_spec.calibration_logit_epsilon,
    )
    subject_counts = evaluation[parsed_spec.subject_id_column].value_counts()
    repeated_subject_n = int((subject_counts > 1).sum())
    event_n = int(outcomes.sum())
    evaluation_n = int(len(outcomes))
    summary = PredictionValidationSummary(
        evaluation_split=parsed_spec.evaluation_split,
        analysis_unit=parsed_spec.analysis_unit,
        input_n=int(len(frame)),
        evaluation_n=evaluation_n,
        event_n=event_n,
        non_event_n=evaluation_n - event_n,
        evaluation_subject_n=int(subject_counts.size),
        repeated_subject_n=repeated_subject_n,
        clipped_probability_n=clipped_n,
        event_rate=float(outcomes.mean()),
        mean_predicted_probability=float(probabilities.mean()),
        auroc=float(roc_auc_score(outcomes, probabilities)),
        brier_score=float(brier_score_loss(outcomes, probabilities)),
        calibration_status=calibration_status,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )
    return PredictionValidationResult(
        contract_sha256=prediction_validation_spec_sha256(parsed_spec),
        summary=summary,
        calibration_bins=_calibration_bins(
            outcomes,
            probabilities,
            requested_bins=parsed_spec.calibration_bins,
        ),
        threshold_metrics=_threshold_metrics(
            outcomes,
            probabilities,
            thresholds=parsed_spec.thresholds,
        ),
    )


def calibration_in_the_large(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    """Offset-logistic intercept with slope fixed at one, not the joint intercept."""

    logits = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
    return float(brentq(lambda alpha: np.mean(expit(alpha + logits)) - np.mean(outcomes), -50, 50))


def optimism_metrics(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """The four prespecified metrics; an unestimable calibration invalidates a draw."""

    outcomes, probabilities = np.asarray(outcomes), np.asarray(probabilities)
    if (outcomes.ndim != 1 or probabilities.shape != outcomes.shape
            or not np.isin(outcomes, (0, 1)).all() or np.unique(outcomes).size != 2
            or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("bootstrap metrics require aligned binary outcomes and finite probabilities")
    status, _intercept, slope, _clipped = _calibration_model(outcomes, probabilities, epsilon=1e-6)
    if status != "estimated":
        raise RuntimeError(f"prediction_bootstrap_calibration_{status}")
    return {
        "auroc": float(roc_auc_score(outcomes, probabilities)),
        "calibration_slope": float(slope),
        "calibration_in_the_large": calibration_in_the_large(outcomes, probabilities),
        "brier_score": float(brier_score_loss(outcomes, probabilities)),
    }


class BootstrapValidationError(RuntimeError):
    """A bootstrap whose failed fits exceed the prespecified tolerance."""

    def __init__(self, failed_replicates: int, n_replicates: int) -> None:
        self.failed_replicates = failed_replicates
        self.n_replicates = n_replicates
        super().__init__(f"prediction_bootstrap_too_many_failures: {failed_replicates}/{n_replicates}")


def bootstrap_optimism_validation(
    *, outcomes, groups, refit_predict: Callable[[np.ndarray], np.ndarray],
    n_replicates: int, seed: int,
) -> dict[str, Any]:
    """Enhanced patient bootstrap of the entire modelling procedure.

    ``refit_predict`` must freshly fit its complete pipeline on the supplied
    positional rows (including duplicates) and predict every original row.
    Optimism is resample minus original for every metric, including Brier;
    subtracting a negative Brier optimism therefore increases corrected error.
    This estimates performance on new patients from the development population.
    It neither trains on held-out patients nor supplies an external validation.
    """

    from ..contracts.prediction_validation import BootstrapSpec

    spec = BootstrapSpec(replicates=n_replicates, seed=seed)
    y = np.asarray(outcomes)
    codes, patients = pd.factorize(np.asarray(groups), sort=False)
    if codes.shape != y.shape or (codes < 0).any() or len(patients) < 2:
        raise ValueError("bootstrap requires aligned complete patient groups")
    original_rows = np.arange(len(y))
    apparent = optimism_metrics(y, refit_predict(original_rows))
    order = np.argsort(codes, kind="stable")
    patient_rows = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    generator = np.random.default_rng(spec.seed)
    replicates, failures = [], []
    for replicate in range(spec.replicates):
        draws = generator.integers(0, len(patients), size=len(patients))
        rows = np.concatenate([patient_rows[i] for i in draws])
        try:
            if np.unique(y[rows]).size != 2:
                raise ValueError("single_class_resample")
            probabilities = np.asarray(refit_predict(rows))
            original = optimism_metrics(y, probabilities)
            resample = optimism_metrics(y[rows], probabilities[rows])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append({"replicate": replicate + 1, "reason": str(exc)})
            continue
        replicates.append({"replicate": replicate + 1, "resample": resample, "original": original})
    if len(failures) > 0.1 * spec.replicates:
        raise BootstrapValidationError(len(failures), spec.replicates)
    metrics = {}
    for name, apparent_value in apparent.items():
        optimism = float(np.mean([r["resample"][name] - r["original"][name] for r in replicates]))
        metrics[name] = {"apparent": apparent_value, "optimism": optimism, "corrected": apparent_value - optimism}
    return {
        "method": "patient_bootstrap_optimism_full_pipeline_refit",
        "n_replicates": spec.replicates, "seed": spec.seed,
        "max_failed_fraction": 0.1, "failed_replicates": len(failures),
        "successful_replicates": len(replicates), "patient_n": len(patients), "row_n": len(y),
        "metrics": metrics, "replicates": replicates, "failures": failures,
    }


def recalibrate_score(*, outcomes, scores, evaluation_scores) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a score's own unpenalized logistic mapping on development rows only."""

    y, score, evaluation = np.asarray(outcomes), np.asarray(scores, dtype=float), np.asarray(evaluation_scores, dtype=float)
    if (y.ndim != 1 or score.shape != y.shape or np.unique(y).size != 2
            or not np.isin(y, (0, 1)).all() or not np.isfinite(score).all()
            or not np.isfinite(evaluation).all() or np.unique(score).size < 2
            or _perfectly_separated(y, score)):
        raise ValueError("benchmark_recalibration_not_estimable")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerfectSeparationWarning)
        fitted = sm.GLM(y, sm.add_constant(score, has_constant="add"), family=sm.families.Binomial()).fit(maxiter=100, tol=1e-10)
    if (not fitted.converged or not np.isfinite(fitted.params).all()
            or any(issubclass(w.category, PerfectSeparationWarning) for w in caught)):
        raise RuntimeError("benchmark_recalibration_not_estimable")
    intercept, slope = map(float, fitted.params)
    return expit(intercept + slope * evaluation), {
        "intercept": intercept, "slope": slope, "fit_n": len(y),
        "fit_scope": "development_partition_only", "method": "unpenalized_logistic_score",
    }


__all__ = ["evaluate_binary_predictions", "bootstrap_optimism_validation", "recalibrate_score"]


@dataclass
class ValidationSource:
    """Metadata and a lazy reader; the runner supplies verified intake facts."""

    database: str
    columns: tuple[str, ...]
    meanings: dict[str, str]
    read: Callable[[], tuple[pd.DataFrame, str]]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc = delong_auc_ci(y, p)
    status, intercept, slope, clipped = _calibration_model(y, p, epsilon=1e-6)
    return {
        "n": len(y), "event_n": int(y.sum()), "event_rate": float(y.mean()),
        "mean_predicted_probability": float(p.mean()), "auroc": auc.auc,
        "auroc_ci_low": auc.ci_low, "auroc_ci_high": auc.ci_high,
        "auroc_ci_method": "delong_logit", "confidence_level": 0.95,
        "calibration_in_the_large": calibration_in_the_large(y, p),
        "calibration_joint_intercept": intercept, "calibration_slope": slope,
        "calibration_status": status, "clipped_probability_n": clipped,
        "brier_score": float(np.mean((y - p)**2)),
    }


def _flexible_calibration(y: np.ndarray, p: np.ndarray) -> tuple[list[dict], str]:
    """Four-df cubic logistic spline, on a fixed 41-point aggregate grid."""
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))
    if len(np.unique(x)) < 5 or min(y.sum(), len(y) - y.sum()) < 10:
        return [], "not_estimable_insufficient_support"
    grid = np.linspace(float(x.min()), float(x.max()), 41)
    design = dmatrix("bs(x, df=4, degree=3) + 1", {"x": x})
    grid_design = build_design_matrices([design.design_info], {"x": grid})[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(maxiter=100)
        observed = fit.predict(grid_design)
        if not fit.converged or not np.isfinite(observed).all():
            return [], "not_estimable_nonconvergence"
    except (ValueError, np.linalg.LinAlgError, PerfectSeparationError, PerfectSeparationWarning):
        return [], "not_estimable_nonconvergence"
    return [
        {"predicted": float(a), "observed": float(b)}
        for a, b in zip(expit(grid), observed, strict=True)
    ], "cubic_logistic_spline_df4"


def _source_gate(model: FrozenLogisticModel, spec: ExternalValidationSpec, source: ValidationSource) -> list[dict]:
    required = {p.concept for p in model.predictors}
    absent = sorted(required - set(source.columns))
    if absent:
        raise ExternalValidationError("predictor_unavailable", database=source.database, concepts=absent)
    if model.outcome_concept not in source.columns:
        raise ExternalValidationError("outcome_unavailable", database=source.database)
    changes = []
    meanings = {p.concept: p.meaning for p in model.predictors}
    meanings[model.outcome_concept] = model.outcome_meaning
    confirmations = {(c.database, c.concept): c for c in spec.confirmed_changes}
    for concept, meaning in meanings.items():
        observed = source.meanings.get(concept)
        if not observed:
            raise ExternalValidationError("concept_meaning_unavailable", database=source.database, concept=concept)
        if observed == meaning:
            continue
        confirmation = confirmations.get((source.database, concept))
        if (confirmation is None or confirmation.development_meaning != meaning
                or confirmation.validation_meaning != observed):
            raise ExternalValidationError(
                "meaning_change_requires_confirmation", database=source.database, concept=concept,
                development_meaning=meaning, validation_meaning=observed,
            )
        changes.append(confirmation.model_dump())
    return changes


def _read_after_freeze(provenance: list[dict], spec: ExternalValidationSpec, source: ValidationSource):
    """A validation read is legal only behind this invocation's freeze event."""
    if (not provenance or provenance[0].get("event") != "model_frozen"
            or provenance[0].get("frozen_model_sha256") != spec.frozen_model_sha256
            or provenance[0].get("plan_spec_sha256") != canonical_sha256(spec.model_dump(mode="json"))):
        raise ExternalValidationError("model_not_frozen_before_validation_read")
    frame, digest = source.read()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ExternalValidationError("validation_source_digest_invalid")
    provenance.append({
        "event": "validation_read", "database": source.database, "sequence": len(provenance),
        "frozen_model_sha256": spec.frozen_model_sha256, "source_sha256": digest,
    })
    return frame


def _arrays_and_case_mix(frame: pd.DataFrame, model: FrozenLogisticModel):
    missing = sorted({p.concept for p in model.predictors} - set(frame.columns))
    if missing:
        raise ExternalValidationError("predictor_unavailable", concepts=missing)
    if model.subject_id_column not in frame or frame[model.subject_id_column].isna().any() or frame[model.subject_id_column].duplicated().any():
        raise ExternalValidationError("independent_subjects_required")
    if frame[model.subject_id_column].map(lambda x: isinstance(x, str) and not x.strip()).any():
        raise ExternalValidationError("independent_subjects_required")
    if model.outcome_concept not in frame:
        raise ExternalValidationError("outcome_unavailable")
    outcome = frame[model.outcome_concept]
    if not pd.api.types.is_numeric_dtype(outcome.dtype):
        raise ExternalValidationError("outcome_invalid")
    y = outcome.to_numpy(dtype=float, na_value=np.nan)
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2 or min(np.sum(y == 0), np.sum(y == 1)) < 2:
        raise ExternalValidationError("outcome_invalid")
    eta = np.full(len(frame), model.intercept, dtype=float)
    case_mix = []
    for predictor in model.predictors:
        raw = frame[predictor.concept]
        if not pd.api.types.is_numeric_dtype(raw.dtype) or pd.api.types.is_bool_dtype(raw.dtype):
            raise ExternalValidationError("predictor_invalid", concept=predictor.concept)
        values = raw.to_numpy(dtype=float, na_value=np.nan)
        if np.isnan(values).all():
            raise ExternalValidationError("predictor_unavailable", concepts=[predictor.concept])
        if np.isinf(values).any():
            raise ExternalValidationError("predictor_invalid", concept=predictor.concept)
        observed = values[~np.isnan(values)]
        estimable = len(observed) > 1
        mean = float(observed.mean())
        sd = float(observed.std(ddof=1)) if len(observed) > 1 else 0.0
        pooled = np.sqrt((sd**2 + predictor.development_sd**2) / 2)
        difference = mean - predictor.development_mean
        case_mix.append({
            "concept": predictor.concept, "development_mean": predictor.development_mean,
            "validation_mean": mean, "development_sd": predictor.development_sd,
            "validation_sd": sd if estimable else None, "validation_observed_n": len(observed),
            "development_missing_fraction": predictor.development_missing_fraction,
            "validation_missing_fraction": float(np.isnan(values).mean()),
            "standardized_difference": (float(difference / pooled) if pooled else 0.0 if difference == 0 else None) if estimable else None,
            "status": ("estimated" if pooled or difference == 0 else "not_estimable_zero_variance") if estimable else "not_estimable_insufficient_observations",
        })
        values = np.where(np.isnan(values), predictor.median, values)
        eta += predictor.coefficient * ((values - predictor.center) / predictor.scale)
    if not np.isfinite(eta).all():
        raise ExternalValidationError("prediction_nonfinite")
    return y.astype(int), expit(eta), case_mix


def _recalibration(y: np.ndarray, p: np.ndarray, kind: str) -> dict | None:
    if kind == "none":
        return None
    slope = 1.0
    if kind == "intercept_only":
        intercept = calibration_in_the_large(y, p)
    else:
        status, intercept, slope, _ = _calibration_model(y, p, epsilon=1e-6)
        if status != "estimated":
            return {"method": kind, "status": status, "evaluation_scope": "apparent_on_recalibration_cohort"}
    updated = expit(intercept + slope * logit(np.clip(p, 1e-6, 1 - 1e-6)))
    return {
        "method": kind, "status": "estimated", "intercept": intercept, "slope": slope,
        "evaluation_scope": "apparent_on_recalibration_cohort", "metrics": _metrics(y, updated),
    }


def run_external_validation(
    *, model_bytes: bytes, spec: ExternalValidationSpec | Mapping,
    sources: Sequence[ValidationSource] | Callable[[FrozenLogisticModel], Sequence[ValidationSource]],
) -> dict:
    """Freeze exact model bytes before even asking intake for source metadata.

    The callable form lets the runner verify materialized sidecars after the
    freeze: those verifiers can read cohort bytes to check their digests.
    """
    spec = ExternalValidationSpec.model_validate(spec)
    if hashlib.sha256(model_bytes).hexdigest() != spec.frozen_model_sha256:
        raise ExternalValidationError("frozen_model_digest_mismatch")
    model = FrozenLogisticModel.model_validate_json(model_bytes)
    if (model.development_database != spec.development_database
            or model.outcome_concept != spec.outcome_concept
            or tuple(p.concept for p in model.predictors) != spec.predictor_concepts):
        raise ExternalValidationError("frozen_model_plan_mismatch")
    provenance = [{
        "event": "model_frozen", "sequence": 0, "frozen_model_sha256": spec.frozen_model_sha256,
        "plan_spec_sha256": canonical_sha256(spec.model_dump(mode="json")),
        "development_source_sha256": model.development_source_sha256,
    }]
    sources = tuple(sources(model) if callable(sources) else sources)
    if tuple(s.database for s in sources) != spec.validation_databases:
        raise ExternalValidationError("validation_database_roster_mismatch")
    changes = [_source_gate(model, spec, source) for source in sources]
    rows = []
    for source, meaning_changes in zip(sources, changes, strict=True):
        frame = _read_after_freeze(provenance, spec, source)
        y, p, case_mix = _arrays_and_case_mix(frame, model)
        curve, curve_status = _flexible_calibration(y, p)
        rows.append({
            "database": source.database, "transport": _metrics(y, p),
            "flexible_calibration": curve, "flexible_calibration_status": curve_status,
            "decision_curve": net_benefit_curve(y, p, thresholds=np.linspace(0.05, 0.5, 10)).to_dict("records"),
            "case_mix": case_mix, "changed_estimand": bool(meaning_changes),
            "meaning_changes": meaning_changes, "recalibration": _recalibration(y, p, spec.recalibration),
        })
    return {
        "schema_version": "easyicu.external_prediction_validation/1",
        "development_database": spec.development_database, "databases": rows,
        "provenance": provenance, "authority_scope": "analysis_only",
        "claim_ceiling": "transport_to_named_cohorts_only", "paper_authorization_allowed": False,
    }


def _refuse(code: str, message: str) -> None:
    raise PredictionSubgroupError("prediction_subgroup_" + code, message)


def _missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())


def subgroup_level_key(value: Any) -> tuple[str, str]:
    """Normalize scalar storage wrappers, retaining typed category identity."""
    if pd.isna(value):
        return ("missing", "")
    if isinstance(value, np.generic):
        value = value.item()
    return typed_level_key(value)


def _validated_evaluation(frame: pd.DataFrame, spec: PredictionSubgroupSpec) -> pd.DataFrame:
    required = {
        spec.unit_id_column, spec.subject_id_column, spec.split_column,
        spec.outcome_column, spec.probability_column, *(axis.column for axis in spec.axes),
    }
    if not frame.columns.is_unique or required - set(frame.columns):
        _refuse("columns_invalid", "the prediction table requires unique declared columns")
    if frame.empty:
        _refuse("empty_input", "the prediction table contains no rows")
    for column in (spec.unit_id_column, spec.subject_id_column):
        if bool(_missing(frame[column]).any()):
            _refuse("identity_missing", "every row requires both declared identities")
    if bool(frame[spec.unit_id_column].duplicated().any()):
        _refuse("duplicate_unit", "prediction unit identifiers must be unique")
    splits = frame[spec.split_column]
    if bool(_missing(splits).any()) or any(not isinstance(value, str) or value != value.strip() for value in splits):
        _refuse("split_invalid", "split labels must be canonical nonempty strings")
    if bool((splits.groupby(frame[spec.subject_id_column]).nunique() > 1).any()):
        _refuse("subject_split_leakage", "a subject cannot occur in multiple splits")
    evaluation = frame.loc[splits.eq(spec.evaluation_split)].copy()
    if evaluation.empty:
        _refuse("evaluation_split_missing", "the declared split contains no rows")
    if spec.analysis_unit == "subject" and bool(evaluation[spec.subject_id_column].duplicated().any()):
        _refuse("subject_unit_not_unique", "subject analysis requires one row per subject")
    for column, role in ((spec.outcome_column, "outcome"), (spec.probability_column, "probability")):
        raw = evaluation[column]
        if pd.api.types.is_bool_dtype(raw.dtype) or not pd.api.types.is_numeric_dtype(raw.dtype):
            _refuse(role + "_invalid", f"{role} requires a numeric non-boolean dtype")
        observed = raw.dropna()
        invalid = ~observed.isin((0, 1)) if role == "outcome" else (
            ~np.isfinite(observed) | observed.lt(0) | observed.gt(1)
        )
        if bool(invalid.any()):
            _refuse(role + "_invalid", f"observed {role} values are outside the declared binary probability contract")
        if spec.missing_value_policy == "error" and bool(raw.isna().any()):
            _refuse(role + "_missing", f"missing {role} requires an explicit complete-case declaration")
    for axis in spec.axes:
        allowed = {subgroup_level_key(level) for level in axis.levels}
        if any(subgroup_level_key(value) not in allowed for value in evaluation[axis.column]):
            _refuse("undeclared_level", "an evaluation subgroup value is outside the closed category roster")
    return evaluation


def _cell(
    frame: pd.DataFrame, spec: PredictionSubgroupSpec, *, scope: str, group_values: dict[str, Any],
) -> PredictionSubgroupPerformance:
    complete = frame[spec.outcome_column].notna() & frame[spec.probability_column].notna()
    evaluated = frame.loc[complete]
    n = len(evaluated)
    outcomes = evaluated[spec.outcome_column].to_numpy(dtype=int)
    probabilities = evaluated[spec.probability_column].to_numpy(dtype=float)
    events = int(outcomes.sum())
    subjects = evaluated[spec.subject_id_column].value_counts()
    small = n < spec.minimum_group_n or min(events, n - events) < spec.minimum_class_n
    reasons = []
    if not n:
        reasons.append("empty_after_declared_complete_case" if len(frame) else "empty_declared_group")
    elif small:
        reasons.append("below_declared_sample_counts")
    values: dict[str, Any] = {}
    if n:
        values.update(event_rate=float(outcomes.mean()), mean_predicted_probability=float(probabilities.mean()))
    if n and events and n - events:
        result = evaluate_binary_predictions(evaluated, PredictionValidationSpec(
            unit_id_column=spec.unit_id_column, subject_id_column=spec.subject_id_column,
            split_column=spec.split_column, outcome_column=spec.outcome_column,
            probability_column=spec.probability_column, evaluation_split=spec.evaluation_split,
            analysis_unit=spec.analysis_unit, thresholds=(0.5,),
            calibration_bins=spec.calibration_bins, calibration_logit_epsilon=spec.calibration_logit_epsilon,
        )).summary
        values.update({name: getattr(result, name) for name in (
            "auroc", "brier_score", "calibration_status", "calibration_intercept",
            "calibration_slope", "clipped_probability_n",
        )})
        values["auroc_status"] = "estimated"
        if result.calibration_status != "estimated":
            reasons.append(result.calibration_status)
    else:
        status = "not_estimable_single_class" if n else "not_estimable_empty"
        values.update(auroc_status=status, calibration_status=status)
        if n:
            # The base owner correctly refuses single-class AUROC/calibration.
            # sklearn still defines the binary Brier score for these cells.
            values["brier_score"] = float(brier_score_loss(outcomes, probabilities, pos_label=1))
            reasons.append(status)
    if bool((subjects > 1).any()):
        reasons.append("repeated_subjects_point_estimates_only")
    return PredictionSubgroupPerformance(
        scope=scope, group_values=group_values, input_n=len(frame), evaluation_n=n,
        excluded_n=len(frame) - n, missing_outcome_n=int(frame[spec.outcome_column].isna().sum()),
        missing_probability_n=int(frame[spec.probability_column].isna().sum()),
        event_n=events, non_event_n=n - events, subject_n=len(subjects),
        repeated_subject_n=int((subjects > 1).sum()),
        sample_status="empty" if not n else "small_sample" if small else "adequate_for_declared_counts",
        reason_codes=tuple(reasons), **values,
    )


def evaluate_prediction_subgroups(
    frame: pd.DataFrame, spec: PredictionSubgroupSpec | Mapping[str, Any],
) -> PredictionSubgroupResult:
    """Evaluate all prespecified cells, retaining empty and non-estimable cells."""
    parsed = PredictionSubgroupSpec.model_validate(spec)
    evaluation = _validated_evaluation(frame, parsed)
    rows = [_cell(evaluation, parsed, scope="overall", group_values={})]
    keys = {axis.column: evaluation[axis.column].map(subgroup_level_key) for axis in parsed.axes}
    for axis in parsed.axes:
        for level in axis.levels:
            mask = keys[axis.column].map(lambda key: key == subgroup_level_key(level))
            rows.append(_cell(evaluation.loc[mask], parsed, scope="marginal", group_values={axis.column: level}))
    if parsed.include_intersections and len(parsed.axes) > 1:
        for levels in product(*(axis.levels for axis in parsed.axes)):
            values = dict(zip((axis.column for axis in parsed.axes), levels))
            mask = pd.Series(True, index=evaluation.index)
            for column, level in values.items():
                mask &= keys[column].map(lambda key: key == subgroup_level_key(level))
            rows.append(_cell(evaluation.loc[mask], parsed, scope="intersection", group_values=values))
    return PredictionSubgroupResult(
        contract_sha256=prediction_subgroup_spec_sha256(parsed), evaluation_split=parsed.evaluation_split,
        analysis_unit=parsed.analysis_unit, input_n=len(frame), evaluation_input_n=len(evaluation),
        evaluation_unit_ids_sha256=canonical_sha256([
            list(subgroup_level_key(value)) for value in evaluation[parsed.unit_id_column]
        ]), rows=tuple(rows),
    )


def prediction_subgroup_result_findings(
    *, frame: pd.DataFrame, spec: PredictionSubgroupSpec | Mapping[str, Any],
    candidate: PredictionSubgroupResult | Mapping[str, Any],
) -> tuple[PredictionSubgroupFinding, ...]:
    """Recompute a candidate, without issuing an authority seal or paper grant."""
    payload = candidate.model_dump(mode="python") if isinstance(candidate, PredictionSubgroupResult) else candidate
    try:
        parsed = PredictionSubgroupResult.model_validate(payload)
    except ValidationError:
        return (PredictionSubgroupFinding(
            reason_code="prediction_subgroup_result_schema_invalid",
            message="candidate subgroup performance is not schema-valid",
        ),)
    expected = evaluate_prediction_subgroups(frame, spec)
    if canonical_sha256(parsed.model_dump(mode="json")) != canonical_sha256(expected.model_dump(mode="json")):
        return (PredictionSubgroupFinding(
            reason_code="prediction_subgroup_result_mismatch",
            message="candidate subgroup performance differs from declared-input recomputation",
        ),)
    return ()


def bind_subgroup_membership(
    predictions: pd.DataFrame, membership: pd.DataFrame, spec: PredictionSubgroupSpec,
) -> pd.DataFrame:
    """Connect identities explicitly; never infer a join from row positions.

    The existing prediction producer encodes a unique source identity as
    ``type:repr``. That encoding is allowed only when explicitly declared and
    the source key is unique; its positional duplicate-key fallback cannot be
    reconstructed safely here and is refused.
    """
    needed = {spec.subgroup_unit_id_column, *(axis.column for axis in spec.axes)}
    if (not membership.columns.is_unique or needed - set(membership.columns)
            or spec.unit_id_column not in predictions or not predictions.columns.is_unique):
        raise PredictionSubgroupError("prediction_subgroup_membership_columns_invalid", "membership and prediction tables require their declared unique columns")
    source = membership[spec.subgroup_unit_id_column]
    if bool(source.isna().any()) or bool(source.map(lambda value: isinstance(value, str) and not value.strip()).any()):
        raise PredictionSubgroupError("prediction_subgroup_membership_identity_missing", "membership requires a nonmissing row key")
    if spec.subgroup_key_encoding == "prediction_typed_scalar":
        source_keys = source.map(lambda value: f"{type(value).__name__}:{value!r}")
        score_keys = predictions[spec.unit_id_column]
    else:
        source_keys = source.map(subgroup_level_key)
        score_keys = predictions[spec.unit_id_column].map(subgroup_level_key)
    if bool(source_keys.duplicated().any()):
        raise PredictionSubgroupError("prediction_subgroup_membership_duplicate_unit", "membership requires a unique key without positional disambiguation")
    positions = {key: position for position, key in enumerate(source_keys)}
    if any(key not in positions for key in score_keys):
        raise PredictionSubgroupError("prediction_subgroup_membership_unmatched", "every prediction row must match exactly one declared membership row")
    selected = membership.iloc[[positions[key] for key in score_keys]]
    result = predictions.copy()
    for axis in spec.axes:
        if axis.column in result:
            raise PredictionSubgroupError("prediction_subgroup_axis_collision", "the score artifact already contains a declared membership column")
        result[axis.column] = selected[axis.column].to_numpy()
    return result


def subgroup_performance_frame(result: PredictionSubgroupResult) -> pd.DataFrame:
    """A deterministic aggregate-only serialization, shared with its output gate."""
    rows = []
    for cell in result.rows:
        row = cell.model_dump(mode="json")
        row["group_values"] = json.dumps(row["group_values"], ensure_ascii=False, sort_keys=True, allow_nan=False)
        row["reason_codes"] = json.dumps(row["reason_codes"], ensure_ascii=False)
        row.update(evaluation_split=result.evaluation_split, analysis_unit=result.analysis_unit, uncertainty=result.uncertainty)
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_provenance(
    result: PredictionSubgroupResult, spec: PredictionSubgroupSpec,
    prediction_sha256: str, membership_sha256: str,
) -> dict[str, Any]:
    return {
        "prediction_artifact_sha256": prediction_sha256,
        "membership_artifact_sha256": membership_sha256,
        "contract_sha256": result.contract_sha256,
        "result_sha256": canonical_sha256(result.model_dump(mode="json")),
        "evaluation_unit_ids_sha256": result.evaluation_unit_ids_sha256,
        "declaration": spec.model_dump(mode="json"),
        "model_refitted": False,
        "model_identity_verification": "a fixed score artifact is verified; independent model identity is not established",
        "calibration_definition": "joint logistic intercept and slope; probabilities clipped only for the calibration logit",
        "sample_status_definition": "declared count flags; they do not establish statistical precision or clinical adequacy",
        "uncertainty": "point_estimates_only; no subgroup difference tests or confidence intervals",
        "interpretation": result.interpretation,
    }
