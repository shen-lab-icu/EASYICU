"""Reviewed, case-neutral primitives for landmark dynamic prediction.

This module deliberately does not choose a model, landmarks, horizons,
features, imputation strategy, or validation split.  Those are scientific
choices owned by the Plan.  It owns three error-prone mechanical operations
that generated analysis code must not reimplement:

* build feature rows using measurements available at or before a landmark;
* label only target horizons that are actually observable under event/censoring;
* evaluate supplied probabilities separately at each landmark and horizon.

Model fitting remains a scikit-learn ``Pipeline`` in the Coder step, with
patient-level splitting and all preprocessing fitted on training rows only.
The helpers are source-digest-bound Coder resources, not a deterministic
primary executor and not publication authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

ID_COL = "stay_id"
TIME_COL = "charttime"
PREDICTION_TIME_COL = "prediction_time_hours"
HORIZON_COL = "target_horizon_hours"
OUTCOME_COL = "outcome"
PROBABILITY_COL = "predicted_probability"

_AGGREGATIONS = frozenset({"last", "mean", "min", "max"})


@dataclass(frozen=True)
class DynamicPredictionEvaluation:
    """Per-landmark performance and calibration products."""

    metrics: pd.DataFrame
    calibration: pd.DataFrame


def _positive_finite(values: Iterable[float], *, label: str) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed or any(not np.isfinite(value) or value <= 0 for value in parsed):
        raise ValueError(f"{label} must contain positive finite values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(parsed))


def _nonnegative_finite(
    values: Iterable[float], *, label: str
) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed or any(not np.isfinite(value) or value < 0 for value in parsed):
        raise ValueError(f"{label} must contain non-negative finite values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(parsed))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def build_landmark_feature_matrix(
    trajectory: pd.DataFrame,
    *,
    feature_concepts: Sequence[str],
    landmark_hours: Sequence[float],
    lookback_hours: float,
    aggregations: Sequence[str] = ("last", "mean", "min", "max"),
) -> pd.DataFrame:
    """Build one leakage-safe feature row per stay and landmark.

    ``trajectory`` must use the standard long representation
    ``stay_id, charttime, concept, value_num`` with ``charttime`` measured in
    hours from the declared time zero.  Each feature uses only rows in
    ``(landmark-lookback, landmark]``.  Missing values remain missing; this
    owner never performs imputation outside a training-only model pipeline.
    """

    _require_columns(
        trajectory,
        (ID_COL, TIME_COL, "concept", "value_num"),
        label="trajectory",
    )
    concepts = tuple(dict.fromkeys(str(value).strip() for value in feature_concepts))
    if not concepts or any(not concept for concept in concepts):
        raise ValueError("feature_concepts must contain non-empty exact concept ids")
    landmarks = _nonnegative_finite(landmark_hours, label="landmark_hours")
    lookback = float(lookback_hours)
    if not np.isfinite(lookback) or lookback <= 0:
        raise ValueError("lookback_hours must be positive and finite")
    requested_aggregations = tuple(
        dict.fromkeys(str(value).strip().lower() for value in aggregations)
    )
    unknown = sorted(set(requested_aggregations) - _AGGREGATIONS)
    if not requested_aggregations or unknown:
        raise ValueError(
            "aggregations must be a non-empty subset of "
            f"{sorted(_AGGREGATIONS)!r}; unknown={unknown!r}"
        )

    if trajectory[ID_COL].isna().any():
        raise ValueError("trajectory requires a non-missing stay_id on every row")
    identities = trajectory[[ID_COL]].drop_duplicates().reset_index(drop=True)
    identities["_stay_order"] = np.arange(len(identities), dtype=int)
    grid = identities.merge(
        pd.DataFrame({PREDICTION_TIME_COL: landmarks}),
        how="cross",
    )

    working = trajectory.loc[
        trajectory["concept"].isin(concepts),
        [ID_COL, TIME_COL, "concept", "value_num"],
    ].copy()
    working[TIME_COL] = pd.to_numeric(working[TIME_COL], errors="coerce")
    working["value_num"] = pd.to_numeric(working["value_num"], errors="coerce")
    if working[TIME_COL].isna().any() or (~np.isfinite(working[TIME_COL])).any():
        raise ValueError("selected trajectory rows require finite charttime values")
    working = working.sort_values([ID_COL, TIME_COL, "concept"])

    feature_names = [
        f"{concept}__{aggregation}"
        for concept in concepts
        for aggregation in requested_aggregations
    ]
    frames: list[pd.DataFrame] = []
    for landmark in landmarks:
        window = working.loc[
            (working[TIME_COL] > landmark - lookback)
            & (working[TIME_COL] <= landmark)
        ]
        if window.empty:
            continue
        grouped = window.groupby([ID_COL, "concept"], sort=False)["value_num"]
        pieces: list[pd.Series] = []
        for aggregation in requested_aggregations:
            if aggregation == "last":
                values = grouped.last()
            else:
                values = getattr(grouped, aggregation)()
            values.name = aggregation
            pieces.append(values)
        summary = pd.concat(pieces, axis=1).reset_index()
        wide = summary.pivot(index=ID_COL, columns="concept")
        wide.columns = [f"{concept}__{aggregation}" for aggregation, concept in wide]
        wide = wide.reindex(columns=feature_names).reset_index()
        wide.insert(1, PREDICTION_TIME_COL, landmark)
        frames.append(wide)

    observed = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=[ID_COL, PREDICTION_TIME_COL, *feature_names])
    )
    # Preserve every declared stay/landmark combination.  A patient with no
    # measurement in a lookback window is missing, not absent from the risk set.
    return (
        grid.merge(
            observed,
            on=[ID_COL, PREDICTION_TIME_COL],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["_stay_order", PREDICTION_TIME_COL])
        .drop(columns="_stay_order")
        .reset_index(drop=True)
    )


def attach_landmark_outcomes(
    feature_matrix: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    event_time_col: str,
    followup_end_col: str,
    horizon_hours: Sequence[float],
) -> pd.DataFrame:
    """Attach binary future outcomes without treating censoring as non-events.

    A row is observable when an event occurs in ``(landmark, landmark+horizon]``
    or follow-up reaches the end of that horizon.  Prevalent events are not at
    risk.  Unobservable horizons remain explicitly marked with a missing label.
    """

    _require_columns(
        feature_matrix,
        (ID_COL, PREDICTION_TIME_COL),
        label="feature_matrix",
    )
    _require_columns(
        outcomes,
        (ID_COL, event_time_col, followup_end_col),
        label="outcomes",
    )
    if outcomes[ID_COL].duplicated().any():
        raise ValueError("outcomes must contain exactly one row per stay_id")
    horizons = _positive_finite(horizon_hours, label="horizon_hours")
    outcome_rows = outcomes[[ID_COL, event_time_col, followup_end_col]].copy()
    outcome_rows[event_time_col] = pd.to_numeric(
        outcome_rows[event_time_col], errors="coerce"
    )
    outcome_rows[followup_end_col] = pd.to_numeric(
        outcome_rows[followup_end_col], errors="coerce"
    )
    if outcome_rows[followup_end_col].isna().any() or (
        ~np.isfinite(outcome_rows[followup_end_col])
    ).any():
        raise ValueError("every outcome row requires a finite follow-up end time")
    event_present = outcome_rows[event_time_col].notna()
    if (
        event_present
        & (
            ~np.isfinite(outcome_rows[event_time_col])
            | (
                outcome_rows[event_time_col]
                > outcome_rows[followup_end_col]
            )
        )
    ).any():
        raise ValueError("event times must be finite and no later than follow-up end")

    base = feature_matrix.merge(outcome_rows, on=ID_COL, how="left", validate="many_to_one")
    if base[followup_end_col].isna().any():
        raise ValueError("feature rows contain stay_id values without outcome authority")
    expanded = pd.concat(
        [base.assign(**{HORIZON_COL: horizon}) for horizon in horizons],
        ignore_index=True,
    )
    prediction_time = pd.to_numeric(
        expanded[PREDICTION_TIME_COL], errors="coerce"
    )
    if prediction_time.isna().any() or (~np.isfinite(prediction_time)).any():
        raise ValueError("prediction_time_hours must be finite")
    event_time = expanded[event_time_col]
    horizon_end = prediction_time + expanded[HORIZON_COL]
    under_observation = expanded[followup_end_col].ge(prediction_time)
    at_risk = under_observation & (
        event_time.isna() | event_time.gt(prediction_time)
    )
    event_in_horizon = event_time.gt(prediction_time) & event_time.le(horizon_end)
    horizon_observed = event_in_horizon | expanded[followup_end_col].ge(horizon_end)
    expanded["eligible_at_landmark"] = at_risk.astype(int)
    expanded["horizon_observed"] = (at_risk & horizon_observed).astype(int)
    expanded[OUTCOME_COL] = np.where(
        at_risk & horizon_observed,
        event_in_horizon.astype(float),
        np.nan,
    )
    return expanded


def evaluate_landmark_probabilities(
    predictions: pd.DataFrame,
    *,
    outcome_col: str = OUTCOME_COL,
    probability_col: str = PROBABILITY_COL,
    calibration_bins: int = 10,
) -> DynamicPredictionEvaluation:
    """Evaluate supplied probabilities separately by landmark and horizon."""

    _require_columns(
        predictions,
        (ID_COL, PREDICTION_TIME_COL, HORIZON_COL, outcome_col, probability_col),
        label="predictions",
    )
    if int(calibration_bins) < 2:
        raise ValueError("calibration_bins must be at least 2")
    working = predictions.copy()
    working[outcome_col] = pd.to_numeric(working[outcome_col], errors="coerce")
    working[probability_col] = pd.to_numeric(
        working[probability_col], errors="coerce"
    )
    for coordinate in (PREDICTION_TIME_COL, HORIZON_COL):
        working[coordinate] = pd.to_numeric(working[coordinate], errors="coerce")
    if (
        working[PREDICTION_TIME_COL].isna().any()
        or (~np.isfinite(working[PREDICTION_TIME_COL])).any()
        or working[PREDICTION_TIME_COL].lt(0).any()
    ):
        raise ValueError("prediction times must be non-negative and finite")
    if (
        working[HORIZON_COL].isna().any()
        or (~np.isfinite(working[HORIZON_COL])).any()
        or working[HORIZON_COL].le(0).any()
    ):
        raise ValueError("target horizons must be positive and finite")
    if working[[outcome_col, probability_col]].isna().any().any():
        raise ValueError("prediction evaluation requires observed labels and probabilities")
    if not set(working[outcome_col].unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("prediction outcomes must be binary")
    probabilities = working[probability_col]
    if (~np.isfinite(probabilities)).any() or probabilities.lt(0).any() or probabilities.gt(1).any():
        raise ValueError("predicted probabilities must be finite and within [0, 1]")
    duplicate_key = [ID_COL, PREDICTION_TIME_COL, HORIZON_COL]
    if working.duplicated(duplicate_key).any():
        raise ValueError("predictions must contain one probability per stay/landmark/horizon")

    metrics: list[dict[str, float | int]] = []
    calibration_rows: list[dict[str, float | int]] = []
    group_columns = [PREDICTION_TIME_COL, HORIZON_COL]
    for (landmark, horizon), group in working.groupby(group_columns, sort=True):
        observed = group[outcome_col].astype(int).to_numpy()
        predicted = group[probability_col].astype(float).to_numpy()
        if len(np.unique(observed)) != 2:
            raise ValueError(
                "each landmark/horizon evaluation requires both outcome classes"
            )
        metrics.append(
            {
                PREDICTION_TIME_COL: float(landmark),
                HORIZON_COL: float(horizon),
                "n": int(len(group)),
                "events": int(observed.sum()),
                "auroc": float(roc_auc_score(observed, predicted)),
                "brier_score": float(brier_score_loss(observed, predicted)),
                "observed_event_rate": float(observed.mean()),
                "mean_predicted_probability": float(predicted.mean()),
            }
        )
        fraction_positive, mean_predicted = calibration_curve(
            observed,
            predicted,
            n_bins=min(int(calibration_bins), len(group)),
            strategy="quantile",
        )
        calibration_rows.extend(
            {
                PREDICTION_TIME_COL: float(landmark),
                HORIZON_COL: float(horizon),
                "bin_index": int(index),
                "mean_predicted_probability": float(mean_probability),
                "observed_event_rate": float(observed_rate),
            }
            for index, (mean_probability, observed_rate) in enumerate(
                zip(mean_predicted, fraction_positive, strict=True), start=1
            )
        )
    return DynamicPredictionEvaluation(
        metrics=pd.DataFrame(metrics),
        calibration=pd.DataFrame(calibration_rows),
    )


__all__ = [
    "DynamicPredictionEvaluation",
    "attach_landmark_outcomes",
    "build_landmark_feature_matrix",
    "evaluate_landmark_probabilities",
]
