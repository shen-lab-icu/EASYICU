"""Leakage-safe early-feature assignment to frozen patient subtypes.

This kernel covers the supervised step that may follow an unsupervised
trajectory study.  The trajectory model and its labels must already be frozen;
this module never discovers, merges, renames, or refits subtypes.  It fits all
preprocessing on a patient-disjoint development set, predicts the held-out
validation set, and returns discrimination, calibration, confusion, and
coefficient products.

The feature window must end before the trajectory window used to define the
labels.  Results remain ``analysis_only``: internal assignment performance is
not external validation and does not turn a data-derived phenotype into a
clinical diagnostic category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SubtypeAssignmentError(ValueError):
    """The requested subtype-assignment analysis violates its data contract."""


@dataclass(frozen=True)
class SubtypeAssignmentEvaluation:
    """Held-out products for one frozen-label subtype classifier."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    calibration: pd.DataFrame
    confusion: pd.DataFrame
    coefficients: pd.DataFrame
    feature_window_end: float
    phenotype_window_start: float
    claim_ceiling: str = "analysis_only"

    def to_json(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(orient="records"),
            "predictions": self.predictions.to_dict(orient="records"),
            "calibration": self.calibration.to_dict(orient="records"),
            "confusion": self.confusion.to_dict(orient="records"),
            "coefficients": self.coefficients.to_dict(orient="records"),
            "feature_window_end": self.feature_window_end,
            "phenotype_window_start": self.phenotype_window_start,
            "claim_ceiling": self.claim_ceiling,
        }


def _patient_ids(values: Sequence[object], *, label: str, n_rows: int) -> np.ndarray:
    ids = np.asarray(list(values), dtype=object)
    if ids.ndim != 1 or ids.shape[0] != n_rows:
        raise SubtypeAssignmentError(f"{label} must contain one id per row")
    if any(value is None or not str(value).strip() for value in ids):
        raise SubtypeAssignmentError(f"{label} must not contain missing ids")
    if len(set(map(str, ids))) != len(ids):
        raise SubtypeAssignmentError(
            f"{label} contains duplicate patients; aggregate early features first"
        )
    return ids


def _feature_matrix(frame: pd.DataFrame, *, label: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[1] == 0:
        raise SubtypeAssignmentError(f"{label} must be a non-empty DataFrame")
    names = tuple(str(column) for column in frame.columns)
    if len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise SubtypeAssignmentError(f"{label} requires unique, non-empty feature names")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SubtypeAssignmentError(f"{label} must contain numeric features") from exc
    if np.isinf(values).any():
        raise SubtypeAssignmentError(f"{label} must not contain infinite values")
    return values, names


def _labels(values: Sequence[object], *, label: str, n_rows: int) -> np.ndarray:
    labels = np.asarray(list(values), dtype=object)
    if labels.ndim != 1 or labels.shape[0] != n_rows:
        raise SubtypeAssignmentError(f"{label} must contain one frozen label per row")
    if any(value is None or not str(value).strip() for value in labels):
        raise SubtypeAssignmentError(f"{label} must not contain missing labels")
    return np.asarray([str(value) for value in labels], dtype=object)


def _calibration_rows(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    *,
    bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for class_index, class_label in enumerate(classes):
        predicted = probabilities[:, class_index]
        observed = (y_true == class_label).astype(float)
        bin_ids = np.minimum(np.digitize(predicted, edges[1:-1], right=False), bins - 1)
        for bin_index in range(bins):
            mask = bin_ids == bin_index
            if not mask.any():
                continue
            rows.append(
                {
                    "subtype": str(class_label),
                    "bin": bin_index + 1,
                    "n": int(mask.sum()),
                    "mean_predicted_probability": float(predicted[mask].mean()),
                    "observed_fraction": float(observed[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def fit_and_evaluate_early_subtype_assignment(
    development_features: pd.DataFrame,
    development_labels: Sequence[object],
    validation_features: pd.DataFrame,
    validation_labels: Sequence[object],
    *,
    development_patient_ids: Sequence[object],
    validation_patient_ids: Sequence[object],
    feature_window_end: float,
    phenotype_window_start: float,
    forbidden_feature_names: Sequence[str] = (),
    calibration_bins: int = 5,
    class_weight: str | None = "balanced",
    random_state: int = 0,
) -> SubtypeAssignmentEvaluation:
    """Fit on development patients and evaluate frozen subtype labels.

    The caller owns cohort construction, the frozen subtype solution, feature
    selection, and the patient-level split.  This helper refuses overlapping
    patients, post-phenotype feature windows, unseen validation labels, and
    feature rosters that change between development and validation.
    """

    x_dev, names = _feature_matrix(development_features, label="development_features")
    x_val, validation_names = _feature_matrix(
        validation_features, label="validation_features"
    )
    if validation_names != names:
        raise SubtypeAssignmentError(
            "development and validation feature columns must match in the same order"
        )
    forbidden = {str(value).strip() for value in forbidden_feature_names if str(value).strip()}
    leaked = sorted(set(names) & forbidden)
    if leaked:
        raise SubtypeAssignmentError(
            f"feature roster contains forbidden label/outcome columns: {leaked}"
        )
    if any(np.isnan(x_dev[:, index]).all() for index in range(x_dev.shape[1])):
        raise SubtypeAssignmentError(
            "every development feature requires at least one observed value"
        )

    dev_ids = _patient_ids(
        development_patient_ids,
        label="development_patient_ids",
        n_rows=x_dev.shape[0],
    )
    val_ids = _patient_ids(
        validation_patient_ids,
        label="validation_patient_ids",
        n_rows=x_val.shape[0],
    )
    overlap = sorted(set(map(str, dev_ids)) & set(map(str, val_ids)))
    if overlap:
        raise SubtypeAssignmentError(
            f"development and validation patients overlap: {overlap[:5]}"
        )

    feature_end = float(feature_window_end)
    phenotype_start = float(phenotype_window_start)
    if not np.isfinite(feature_end) or not np.isfinite(phenotype_start):
        raise SubtypeAssignmentError("feature and phenotype window boundaries must be finite")
    if feature_end >= phenotype_start:
        raise SubtypeAssignmentError(
            "early-feature window must end before the phenotype-defining trajectory window"
        )
    if isinstance(calibration_bins, bool) or int(calibration_bins) != calibration_bins:
        raise SubtypeAssignmentError("calibration_bins must be an integer")
    bins = int(calibration_bins)
    if bins < 2 or bins > 20:
        raise SubtypeAssignmentError("calibration_bins must be between 2 and 20")

    y_dev = _labels(development_labels, label="development_labels", n_rows=x_dev.shape[0])
    y_val = _labels(validation_labels, label="validation_labels", n_rows=x_val.shape[0])
    dev_classes = set(y_dev.tolist())
    if len(dev_classes) < 2:
        raise SubtypeAssignmentError("development labels require at least two frozen subtypes")
    unseen = sorted(set(y_val.tolist()) - dev_classes)
    if unseen:
        raise SubtypeAssignmentError(
            f"validation labels contain subtypes absent from development: {unseen}"
        )

    model = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=int(random_state),
                    solver="lbfgs",
                ),
            ),
        )
    )
    model.fit(x_dev, y_dev)
    probabilities = model.predict_proba(x_val)
    predictions = model.predict(x_val)
    classes = np.asarray(model.named_steps["classifier"].classes_, dtype=object)
    one_hot = (y_val[:, None] == classes[None, :]).astype(float)
    multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    metrics = pd.DataFrame(
        [
            {
                "n_development": int(x_dev.shape[0]),
                "n_validation": int(x_val.shape[0]),
                "n_subtypes": int(classes.shape[0]),
                "balanced_accuracy": float(balanced_accuracy_score(y_val, predictions)),
                "macro_f1": float(f1_score(y_val, predictions, labels=classes, average="macro")),
                "multiclass_log_loss": float(log_loss(y_val, probabilities, labels=classes)),
                "multiclass_brier": multiclass_brier,
                "claim_ceiling": "analysis_only",
            }
        ]
    )
    prediction_rows = pd.DataFrame(
        {
            "patient_id": val_ids,
            "observed_subtype": y_val,
            "predicted_subtype": predictions,
        }
    )
    for index, class_label in enumerate(classes):
        prediction_rows[f"probability__{class_label}"] = probabilities[:, index]

    matrix = confusion_matrix(y_val, predictions, labels=classes)
    confusion_rows = [
        {
            "observed_subtype": str(observed),
            "predicted_subtype": str(predicted),
            "n": int(matrix[row_index, column_index]),
        }
        for row_index, observed in enumerate(classes)
        for column_index, predicted in enumerate(classes)
    ]

    classifier = model.named_steps["classifier"]
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercepts = np.asarray(classifier.intercept_, dtype=float)
    if coefficients.shape[0] == 1 and classes.shape[0] == 2:
        coefficients = np.vstack((-coefficients[0], coefficients[0]))
        intercepts = np.asarray((-intercepts[0], intercepts[0]), dtype=float)
    coefficient_rows = [
        {
            "subtype": str(class_label),
            "feature": feature,
            "standardized_coefficient": float(coefficients[class_index, feature_index]),
            "intercept": float(intercepts[class_index]),
        }
        for class_index, class_label in enumerate(classes)
        for feature_index, feature in enumerate(names)
    ]

    return SubtypeAssignmentEvaluation(
        metrics=metrics,
        predictions=prediction_rows,
        calibration=_calibration_rows(
            probabilities,
            y_val,
            classes,
            bins=bins,
        ),
        confusion=pd.DataFrame(confusion_rows),
        coefficients=pd.DataFrame(coefficient_rows),
        feature_window_end=feature_end,
        phenotype_window_start=phenotype_start,
    )


__all__ = [
    "SubtypeAssignmentError",
    "SubtypeAssignmentEvaluation",
    "fit_and_evaluate_early_subtype_assignment",
]
