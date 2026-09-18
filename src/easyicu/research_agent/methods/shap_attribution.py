"""Deterministic SHAP attribution for fixed tree models (exact values).

Given an already-fitted tree model and a fixed numeric matrix, this kernel
computes exact :class:`shap.TreeExplainer` values and reports per-feature
``mean|SHAP|`` plus the numeric matrices a beeswarm or waterfall plot needs
(full SHAP matrix, base value, and model predictions).  No figure is drawn;
only numbers are returned.

Determinism contract
--------------------
``TreeExplainer`` in exact mode is a pure function of (model bytes, data
bytes): a fixed model evaluated on fixed data yields byte-identical
:func:`shap.TreeExplainer.shap_values` output, so rerunning
:func:`shap_attribute` on identical inputs reproduces the origin digest.  The
kernel consumes no RNG of its own.

Scope and fail-closed behavior: only tree models are accepted.  Anything
else (linear models, MLPs, pipelines, ...) is refused with
:class:`ShapAttributionError` that names the supported fallback
(:func:`sklearn.inspection.permutation_importance`).  Multi-output /
multiclass SHAP tensors that are not a single 2D matrix are likewise
refused instead of being silently squeezed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"

PERMUTATION_FALLBACK = "sklearn.inspection.permutation_importance"

_SKLEARN_TREE_TYPES: tuple[type, ...] = (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)


class ShapAttributionError(ValueError):
    """A model or input the SHAP kernel refuses to attribute."""


def _tree_model_types() -> tuple[type, ...]:
    accepted: list[type] = list(_SKLEARN_TREE_TYPES)
    try:
        import xgboost as xgb
    except ImportError:
        pass
    else:
        accepted.extend(
            [
                xgb.XGBRegressor,
                xgb.XGBClassifier,
                xgb.XGBRFRegressor,
                xgb.XGBRFClassifier,
                xgb.Booster,
            ]
        )
    try:
        import lightgbm as lgb
    except ImportError:
        pass
    else:
        accepted.extend([lgb.LGBMRegressor, lgb.LGBMClassifier])
    return tuple(accepted)


def _require_tree_model(model: object) -> str:
    if not isinstance(model, _tree_model_types()):
        raise ShapAttributionError(
            f"shap_attribution accepts tree models only, got "
            f"{type(model).__name__!r}; fallback for non-tree models: "
            f"{PERMUTATION_FALLBACK}"
        )
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ShapAttributionError(
            f"tree model {type(model).__name__!r} has no predict method; "
            f"fallback: {PERMUTATION_FALLBACK}"
        )
    return type(model).__name__


def _coerce_matrix(X: object) -> tuple[np.ndarray, pd.DataFrame | None]:
    frame: pd.DataFrame | None = X if isinstance(X, pd.DataFrame) else None
    if frame is not None:
        if frame.empty or frame.shape[1] == 0:
            raise ShapAttributionError("X must be a non-empty numeric matrix")
        non_numeric = [
            str(column)
            for column, dtype in frame.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ]
        if non_numeric:
            raise ShapAttributionError(
                "X must be fully numeric; non-numeric columns: "
                + ",".join(sorted(non_numeric))
            )
        values = frame.to_numpy(dtype=float)
    else:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise ShapAttributionError(
                "X must be a numeric 2D array or DataFrame"
            ) from None
        if values.dtype == object or values.ndim != 2:
            raise ShapAttributionError("X must be a numeric 2D array or DataFrame")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ShapAttributionError("X must be a non-empty numeric matrix")
    if not np.all(np.isfinite(values)):
        raise ShapAttributionError("X must contain only finite values (no NaN/inf)")
    return values, frame


def _resolve_feature_names(
    x_frame: pd.DataFrame | None,
    feature_names: Sequence[str] | None,
    n_features: int,
) -> tuple[str, ...]:
    if feature_names is None:
        if x_frame is not None:
            columns = [str(column) for column in x_frame.columns]
        else:
            columns = [f"x{i}" for i in range(n_features)]
    else:
        columns = [str(name) for name in list(feature_names)]
    if len(columns) != n_features:
        raise ShapAttributionError(
            f"feature_names length {len(columns)} does not match "
            f"X features {n_features}"
        )
    if any(not name.strip() for name in columns):
        raise ShapAttributionError("feature_names must all be non-blank")
    if len(set(columns)) != len(columns):
        raise ShapAttributionError("feature_names must be unique")
    return tuple(columns)


@dataclass(frozen=True)
class ShapAttributionResult:
    """Typed SHAP output; JSON form is the digest input."""

    model_kind: str
    feature_names: tuple[str, ...]
    base_value: float
    mean_abs_shap: tuple[float, ...]
    shap_values: tuple[tuple[float, ...], ...]
    predictions: tuple[float, ...]
    n_samples: int
    n_features: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "feature_names": list(self.feature_names),
            "base_value": float(self.base_value),
            "mean_abs_shap": [float(value) for value in self.mean_abs_shap],
            "shap_values": [
                [float(value) for value in row] for row in self.shap_values
            ],
            "predictions": [float(value) for value in self.predictions],
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
        }


def result_sha256(result: ShapAttributionResult) -> str:
    """Return the canonical digest of one attribution result."""

    if not isinstance(result, ShapAttributionResult):
        raise TypeError("result_sha256 requires a ShapAttributionResult")
    return canonical_sha256(result.to_json())


def shap_attribute(
    model: object,
    X: object,
    *,
    feature_names: Sequence[str] | None = None,
) -> ShapAttributionResult:
    """Compute exact TreeExplainer SHAP values for a fixed tree model."""

    model_kind = _require_tree_model(model)
    x_values, x_frame = _coerce_matrix(X)
    names = _resolve_feature_names(x_frame, feature_names, x_values.shape[1])
    try:
        import shap
    except ImportError:
        raise ShapAttributionError(
            "the 'shap' package is required for shap_attribution"
        ) from None

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(x_values)
    if isinstance(raw_values, list):
        raise ShapAttributionError(
            "multi-output SHAP (list of matrices) is not supported; "
            "restrict the model to a single binary/regression output"
        )
    matrix = np.asarray(raw_values, dtype=float)
    if matrix.ndim != 2 or matrix.shape != x_values.shape:
        raise ShapAttributionError(
            f"expected a 2D SHAP matrix matching X{matrix.shape} vs "
            f"{x_values.shape}; multi-output models are not supported"
        )
    if not np.all(np.isfinite(matrix)):
        raise ShapAttributionError("SHAP values must be finite")
    try:
        base_value = float(np.ravel(np.asarray(explainer.expected_value))[0])
    except (TypeError, ValueError, IndexError):
        raise ShapAttributionError(
            "expected_value must reduce to a single base value"
        ) from None
    if not np.isfinite(base_value):
        raise ShapAttributionError("SHAP base value must be finite")
    predictions = np.asarray(
        model.predict(x_values),  # type: ignore[union-attr]
        dtype=float,
    ).ravel()
    if predictions.shape[0] != x_values.shape[0]:
        raise ShapAttributionError("model predictions do not match X rows")
    if not np.all(np.isfinite(predictions)):
        raise ShapAttributionError("model predictions must be finite")

    mean_abs = tuple(float(value) for value in np.abs(matrix).mean(axis=0))
    return ShapAttributionResult(
        model_kind=model_kind,
        feature_names=names,
        base_value=base_value,
        mean_abs_shap=mean_abs,
        shap_values=tuple(
            tuple(float(value) for value in row) for row in matrix.tolist()
        ),
        predictions=tuple(float(value) for value in predictions.tolist()),
        n_samples=int(x_values.shape[0]),
        n_features=int(x_values.shape[1]),
    )


__all__ = [
    "PERMUTATION_FALLBACK",
    "TOOL_VERSION",
    "ShapAttributionError",
    "ShapAttributionResult",
    "shap_attribute",
    "result_sha256",
]
