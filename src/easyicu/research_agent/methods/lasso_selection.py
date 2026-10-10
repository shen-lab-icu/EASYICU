"""Deterministic Lasso variable selection (fixed seed + fixed CV splits).

This kernel owns variable-selection mechanics only: given a complete-case
numeric design matrix ``X`` and outcome vector ``y`` it fits either a
fixed-alpha :class:`~sklearn.linear_model.Lasso` (``method="lasso"``) or a
cross-validated :class:`~sklearn.linear_model.LassoCV` (``method="lassocv"``)
and reports the selected variables, coefficients, the alpha used, and a
CV mean-squared-error curve summary.

Determinism contract
--------------------
* The caller-visible randomness is fully pinned: ``KFold(shuffle=True,
  random_state=...)`` supplies the CV splits and both estimators run with
  ``selection="cyclic"`` (any other ``selection`` is refused fail-closed, so
  the coordinate-descent order cannot depend on an RNG).
* No global RNG is consumed; rerunning with identical inputs yields
  byte-identical :meth:`LassoSelectionResult.to_json` output.
* On the fixed-alpha ``"lasso"`` path the fitted coefficients and the
  selected set are additionally seed-invariant (``cyclic`` updates ignore
  ``random_state``); only the split-dependent CV MSE summary follows the
  pinned KFold seed.  Because the digest binds that summary as well, a
  rerun with a *different* ``random_state`` reproduces the numbers but not
  the digest bytes.
* The ``"lassocv"`` path is deterministic for a fixed seed, but different
  ``random_state`` values legitimately change the CV folds and may change
  the chosen alpha -- cross-seed byte stability must not be claimed for it
  (see the known-limitation note in the Tool Card test).

Grouped rows
------------
Rows that share a patient are not independent: a random K-fold puts one
patient's ICU stays on both sides of a fold, so the CV error is optimistic
and ``lassocv`` may choose too small a penalty.  ``groups`` (one patient
identifier per row) replaces the shuffled ``KFold`` with ``GroupKFold``,
which keeps each patient's rows in one fold.  Its folds are assigned by
group size, without a random number, so the determinism contract above
holds unchanged.  A grouped result states ``cv_strategy`` and ``cv_group_n``
in its JSON; an ungrouped result's JSON is unchanged, so its digest -- and
the Tool Card's synthetic-origin digest -- reproduce as before.  A caller
whose rows may repeat a patient must pass ``groups``.

Fail-closed inputs: non-numeric dtypes, NaN/inf, empty frames, length
mismatches, unusable CV splits, non-positive alphas, ``selection`` other
than ``"cyclic"``, and incomplete groups or fewer groups than folds are all
refused with :class:`LassoSelectionError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GroupKFold, KFold

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"
DEFAULT_RANDOM_STATE = 0
DEFAULT_CV_SPLITS = 5
DEFAULT_SELECTION_THRESHOLD = 1e-8
UNGROUPED_CV_STRATEGY = "kfold_shuffled"
GROUPED_CV_STRATEGY = "group_kfold"


class LassoSelectionError(ValueError):
    """A Lasso input or configuration the kernel refuses to run on."""


def _require_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise LassoSelectionError(f"{field} must be an int, got {value!r}")
    result = int(value)
    if result < minimum:
        raise LassoSelectionError(f"{field} must be >= {minimum}, got {result}")
    return result


def _require_positive_float(value: object, *, field: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise LassoSelectionError(
            f"{field} must be a positive finite float, got {value!r}"
        ) from None
    if not np.isfinite(result) or result <= 0:
        raise LassoSelectionError(
            f"{field} must be a positive finite float, got {value!r}"
        )
    return result


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
        raise LassoSelectionError(
            f"feature_names length {len(columns)} does not match "
            f"X features {n_features}"
        )
    if any(not name.strip() for name in columns):
        raise LassoSelectionError("feature_names must all be non-blank")
    if len(set(columns)) != len(columns):
        raise LassoSelectionError("feature_names must be unique")
    return tuple(columns)


def _coerce_design_matrix(X: object) -> tuple[np.ndarray, pd.DataFrame | None]:
    frame: pd.DataFrame | None = X if isinstance(X, pd.DataFrame) else None
    if frame is not None:
        if frame.empty or frame.shape[1] == 0:
            raise LassoSelectionError("X must be a non-empty design matrix")
        non_numeric = [
            str(column)
            for column, dtype in frame.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ]
        if non_numeric:
            raise LassoSelectionError(
                "X must be fully numeric; non-numeric columns: "
                + ",".join(sorted(non_numeric))
            )
        values = frame.to_numpy(dtype=float)
    else:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise LassoSelectionError(
                "X must be a numeric 2D array or DataFrame"
            ) from None
        if values.dtype == object or values.ndim != 2:
            raise LassoSelectionError("X must be a numeric 2D array or DataFrame")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise LassoSelectionError("X must be a non-empty design matrix")
    if not np.all(np.isfinite(values)):
        raise LassoSelectionError("X must contain only finite values (no NaN/inf)")
    return values, frame


def _coerce_outcome(y: object, n_samples: int) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
            raise LassoSelectionError("y must be a one-dimensional outcome vector")
        raw = y.to_numpy(dtype=float).ravel()
    else:
        try:
            raw = np.asarray(y, dtype=float).ravel()
        except (TypeError, ValueError):
            raise LassoSelectionError(
                "y must be a numeric one-dimensional outcome vector"
            ) from None
    if raw.shape[0] != n_samples:
        raise LassoSelectionError(
            f"y length {raw.shape[0]} does not match X rows {n_samples}"
        )
    if raw.shape[0] == 0:
        raise LassoSelectionError("y must be non-empty")
    if not np.all(np.isfinite(raw)):
        raise LassoSelectionError("y must contain only finite values (no NaN/inf)")
    return raw


def _coerce_groups(
    groups: object, n_samples: int, n_splits: int
) -> tuple[np.ndarray, int]:
    """One integer code per row for ``GroupKFold``, and the number of groups."""

    values = pd.Series(list(groups), dtype=object)
    if values.shape[0] != n_samples:
        raise LassoSelectionError(
            f"groups has {values.shape[0]} entries for {n_samples} rows"
        )
    if values.isna().any() or values.map(lambda value: str(value).strip() == "").any():
        raise LassoSelectionError("groups must name a group for every row")
    codes, uniques = pd.factorize(values, sort=False)
    if len(uniques) < n_splits:
        raise LassoSelectionError(
            f"need at least as many groups ({len(uniques)}) as cv splits ({n_splits})"
        )
    return np.asarray(codes, dtype=int), int(len(uniques))


@dataclass(frozen=True)
class LassoSelectionResult:
    """Typed Lasso selection output; JSON form is the digest input."""

    method: str
    feature_names: tuple[str, ...]
    coefs: tuple[float, ...]
    selected_vars: tuple[str, ...]
    alpha: float
    alphas: tuple[float, ...]
    cv_mean_mse: tuple[float, ...]
    cv_std_mse: tuple[float, ...]
    selection_threshold: float
    n_samples: int
    n_features: int
    cv_splits: int
    random_state: int
    cv_strategy: str = UNGROUPED_CV_STRATEGY
    cv_group_n: int | None = None

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "method": self.method,
            "feature_names": list(self.feature_names),
            "coefs": [float(value) for value in self.coefs],
            "selected_vars": list(self.selected_vars),
            "alpha": float(self.alpha),
            "alphas": [float(value) for value in self.alphas],
            "cv_mean_mse": [float(value) for value in self.cv_mean_mse],
            "cv_std_mse": [float(value) for value in self.cv_std_mse],
            "selection_threshold": float(self.selection_threshold),
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "cv_splits": int(self.cv_splits),
            "random_state": int(self.random_state),
        }
        if self.cv_strategy != UNGROUPED_CV_STRATEGY:
            payload["cv_strategy"] = self.cv_strategy
            payload["cv_group_n"] = int(self.cv_group_n or 0)
        return payload


def result_sha256(result: LassoSelectionResult) -> str:
    """Return the canonical digest of one selection result."""

    if not isinstance(result, LassoSelectionResult):
        raise TypeError("result_sha256 requires a LassoSelectionResult")
    return canonical_sha256(result.to_json())


def _fold_mse(
    estimator_factory: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[float], list[float]]:
    fold_mses: list[float] = []
    for train_index, valid_index in folds:
        estimator = estimator_factory()
        estimator.fit(x_values[train_index], y_values[train_index])
        predicted = np.asarray(
            estimator.predict(x_values[valid_index]), dtype=float
        ).ravel()
        actual = y_values[valid_index]
        fold_mses.append(float(np.mean((predicted - actual) ** 2)))
    stacked = np.asarray(fold_mses, dtype=float)
    mean = float(stacked.mean())
    std = float(stacked.std(ddof=1)) if stacked.shape[0] > 1 else 0.0
    return [mean], [std]


def lasso_select(
    X: object,
    y: object,
    *,
    feature_names: Sequence[str] | None = None,
    method: str = "lasso",
    alpha: float = 0.1,
    alphas: Sequence[float] | None = None,
    cv: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    selection_threshold: float = DEFAULT_SELECTION_THRESHOLD,
    max_iter: int = 5000,
    tol: float = 1e-4,
    groups: Sequence[object] | None = None,
) -> LassoSelectionResult:
    """Fit a deterministic Lasso and report the selected variables.

    ``method="lasso"`` fits :class:`~sklearn.linear_model.Lasso` at the
    single pinned ``alpha``; ``method="lassocv"`` fits
    :class:`~sklearn.linear_model.LassoCV` over ``alphas`` (or the sklearn
    default grid) with ``KFold(shuffle=True, random_state=random_state)``
    splits.  In both cases a K-fold MSE summary under the same pinned
    splits is reported.  With ``groups`` both use ``GroupKFold`` folds that
    keep each group's rows together (see the module docstring).
    """

    normalized = str(method or "").strip().lower()
    if normalized not in {"lasso", "lassocv"}:
        raise LassoSelectionError(f"method must be 'lasso' or 'lassocv', got {method!r}")
    n_splits = _require_int(cv, field="cv", minimum=2)
    seed = _require_int(random_state, field="random_state", minimum=0)
    try:
        threshold = float(selection_threshold)
    except (TypeError, ValueError):
        raise LassoSelectionError(
            "selection_threshold must be a non-negative finite float"
        ) from None
    if not np.isfinite(threshold) or threshold < 0:
        raise LassoSelectionError(
            "selection_threshold must be a non-negative finite float"
        )
    max_iter_value = _require_int(max_iter, field="max_iter", minimum=100)
    try:
        tol_value = float(tol)
    except (TypeError, ValueError):
        raise LassoSelectionError("tol must be a positive finite float") from None
    if not np.isfinite(tol_value) or tol_value <= 0:
        raise LassoSelectionError("tol must be a positive finite float")

    x_values, x_frame = _coerce_design_matrix(X)
    y_values = _coerce_outcome(y, x_values.shape[0])
    if x_values.shape[0] <= n_splits:
        raise LassoSelectionError(
            f"need more samples ({x_values.shape[0]}) than cv splits ({n_splits})"
        )
    names = _resolve_feature_names(x_frame, feature_names, x_values.shape[1])
    if groups is None:
        cv_strategy, group_n = UNGROUPED_CV_STRATEGY, None
        folds = list(
            KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x_values)
        )
    else:
        codes, group_n = _coerce_groups(groups, x_values.shape[0], n_splits)
        cv_strategy = GROUPED_CV_STRATEGY
        folds = list(
            GroupKFold(n_splits=n_splits).split(x_values, y_values, codes)
        )

    if normalized == "lasso":
        fixed_alpha = _require_positive_float(alpha, field="alpha")
        estimator = Lasso(
            alpha=fixed_alpha,
            max_iter=max_iter_value,
            tol=tol_value,
            random_state=seed,
            selection="cyclic",
        )
        estimator.fit(x_values, y_values)
        coefs = np.asarray(estimator.coef_, dtype=float).ravel()
        mean_mse, std_mse = _fold_mse(
            lambda: Lasso(
                alpha=fixed_alpha,
                max_iter=max_iter_value,
                tol=tol_value,
                random_state=seed,
                selection="cyclic",
            ),
            x_values,
            y_values,
            folds,
        )
        grid = (fixed_alpha,)
        chosen = fixed_alpha
    else:
        if alphas is None:
            grid_values: tuple[float, ...] = ()
            estimator_cv = LassoCV(
                cv=folds,
                max_iter=max_iter_value,
                tol=tol_value,
                random_state=seed,
                selection="cyclic",
            )
        else:
            grid_values = tuple(
                _require_positive_float(item, field="alphas entry")
                for item in list(alphas)
            )
            if not grid_values:
                raise LassoSelectionError("alphas must be non-empty when provided")
            estimator_cv = LassoCV(
                alphas=np.asarray(grid_values, dtype=float),
                cv=folds,
                max_iter=max_iter_value,
                tol=tol_value,
                random_state=seed,
                selection="cyclic",
            )
        estimator_cv.fit(x_values, y_values)
        coefs = np.asarray(estimator_cv.coef_, dtype=float).ravel()
        grid = tuple(float(item) for item in np.ravel(estimator_cv.alphas_))
        chosen = float(estimator_cv.alpha_)
        path = np.asarray(estimator_cv.mse_path_, dtype=float)
        mean_mse = [float(value) for value in path.mean(axis=1)]
        std_mse = [float(value) for value in path.std(axis=1, ddof=1)]

    selected = tuple(
        name for name, coef in zip(names, coefs) if abs(float(coef)) > threshold
    )
    return LassoSelectionResult(
        method=normalized,
        feature_names=names,
        coefs=tuple(float(value) for value in coefs),
        selected_vars=selected,
        alpha=float(chosen),
        alphas=tuple(float(value) for value in grid),
        cv_mean_mse=tuple(float(value) for value in mean_mse),
        cv_std_mse=tuple(float(value) for value in std_mse),
        selection_threshold=float(threshold),
        n_samples=int(x_values.shape[0]),
        n_features=int(x_values.shape[1]),
        cv_splits=int(n_splits),
        random_state=int(seed),
        cv_strategy=cv_strategy,
        cv_group_n=group_n,
    )


__all__ = [
    "DEFAULT_CV_SPLITS",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_SELECTION_THRESHOLD",
    "GROUPED_CV_STRATEGY",
    "TOOL_VERSION",
    "UNGROUPED_CV_STRATEGY",
    "LassoSelectionError",
    "LassoSelectionResult",
    "lasso_select",
    "result_sha256",
]
