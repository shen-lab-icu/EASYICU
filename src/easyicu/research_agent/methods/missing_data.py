"""Missing-data sensitivity (O25).

Two helpers that turn the existing missingness profile into a
reviewer-facing sensitivity story:

* :func:`mice_impute` — legacy name for deterministic ridge single
  imputation of one target with fully observed predictors. This is NOT
  chained-equation multiple imputation and provides no pooled uncertainty.
* :func:`tipping_point_analysis` — given a binary outcome and a
  covariate with missingness, sweeps the missing-data mechanism
  across a grid of imputed values (tipping-point analysis) and
  reports the smallest imputation that flips the sign of the
  primary OR.

Pure numpy / stdlib. Multiple imputation requires a separately validated
sampling and pooling procedure; deterministic filling does not supply one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Deterministic single imputation (legacy public names retained)
# ---------------------------------------------------------------------------


@dataclass
class MICEImputationResult:
    column: str
    n_imputed: int
    imputed_mean: float
    imputed_std: float
    n_iterations: int
    converged: bool

    def to_json(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "n_imputed": self.n_imputed,
            "imputed_mean": self.imputed_mean,
            "imputed_std": self.imputed_std,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "method": "deterministic_ridge_single_imputation",
            "multiple_imputation": False,
        }


def _ridge_fit(X: Any, y: Any, alpha: float = 1.0) -> Any:
    """Small ridge-regression helper without sklearn."""
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or y.size == 0:
        raise ValueError("ridge fit requires aligned observed rows")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("ridge fit requires finite observed values")
    Xt = X.T
    k = X.shape[1]
    beta = np.linalg.solve(Xt @ X + alpha * np.eye(k), Xt @ y)
    return beta


def mice_impute(
    *,
    column: str,
    target: Sequence[float],
    predictors: Sequence[Sequence[float]],
    max_iter: int = 10,
    tol: float = 1e-4,
) -> Tuple[List[float], MICEImputationResult]:
    """Single ridge fit on observed targets, predicting only missing targets.

    ``max_iter`` and ``tol`` are retained for call compatibility; the fit is
    closed form, not iterative. ``converged`` denotes a finite completed fit,
    and ``imputed_std`` is the spread of filled values, not inferential error.
    """
    if np is None:
        raise RuntimeError("mice_impute requires numpy")
    y = np.asarray(target, dtype=float)
    X = np.asarray(predictors, dtype=float)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("target must be 1D and predictors 2D with matching rows")
    if not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    if np.isinf(y).any() or not np.isfinite(X).all():
        raise ValueError("observed target and all predictors must be finite")
    missing = np.isnan(y)
    if not (~missing).any():
        raise ValueError("imputation requires at least one observed target")
    if not missing.any():
        return list(y), MICEImputationResult(
            column=column,
            n_imputed=0,
            imputed_mean=float("nan"),
            imputed_std=float("nan"),
            n_iterations=0,
            converged=True,
        )
    y_filled = y.copy()
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    beta = _ridge_fit(X_design[~missing], y[~missing])
    pred = X_design[missing] @ beta
    if not np.isfinite(pred).all():
        raise ValueError("ridge imputation produced non-finite predictions")
    y_filled[missing] = pred
    result = MICEImputationResult(
        column=column,
        n_imputed=int(missing.sum()),
        imputed_mean=float(np.mean(pred)),
        imputed_std=float(np.std(pred)),
        n_iterations=1,
        converged=True,
    )
    return list(y_filled), result


# ---------------------------------------------------------------------------
# Tipping-point analysis
# ---------------------------------------------------------------------------


@dataclass
class TippingPointResult:
    column: str
    baseline_or: Optional[float]
    grid: List[float] = field(default_factory=list)
    or_by_imputed_value: List[Optional[float]] = field(default_factory=list)
    tipping_point: Optional[float] = None
    note: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "baseline_or": self.baseline_or,
            "grid": list(self.grid),
            "or_by_imputed_value": list(self.or_by_imputed_value),
            "tipping_point": self.tipping_point,
            "note": self.note,
        }


def _logistic_or(x: Any, y: Any) -> Optional[float]:
    """Single-predictor logistic OR via IRLS. Returns None on failure."""
    if np is None:
        return None
    n = len(x)
    if n < 30 or len(np.unique(y)) < 2:
        return None
    X = np.column_stack([np.ones(n), x])
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
        W = p * (1 - p)
        XtWX = X.T @ (W[:, None] * X)
        try:
            inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            return None
        z = eta + (y - p) / np.maximum(W, 1e-9)
        new_beta = inv @ (X.T @ (W * z))
        if np.max(np.abs(new_beta - beta)) < 1e-6:
            beta = new_beta
            break
        beta = new_beta
    return float(math.exp(beta[1]))


def tipping_point_analysis(
    *,
    predictor_column: str,
    predictor_values: Sequence[float],
    outcome: Sequence[int],
    missing_mask: Sequence[bool],
    grid: Optional[Sequence[float]] = None,
) -> TippingPointResult:
    """Sweep the imputed value for the predictor's missing rows.

    For each value ``v`` in ``grid`` (defaults to
    ``[-4σ, -2σ, 0, +2σ, +4σ]`` around the observed mean), set all
    ``missing_mask == True`` rows to ``v`` and refit a tiny logistic
    regression of ``outcome ~ predictor``. The tipping point is the
    smallest absolute ``v`` at which the OR crosses 1.0.
    """
    if np is None:
        return TippingPointResult(
            column=predictor_column, baseline_or=None, note="numpy unavailable"
        )
    x = np.asarray(predictor_values, dtype=float)
    y = np.asarray(outcome, dtype=int)
    m = np.asarray(missing_mask, dtype=bool)
    if not m.any():
        return TippingPointResult(
            column=predictor_column,
            baseline_or=_logistic_or(x, y),
            note="no missing rows; tipping-point analysis skipped",
        )
    obs = x[~m]
    if len(obs) < 30:
        return TippingPointResult(
            column=predictor_column,
            baseline_or=None,
            note="too few observed rows for tipping-point analysis",
        )
    mean = float(np.mean(obs))
    sigma = float(np.std(obs)) or 1.0
    if grid is None:
        grid = [mean - 4 * sigma, mean - 2 * sigma, mean, mean + 2 * sigma, mean + 4 * sigma]
    else:
        grid = list(grid)
    ors: List[Optional[float]] = []
    baseline_or = _logistic_or(x[~m], y[~m])
    tipping: Optional[float] = None
    for v in grid:
        x_v = x.copy()
        x_v[m] = v
        orv = _logistic_or(x_v, y)
        ors.append(orv)
        if baseline_or is not None and orv is not None:
            if tipping is None and (
                (baseline_or > 1 and orv < 1) or (baseline_or < 1 and orv > 1)
            ):
                tipping = float(v)
    return TippingPointResult(
        column=predictor_column,
        baseline_or=baseline_or,
        grid=list(grid),
        or_by_imputed_value=ors,
        tipping_point=tipping,
    )


__all__ = [
    "MICEImputationResult",
    "TippingPointResult",
    "mice_impute",
    "tipping_point_analysis",
]
