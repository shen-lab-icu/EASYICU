"""Doubly robust ATE for a binary outcome: AIPTW (AIPW) with influence-function SE.

When treatment ``A`` is binary and the outcome ``Y`` is binary, the average
treatment effect ``ATE = E[Y(1) - Y(0)]`` can be estimated without committing
to a single modelling path. Let ``e(X) = P(A = 1 | X)`` be the propensity
score, ``m1(X) = E[Y | A = 1, X]`` and ``m0(X) = E[Y | A = 0, X]`` the outcome
regressions. The augmented inverse-probability-weighted (AIPTW/AIPW) score for
subject ``i`` is::

    tau_i = (m1_i - m0_i)
            + A_i * (Y_i - m1_i) / e_i
            - (1 - A_i) * (Y_i - m0_i) / (1 - e_i)

``ATE_hat = mean(tau_i)`` is *doubly robust*: it stays consistent when either
the propensity model or the outcome model is correct (Bang and Robins, 2005;
Lunceford and Davidian, 2004). Uncertainty comes from the influence function —
``SE = sd(tau, ddof=1) / sqrt(n)`` with a normal Wald interval — computed in
pure numpy from the fitted values, so the point estimate and the SE are fully
deterministic given fixed model seeds.

Mechanical choices in this skeleton
-----------------------------------
* Propensity: ``sklearn.linear_model.LogisticRegression`` (lbfgs) with a fixed
  ``random_state`` (default 0).
* Outcome: per-arm ``LogisticRegression`` with the same fixed seed
  (``outcome_model="logistic"``), or per-arm ``LinearRegression`` clipped to
  ``[0, 1]`` (``outcome_model="linear"``, deterministic, no seed needed).
* Positivity is fail closed: propensity values outside ``[1e-6, 1 - 1e-6]`` or
  outside the declared trim window (default ``[0.025, 0.975]``) raise
  :class:`DoublyRobustError` and report the would-be trim proportion instead of
  silently trimming or stabilising weights. Use :func:`positivity_diagnostics`
  for a non-raising report.

What is NOT in this module (scope boundary)
-------------------------------------------
The time-varying marginal structural model (MSM) full estimator is
intentionally absent: it needs the complete longitudinal machinery that this
point-treatment skeleton does not own — per-visit treatment/confounder
histories, a sequential-conditional-exchangeability statement per visit, pooled
stabilised inverse-probability weights over follow-up, a pooled MSM fit, and
weight-truncation diagnostics across visits (a longitudinal g-formula would be
the alternative engine with its own Monte-Carlo integration). Dynamic
treatment regimes (DTRs, "what rule is optimal") are likewise out of scope:
this module estimates the contrast of two fixed point treatments only.

Evidence ceiling: ``analysis_only``. A small standard error means the *statistical*
fit is precise; it cannot substitute for human confirmation of the causal
identification assumptions (consistency, conditional exchangeability given the
modelled ``X``, positivity — see also the target-trial checklist,
"需人工确认的因果假设清单"). Nothing here is reportable as a causal finding.

References
----------
Bang H, Robins JM. "Doubly robust estimation in missing data and causal
inference models." *Biometrics* 2005;61:962-973.
Lunceford JK, Davidian M. "Stratification and weighting via the propensity
score in estimation of causal treatment effects." *Stat Med* 2004;23:2937-2960.
Hernan MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC, 2020,
Ch. 13-14 (IP weighting, standardisation, doubly robust go together).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

#: Evidence ceiling for every product of this module.
EVIDENCE_CEILING = "analysis_only"

#: Hard propensity bounds — values at or beyond these mean the weights
#: ``1/e`` / ``1/(1-e)`` are numerically degenerate. Fail closed.
_PS_HARD_EPS = 1e-6

#: Fixed optimiser budget so fits are deterministic and complete quickly.
_MAX_ITER = 5000

OutcomeModel = Literal["logistic", "linear"]


class DoublyRobustError(ValueError):
    """The doubly robust estimator refuses to proceed (fail closed)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "doubly_robust_invalid",
        trim_proportion: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.trim_proportion = float(trim_proportion)


@dataclass(frozen=True)
class AIPTWResult:
    """Point-treatment ATE with an influence-function Wald interval."""

    ate: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float
    n: int
    n_treated: int
    n_control: int
    outcome_model: str
    random_state: int
    ps_trim_low: float
    ps_trim_high: float
    trim_proportion: float
    ps_min: float
    ps_max: float
    ps_mean: float
    evidence_ceiling: str = EVIDENCE_CEILING


# ---------------------------------------------------------------------------
# Input validation (fail closed, no silent coercion).
# ---------------------------------------------------------------------------


def _as_float_matrix(values: Any, *, label: str, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise DoublyRobustError(
            f"{label} must have {n} rows, got shape {arr.shape}",
            code="doubly_robust_shape_mismatch",
        )
    if not bool(np.isfinite(arr).all()):
        raise DoublyRobustError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="doubly_robust_covariates_not_finite",
        )
    return arr


def _as_binary_vector(values: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise DoublyRobustError(
            f"{label} must be one-dimensional, got shape {arr.shape}",
            code="doubly_robust_shape_mismatch",
        )
    flat = arr.ravel()
    if flat.shape[0] == 0:
        raise DoublyRobustError(
            f"{label} is empty",
            code="doubly_robust_empty_input",
        )
    try:
        numeric = flat.astype(float)
    except (TypeError, ValueError) as exc:
        raise DoublyRobustError(
            f"{label} must be binary 0/1",
            code="doubly_robust_not_binary",
        ) from exc
    if not bool(np.isfinite(numeric).all()):
        raise DoublyRobustError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="doubly_robust_not_finite",
        )
    unique = set(np.unique(numeric).tolist())
    if unique - {0.0, 1.0}:
        raise DoublyRobustError(
            f"{label} must be binary 0/1, got values {sorted(unique)[:5]}",
            code="doubly_robust_not_binary",
        )
    return numeric.astype(float)


def _check_trim_bounds(ps_trim: Any) -> tuple[float, float]:
    try:
        low, high = (float(ps_trim[0]), float(ps_trim[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise DoublyRobustError(
            "ps_trim must be a (low, high) pair",
            code="doubly_robust_trim_bounds_invalid",
        ) from exc
    if not (0.0 < low < high < 1.0):
        raise DoublyRobustError(
            f"ps_trim must satisfy 0 < low < high < 1, got ({low}, {high})",
            code="doubly_robust_trim_bounds_invalid",
        )
    return low, high


# ---------------------------------------------------------------------------
# Model fits (fixed seeds, deterministic).
# ---------------------------------------------------------------------------


def _fit_propensity(
    X: np.ndarray, A: np.ndarray, *, random_state: int
) -> np.ndarray:
    model = LogisticRegression(max_iter=_MAX_ITER, random_state=int(random_state))
    model.fit(X, A)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _fit_outcome(
    X_arm: np.ndarray,
    y_arm: np.ndarray,
    X_all: np.ndarray,
    *,
    outcome_model: str,
    random_state: int,
    label: str,
) -> np.ndarray:
    if outcome_model == "logistic":
        if len(set(np.unique(y_arm).tolist())) < 2:
            raise DoublyRobustError(
                f"outcome is constant within {label}; "
                "logistic outcome regression cannot be fit",
                code="doubly_robust_outcome_single_class",
            )
        model = LogisticRegression(max_iter=_MAX_ITER, random_state=int(random_state))
        model.fit(X_arm, y_arm)
        return np.asarray(model.predict_proba(X_all)[:, 1], dtype=float)
    if outcome_model == "linear":
        model = LinearRegression()
        model.fit(X_arm, y_arm)
        # Clip to the outcome support: the estimand is a probability contrast.
        return np.clip(np.asarray(model.predict(X_all), dtype=float), 0.0, 1.0)
    raise DoublyRobustError(
        f"outcome_model must be 'logistic' or 'linear', got {outcome_model!r}",
        code="doubly_robust_outcome_model_invalid",
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def positivity_diagnostics(
    X: Any,
    A: Any,
    *,
    random_state: int = 0,
    ps_trim: tuple[float, float] = (0.025, 0.975),
) -> dict[str, Any]:
    """Report the fitted propensity distribution without raising.

    Fits the same fixed-seed propensity model as :func:`aiptw_ate` and returns
    ``ps_min/max/mean``, the trim window, the count and proportion of fitted
    scores outside it, and any hard-bound violations. Use this to inspect
    overlap *before* calling the fail-closed estimator.
    """

    low, high = _check_trim_bounds(ps_trim)
    a_vec = _as_binary_vector(A, label="treatment")
    x_mat = _as_float_matrix(X, label="covariates", n=int(a_vec.shape[0]))
    ps = _fit_propensity(x_mat, a_vec, random_state=int(random_state))
    outside = (ps < low) | (ps > high)
    hard = (~np.isfinite(ps)) | (ps <= _PS_HARD_EPS) | (ps >= 1.0 - _PS_HARD_EPS)
    return {
        "n": int(a_vec.shape[0]),
        "ps_min": float(np.min(ps)),
        "ps_max": float(np.max(ps)),
        "ps_mean": float(np.mean(ps)),
        "ps_trim_low": low,
        "ps_trim_high": high,
        "n_trimmed": int(outside.sum()),
        "trim_proportion": float(outside.mean()),
        "n_hard_violations": int(hard.sum()),
        "random_state": int(random_state),
    }


def aiptw_ate(
    X: Any,
    A: Any,
    Y: Any,
    *,
    ps_trim: tuple[float, float] = (0.025, 0.975),
    random_state: int = 0,
    outcome_model: OutcomeModel = "logistic",
    ci_level: float = 0.95,
) -> AIPTWResult:
    """AIPTW average treatment effect for a binary outcome (fail closed).

    Parameters
    ----------
    X:
        Baseline covariates, shape ``(n, k)`` (1-D input is treated as a
        single covariate). Must be finite and complete.
    A:
        Binary treatment vector (0/1).
    Y:
        Binary outcome vector (0/1).
    ps_trim:
        Acceptable propensity window. Any fitted score outside it raises
        :class:`DoublyRobustError` instead of being trimmed — the caller must
        resolve the positivity problem, not the estimator.
    random_state:
        Fixed seed for the propensity and logistic-outcome fits.
    outcome_model:
        ``"logistic"`` (per-arm logistic regression) or ``"linear"``
        (per-arm linear regression clipped to ``[0, 1]``).
    ci_level:
        Wald interval level in ``(0, 1)``.

    Returns :class:`AIPTWResult` with the point estimate, the
    influence-function SE, and positivity diagnostics. ``trim_proportion`` is
    0.0 on success (any positive value raises).
    """

    if outcome_model not in ("logistic", "linear"):
        raise DoublyRobustError(
            f"outcome_model must be 'logistic' or 'linear', got {outcome_model!r}",
            code="doubly_robust_outcome_model_invalid",
        )
    level = float(ci_level)
    if not 0.0 < level < 1.0:
        raise DoublyRobustError(
            f"ci_level must be in (0, 1), got {ci_level!r}",
            code="doubly_robust_ci_level_invalid",
        )
    low, high = _check_trim_bounds(ps_trim)
    seed = int(random_state)

    a_vec = _as_binary_vector(A, label="treatment")
    y_vec = _as_binary_vector(Y, label="outcome")
    if a_vec.shape[0] != y_vec.shape[0]:
        raise DoublyRobustError(
            "treatment and outcome must have equal length",
            code="doubly_robust_shape_mismatch",
        )
    n = int(a_vec.shape[0])
    x_mat = _as_float_matrix(X, label="covariates", n=n)

    treated = a_vec == 1.0
    n_treated = int(treated.sum())
    n_control = int(n - n_treated)
    if n_treated == 0 or n_control == 0:
        raise DoublyRobustError(
            "treatment must vary (one arm is empty)",
            code="doubly_robust_single_arm",
        )
    if n_treated < 2 or n_control < 2:
        raise DoublyRobustError(
            "each treatment arm needs at least two observations",
            code="doubly_robust_arm_too_small",
        )

    ps = _fit_propensity(x_mat, a_vec, random_state=seed)
    if not bool(np.isfinite(ps).all()):
        raise DoublyRobustError(
            "propensity fit produced non-finite scores",
            code="doubly_robust_propensity_not_finite",
            trim_proportion=1.0,
        )
    hard = (ps <= _PS_HARD_EPS) | (ps >= 1.0 - _PS_HARD_EPS)
    outside = (ps < low) | (ps > high)
    trim_proportion = float(outside.mean())
    if bool(hard.any()):
        raise DoublyRobustError(
            f"positivity violated: {int(hard.sum())}/{n} propensity scores "
            f"at or beyond [{_PS_HARD_EPS}, {1.0 - _PS_HARD_EPS}] "
            f"(trim-window proportion {trim_proportion:.4f}); refusing to estimate",
            code="doubly_robust_propensity_out_of_bounds",
            trim_proportion=trim_proportion,
        )
    if bool(outside.any()):
        raise DoublyRobustError(
            f"positivity violated: {int(outside.sum())}/{n} propensity scores "
            f"outside trim window [{low}, {high}] "
            f"(trim proportion {trim_proportion:.4f}); refusing to trim silently",
            code="doubly_robust_positivity_trim_triggered",
            trim_proportion=trim_proportion,
        )

    mu1 = _fit_outcome(
        x_mat[treated],
        y_vec[treated],
        x_mat,
        outcome_model=outcome_model,
        random_state=seed,
        label="the treated arm",
    )
    mu0 = _fit_outcome(
        x_mat[~treated],
        y_vec[~treated],
        x_mat,
        outcome_model=outcome_model,
        random_state=seed,
        label="the control arm",
    )
    if not bool(np.isfinite(mu1).all()) or not bool(np.isfinite(mu0).all()):
        raise DoublyRobustError(
            "outcome regression produced non-finite predictions",
            code="doubly_robust_outcome_not_finite",
        )

    scores = (mu1 - mu0) + treated * (y_vec - mu1) / ps - (~treated) * (
        y_vec - mu0
    ) / (1.0 - ps)
    if not bool(np.isfinite(scores).all()):
        raise DoublyRobustError(
            "AIPTW influence scores are non-finite",
            code="doubly_robust_scores_not_finite",
        )
    ate = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    return AIPTWResult(
        ate=ate,
        se=se,
        ci_low=ate - z * se,
        ci_high=ate + z * se,
        ci_level=level,
        n=n,
        n_treated=n_treated,
        n_control=n_control,
        outcome_model=outcome_model,
        random_state=seed,
        ps_trim_low=low,
        ps_trim_high=high,
        trim_proportion=0.0,
        ps_min=float(np.min(ps)),
        ps_max=float(np.max(ps)),
        ps_mean=float(np.mean(ps)),
    )


__all__ = [
    "EVIDENCE_CEILING",
    "AIPTWResult",
    "DoublyRobustError",
    "OutcomeModel",
    "aiptw_ate",
    "positivity_diagnostics",
]
