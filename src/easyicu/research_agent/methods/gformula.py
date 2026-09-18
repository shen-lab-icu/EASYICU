"""Point-intervention parametric g-formula: standardisation over an outcome model.

For a binary point treatment ``A`` and baseline covariates ``X``, the
parametric g-formula identifies ``ATE = E[Y(1) - Y(0)]`` by fitting one outcome
regression on ``(A, X)`` and standardising its counterfactual predictions over
the empirical covariate distribution::

    mu1_i = m(A=1, X_i),   mu0_i = m(A=0, X_i),   ATE_hat = mean(mu1 - mu0)

with ``m`` a logistic fit (binary outcome) or a linear fit (continuous
outcome). Uncertainty is a percentile bootstrap with a fixed seed; the
bootstrap SE is the replicate standard deviation (``ddof=1``).

Assumption gate (fail closed, mirroring
:mod:`easyicu.research_agent.methods.mediation`): the caller must pass both
``require_exchangeability=True`` (no unmeasured confounding given the modelled
``X``) and ``require_correct_specification=True`` (the outcome regression is
correctly specified). Anything else raises :class:`GFormulaError` — the kernel
refuses to print causal numbers while its identifying assumptions are
explicitly disavowed.

Mechanical choices in this skeleton
-----------------------------------
* Outcome regression on the design ``[A, X]``: ``sklearn`` logistic
  (``max_iter=5000``, fixed ``random_state``) for ``outcome_model="logistic"``,
  or ``sklearn`` linear regression (deterministic, no seed needed) for
  ``outcome_model="linear"``. The linear path does NOT clip predictions: the
  outcome may be continuous and unbounded.
* Positivity is screened mechanically, mirroring
  :mod:`easyicu.research_agent.methods.doubly_robust`: a fixed-seed logistic
  propensity ``P(A=1 | X)`` outside the hard bounds ``[1e-6, 1 - 1e-6]`` or
  outside the declared trim window (default ``[0.025, 0.975]``) raises
  :class:`GFormulaError` with the would-be trim proportion. Counterfactual
  predictions that extrapolate beyond observed treatment support are refused,
  not silently reported.
* Bootstrap replicates that cannot be fit (single-class resample under the
  logistic path, rank-deficient design) are skipped, not imputed; the usable
  count is recorded on the result.

Identification assumptions (untestable from the data — stated, not verified)
----------------------------------------------------------------------------
1. **Conditional exchangeability** given the modelled ``X`` (gated by
   ``require_exchangeability=True``).
2. **Correct outcome-model specification** (gated by
   ``require_correct_specification=True``).
3. **Consistency** and **positivity** (positivity screened mechanically above;
   the screen sees fitted support only).

What is NOT in this module (scope boundary)
--------------------------------------------
* The longitudinal (time-varying) g-formula: no visit loop, no Monte-Carlo
  integration over time-varying confounders, no treatment-confounder feedback.
  Point intervention only.
* Stochastic or dynamic regimes: both counterfactuals set ``A`` to a fixed
  constant for the full sample.
* Pooled fitting of time-varying confounding and multi-stage optimal-rule
  estimation.

Evidence ceiling: ``analysis_only``. A tight bootstrap interval means the
*statistical* fit is precise; it cannot substitute for human confirmation of
the assumptions above. Nothing here is reportable as a causal finding.

References
----------
Robins JM. "A new approach to causal inference in mortality studies with a
sustained exposure period — application to control of the healthy worker
survivor effect." *Math Model* 1986;7:1393-1512.
Hernan MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC, 2020,
Ch. 13-14 (standardisation and the g-formula).
Bang H, Robins JM. "Doubly robust estimation in missing data and causal
inference models." *Biometrics* 2005;61:962-973.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

#: Evidence ceiling for every product of this module.
EVIDENCE_CEILING = "analysis_only"

#: Hard propensity bounds for the mechanical positivity screen. Fail closed.
#: Mirrors the doubly-robust kernel convention.
_PS_HARD_EPS = 1e-6

#: Fixed optimiser budget so fits are deterministic and complete quickly.
_MAX_ITER = 5000

#: Known limitations, repeated verbatim on every result so no caller can miss
#: the scope boundary.
LIMITATIONS: tuple[str, ...] = (
    "Point intervention only: no longitudinal g-formula, no Monte-Carlo "
    "integration over time-varying confounders.",
    "No stochastic or dynamic regimes: both counterfactuals fix A to a "
    "constant for the full sample.",
    "No multi-stage optimal-rule estimation.",
    "Uncertainty is a fixed-seed percentile bootstrap; failed replicates "
    "are skipped, not imputed.",
)

OutcomeModel = Literal["logistic", "linear"]


class GFormulaError(ValueError):
    """The g-formula kernel refuses to proceed (fail closed)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "gformula_invalid",
        trim_proportion: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.trim_proportion = float(trim_proportion)


@dataclass(frozen=True)
class GFormulaResult:
    """Point-intervention g-formula ATE with a bootstrap interval."""

    ate: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float
    mu1_mean: float
    mu0_mean: float
    n: int
    n_treated: int
    n_control: int
    n_covariates: int
    outcome_model: str
    n_bootstrap: int
    n_successful: int
    random_state: int
    ps_trim_low: float
    ps_trim_high: float
    trim_proportion: float
    ps_min: float
    ps_max: float
    ps_mean: float
    limitations: tuple[str, ...] = LIMITATIONS
    evidence_ceiling: str = EVIDENCE_CEILING
    method: str = "point_gformula_standardisation"

    def to_json(self) -> dict[str, Any]:
        return {
            "ate": self.ate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "mu1_mean": self.mu1_mean,
            "mu0_mean": self.mu0_mean,
            "n": self.n,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_covariates": self.n_covariates,
            "outcome_model": self.outcome_model,
            "n_bootstrap": self.n_bootstrap,
            "n_successful": self.n_successful,
            "random_state": self.random_state,
            "ps_trim_low": self.ps_trim_low,
            "ps_trim_high": self.ps_trim_high,
            "trim_proportion": self.trim_proportion,
            "ps_min": self.ps_min,
            "ps_max": self.ps_max,
            "ps_mean": self.ps_mean,
            "require_exchangeability": True,
            "require_correct_specification": True,
            "limitations": list(self.limitations),
            "evidence_ceiling": self.evidence_ceiling,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Input validation (fail closed, no silent coercion).
# ---------------------------------------------------------------------------


def _as_float_matrix(values: Any, *, label: str, n: int) -> np.ndarray:
    if values is None:
        return np.zeros((n, 0), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise GFormulaError(
            f"{label} must have {n} rows, got shape {arr.shape}",
            code="gformula_shape_mismatch",
        )
    if not bool(np.isfinite(arr).all()):
        raise GFormulaError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="gformula_covariates_not_finite",
        )
    return arr


def _as_binary_vector(values: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise GFormulaError(
            f"{label} must be one-dimensional, got shape {arr.shape}",
            code="gformula_shape_mismatch",
        )
    flat = arr.ravel()
    if flat.shape[0] == 0:
        raise GFormulaError(f"{label} is empty", code="gformula_empty_input")
    try:
        numeric = flat.astype(float)
    except (TypeError, ValueError) as exc:
        raise GFormulaError(
            f"{label} must be binary 0/1", code="gformula_not_binary"
        ) from exc
    if not bool(np.isfinite(numeric).all()):
        raise GFormulaError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="gformula_not_finite",
        )
    unique = set(np.unique(numeric).tolist())
    if unique - {0.0, 1.0}:
        raise GFormulaError(
            f"{label} must be binary 0/1, got values {sorted(unique)[:5]}",
            code="gformula_not_binary",
        )
    return numeric.astype(float)


def _as_finite_vector(values: Any, *, label: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise GFormulaError(
            f"{label} must be numeric", code="gformula_outcome_invalid"
        ) from exc
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise GFormulaError(
            f"{label} must be a non-empty one-dimensional vector",
            code="gformula_shape_mismatch",
        )
    if not bool(np.isfinite(arr).all()):
        raise GFormulaError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="gformula_not_finite",
        )
    return arr


def _check_trim_bounds(ps_trim: Any) -> tuple[float, float]:
    try:
        low, high = (float(ps_trim[0]), float(ps_trim[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise GFormulaError(
            "ps_trim must be a (low, high) pair",
            code="gformula_trim_bounds_invalid",
        ) from exc
    if not (0.0 < low < high < 1.0):
        raise GFormulaError(
            f"ps_trim must satisfy 0 < low < high < 1, got ({low}, {high})",
            code="gformula_trim_bounds_invalid",
        )
    return low, high


# ---------------------------------------------------------------------------
# Model fits (fixed seeds, deterministic).
# ---------------------------------------------------------------------------


def _fit_propensity(
    X: np.ndarray, A: np.ndarray, *, random_state: int
) -> np.ndarray:
    model = LogisticRegression(max_iter=_MAX_ITER, random_state=int(random_state))
    model.fit(X if X.shape[1] > 0 else np.zeros((X.shape[0], 1)), A)
    return np.asarray(model.predict_proba(
        X if X.shape[1] > 0 else np.zeros((X.shape[0], 1))
    )[:, 1], dtype=float)


def _fit_outcome(
    design: np.ndarray,
    y: np.ndarray,
    *,
    outcome_model: str,
    random_state: int,
) -> Any:
    if outcome_model == "logistic":
        if len(set(np.unique(y).tolist())) < 2:
            raise GFormulaError(
                "outcome is constant; logistic outcome regression cannot be fit",
                code="gformula_outcome_single_class",
            )
        model = LogisticRegression(max_iter=_MAX_ITER, random_state=int(random_state))
        try:
            model.fit(design, y)
        except Exception as exc:
            raise GFormulaError(
                f"logistic outcome model failed to fit: {exc}",
                code="gformula_outcome_fit_failed",
            ) from exc
        n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
        if n_iter >= _MAX_ITER:
            raise GFormulaError(
                "logistic outcome model failed to converge; refusing to report",
                code="gformula_outcome_not_converged",
            )
        return model
    if outcome_model == "linear":
        model = LinearRegression()
        try:
            model.fit(design, y)
        except Exception as exc:
            raise GFormulaError(
                f"linear outcome model failed to fit: {exc}",
                code="gformula_outcome_fit_failed",
            ) from exc
        return model
    raise GFormulaError(
        f"outcome_model must be 'logistic' or 'linear', got {outcome_model!r}",
        code="gformula_outcome_model_invalid",
    )


def _predict_outcome(model: Any, design: np.ndarray, *, outcome_model: str) -> np.ndarray:
    if outcome_model == "logistic":
        preds = np.asarray(model.predict_proba(design)[:, 1], dtype=float)
    else:
        preds = np.asarray(model.predict(design), dtype=float)
    if not bool(np.isfinite(preds).all()):
        raise GFormulaError(
            "outcome regression produced non-finite predictions",
            code="gformula_outcome_not_finite",
        )
    return preds


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def gformula_ate(
    X: Any,
    A: Any,
    Y: Any,
    *,
    outcome_model: OutcomeModel = "logistic",
    require_exchangeability: bool = False,
    require_correct_specification: bool = False,
    ps_trim: tuple[float, float] = (0.025, 0.975),
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 0,
) -> GFormulaResult:
    """Point-intervention g-formula ATE by outcome-model standardisation.

    Parameters
    ----------
    X:
        Baseline covariates, shape ``(n, k)`` (1-D input is treated as a
        single covariate; ``None`` means no covariates). Must be finite and
        complete.
    A:
        Binary point-treatment vector (0/1).
    Y:
        Outcome vector: binary 0/1 under ``outcome_model="logistic"``,
        any finite values under ``outcome_model="linear"`` (no clipping).
    outcome_model:
        ``"logistic"`` or ``"linear"`` outcome regression on ``[A, X]``.
    require_exchangeability, require_correct_specification:
        Must both be ``True``; the kernel raises :class:`GFormulaError`
        otherwise (fail closed).
    ps_trim:
        Mechanical positivity screen on the fixed-seed propensity
        ``P(A=1 | X)``. Any fitted score outside it (or at the hard bounds)
        raises instead of extrapolating silently.
    n_bootstrap:
        Number of fixed-seed percentile-bootstrap replicates.
    ci_level:
        Interval level in ``(0, 1)``.
    random_state:
        Fixed seed for the outcome fit, the positivity-screen fit, and the
        bootstrap resampling.
    """

    if require_exchangeability is not True:
        raise GFormulaError(
            "refusing to estimate: g-formula requires an explicit "
            "require_exchangeability=True declaration (no unmeasured "
            "confounding given the modelled X)",
            code="gformula_assumption_not_declared",
        )
    if require_correct_specification is not True:
        raise GFormulaError(
            "refusing to estimate: g-formula requires an explicit "
            "require_correct_specification=True declaration (outcome model "
            "correctly specified)",
            code="gformula_assumption_not_declared",
        )
    if outcome_model not in ("logistic", "linear"):
        raise GFormulaError(
            f"outcome_model must be 'logistic' or 'linear', got {outcome_model!r}",
            code="gformula_outcome_model_invalid",
        )
    if isinstance(n_bootstrap, bool) or not isinstance(
        n_bootstrap, (int, np.integer)
    ):
        raise GFormulaError(
            "n_bootstrap must be an integer", code="gformula_bootstrap_invalid"
        )
    n_boot = int(n_bootstrap)
    if n_boot < 1:
        raise GFormulaError(
            "n_bootstrap must be a positive integer",
            code="gformula_bootstrap_invalid",
        )
    if isinstance(ci_level, bool) or not isinstance(
        ci_level, (int, float, np.floating)
    ):
        raise GFormulaError(
            "ci_level must be a number in (0, 1)", code="gformula_ci_level_invalid"
        )
    level = float(ci_level)
    if not 0.0 < level < 1.0:
        raise GFormulaError(
            f"ci_level must be in (0, 1), got {ci_level!r}",
            code="gformula_ci_level_invalid",
        )
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise GFormulaError(
            "random_state must be an integer", code="gformula_seed_invalid"
        )
    seed = int(random_state)
    low, high = _check_trim_bounds(ps_trim)

    a_vec = _as_binary_vector(A, label="treatment")
    n = int(a_vec.shape[0])
    x_mat = _as_float_matrix(X, label="covariates", n=n)
    if outcome_model == "logistic":
        y_vec = _as_binary_vector(Y, label="outcome")
    else:
        y_vec = _as_finite_vector(Y, label="outcome")
    if y_vec.shape[0] != n:
        raise GFormulaError(
            "treatment and outcome must have equal length",
            code="gformula_shape_mismatch",
        )

    treated = a_vec == 1.0
    n_treated = int(treated.sum())
    n_control = int(n - n_treated)
    if n_treated == 0 or n_control == 0:
        raise GFormulaError(
            "treatment must vary (one arm is empty)",
            code="gformula_single_arm",
        )
    if outcome_model == "logistic" and (
        not bool((y_vec == 1.0).any()) or not bool((y_vec == 0.0).any())
    ):
        raise GFormulaError(
            "binary outcome needs both classes observed",
            code="gformula_outcome_single_class",
        )
    if x_mat.shape[1] > 0 and (
        np.linalg.matrix_rank(
            np.column_stack([np.ones(n), a_vec, x_mat])
        )
        < 2 + x_mat.shape[1]
    ):
        raise GFormulaError(
            "outcome design [A, X] is rank-deficient; effects are unidentified",
            code="gformula_design_rank_deficient",
        )

    # Mechanical positivity screen (fail closed, doubly-robust convention).
    ps = _fit_propensity(x_mat, a_vec, random_state=seed)
    if not bool(np.isfinite(ps).all()):
        raise GFormulaError(
            "propensity fit produced non-finite scores",
            code="gformula_propensity_not_finite",
            trim_proportion=1.0,
        )
    hard = (ps <= _PS_HARD_EPS) | (ps >= 1.0 - _PS_HARD_EPS)
    outside = (ps < low) | (ps > high)
    trim_proportion = float(outside.mean())
    if bool(hard.any()):
        raise GFormulaError(
            f"positivity violated: {int(hard.sum())}/{n} propensity scores "
            f"at or beyond [{_PS_HARD_EPS}, {1.0 - _PS_HARD_EPS}] "
            f"(trim-window proportion {trim_proportion:.4f}); refusing to estimate",
            code="gformula_propensity_out_of_bounds",
            trim_proportion=trim_proportion,
        )
    if bool(outside.any()):
        raise GFormulaError(
            f"positivity violated: {int(outside.sum())}/{n} propensity scores "
            f"outside trim window [{low}, {high}] "
            f"(trim proportion {trim_proportion:.4f}); refusing to extrapolate",
            code="gformula_positivity_trim_triggered",
            trim_proportion=trim_proportion,
        )

    def _standardised_ate(
        design: np.ndarray, response: np.ndarray
    ) -> tuple[float, float, float]:
        model = _fit_outcome(
            design, response, outcome_model=outcome_model, random_state=seed
        )
        cols = design[:, 1:]
        mu1 = _predict_outcome(
            model,
            np.column_stack([np.ones(n), cols]) if cols.shape[1] > 0
            else np.ones((n, 1)),
            outcome_model=outcome_model,
        )
        mu0 = _predict_outcome(
            model,
            np.column_stack([np.zeros(n), cols]) if cols.shape[1] > 0
            else np.zeros((n, 1)),
            outcome_model=outcome_model,
        )
        return float(np.mean(mu1 - mu0)), float(np.mean(mu1)), float(np.mean(mu0))

    full_design = (
        np.column_stack([a_vec, x_mat])
        if x_mat.shape[1] > 0
        else a_vec.reshape(-1, 1)
    )
    ate, mu1_mean, mu0_mean = _standardised_ate(full_design, y_vec)

    # Percentile bootstrap (fixed seed). Failed replicates are skipped, not
    # imputed; the usable count is recorded on the result.
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a_b, y_b = a_vec[idx], y_vec[idx]
        x_b = x_mat[idx]
        if len(set(np.unique(a_b).tolist())) < 2:
            continue
        if outcome_model == "logistic" and (
            not bool((y_b == 1.0).any()) or not bool((y_b == 0.0).any())
        ):
            continue
        design_b = (
            np.column_stack([a_b, x_b])
            if x_b.shape[1] > 0
            else a_b.reshape(-1, 1)
        )
        if x_b.shape[1] > 0 and (
            np.linalg.matrix_rank(
                np.column_stack([np.ones(n), design_b])
            )
            < 2 + x_b.shape[1]
        ):
            continue
        try:
            model_b = _fit_outcome(
                design_b, y_b, outcome_model=outcome_model, random_state=seed
            )
            cols_b = design_b[:, 1:]
            mu1_b = _predict_outcome(
                model_b,
                np.column_stack([np.ones(n), cols_b]) if cols_b.shape[1] > 0
                else np.ones((n, 1)),
                outcome_model=outcome_model,
            )
            mu0_b = _predict_outcome(
                model_b,
                np.column_stack([np.zeros(n), cols_b]) if cols_b.shape[1] > 0
                else np.zeros((n, 1)),
                outcome_model=outcome_model,
            )
        except GFormulaError:
            continue
        boot.append(float(np.mean(mu1_b - mu0_b)))
    n_ok = len(boot)
    if n_ok == 0:
        raise GFormulaError(
            "bootstrap produced no usable replicate",
            code="gformula_bootstrap_empty",
        )
    alpha = 1.0 - level
    boot_arr = np.asarray(boot, dtype=float)
    ci_low = float(np.quantile(boot_arr, alpha / 2.0))
    ci_high = float(np.quantile(boot_arr, 1.0 - alpha / 2.0))
    se = float(np.std(boot_arr, ddof=1)) if n_ok > 1 else 0.0

    return GFormulaResult(
        ate=ate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=level,
        mu1_mean=mu1_mean,
        mu0_mean=mu0_mean,
        n=n,
        n_treated=n_treated,
        n_control=n_control,
        n_covariates=int(x_mat.shape[1]),
        outcome_model=outcome_model,
        n_bootstrap=n_boot,
        n_successful=n_ok,
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
    "LIMITATIONS",
    "GFormulaError",
    "GFormulaResult",
    "OutcomeModel",
    "gformula_ate",
]
