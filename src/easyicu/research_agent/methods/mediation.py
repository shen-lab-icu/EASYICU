"""Natural direct/indirect effects via the product method (analysis only).

For an exposure ``X``, a mediator ``M`` and an outcome ``Y`` (chain
``X -> M -> Y``) with baseline covariates ``C``, the product method fits two
regressions and multiplies coefficients (Baron & Kenny 1986; VanderWeele &
Vansteelandt 2009, 2010):

* Mediator model (OLS): ``M = a0 + a * X + C @ gamma + e``.
* Outcome model, linear ``Y`` (OLS): ``Y = b0 + c' * X + b * M + C @ theta + e``.
* Outcome model, binary ``Y`` (logistic via sklearn, fixed ``random_state``):
  ``logit P(Y=1) = b0 + c' * X + b * M + C @ theta``.

Effects:

* Natural direct effect (NDE) = ``c'``.
* Natural indirect effect (NIE) = ``a * b`` (the product).
* Total effect = ``c' + a * b``.

For a binary outcome the product method operates on the log-odds scale and
is an approximation to the natural effects that additionally leans on a rare
outcome (or a probit/log link argument); the result fields are therefore
documented as log-odds-scale effects, not risk differences.

Causal assumptions (untestable from the data — stated here so no caller can
miss them):

1. **No exposure-mediator interaction** in the outcome model (on the modelled
   scale). The product ``a * b`` is only a valid decomposition when the
   effect of ``M`` on ``Y`` does not depend on ``X``.
2. **Sequential ignorability**: no unmeasured confounding of the
   exposure-outcome, mediator-outcome, or exposure-mediator relationships,
   and no mediator-outcome confounder that is itself affected by exposure.
3. Correct model specification and temporal ordering ``X -> M -> Y``.

Fail-closed enforcement:

* The caller must pass ``assume_no_interaction=True`` **and**
  ``assume_sequential_ignorability=True``. Anything else raises
  ``ValueError``: the kernel refuses to print causal numbers while its
  identifying assumptions are explicitly disavowed.
* An empirical interaction screen is run on top: the outcome model is
  augmented with an ``X * M`` term and the interaction coefficient is given
  a Wald t-test (OLS; for a binary outcome the screen uses a
  linear-probability OLS purely as a screen — the reported estimates still
  come from the sklearn logistic fit). ``p < 0.05`` raises ``ValueError``
  because the product decomposition is invalid under interaction.
* Illegal inputs (length mismatch, NaN/inf, zero-variance exposure or
  mediator, single-class binary outcome, singular designs, unconverged
  logistic fit) all raise ``ValueError``.

Uncertainty is a percentile bootstrap with a fixed seed (the interaction
screen is a full-sample gate and is not re-applied per replicate).

Claim ceiling: ``analysis_only``. Mediation estimates are not reportable
causal findings without a preregistered protocol, sensitivity analysis for
unmeasured confounding, and independent review.

References
----------
Baron RM, Kenny DA. "The moderator-mediator variable distinction in social
psychological research." *J Pers Soc Psychol* 1986;51(6):1173-1182.
VanderWeele TJ, Vansteelandt S. "Conceptual issues concerning mediation,
interventions and composition." *Stat Interface* 2009;2(4):457-468.
VanderWeele TJ, Vansteelandt S. "Odds ratios for mediation analysis for a
dichotomous outcome." *Am J Epidemiol* 2010;172(12):1339-1348.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class MediationResult:
    """Product-method natural direct/indirect effects with bootstrap CIs."""

    nde: float
    nie: float
    total_effect: float
    prop_mediated: float
    ci_nde: Tuple[float, float]
    ci_nie: Tuple[float, float]
    ci_total: Tuple[float, float]
    se_nde: float
    se_nie: float
    se_total: float
    coef_a: float
    coef_b: float
    coef_c_prime: float
    interaction_p_value: float
    outcome_type: str
    n: int
    n_covariates: int
    n_bootstrap: int
    n_successful: int
    ci_level: float
    random_state: int
    claim_ceiling: str = "analysis_only"
    method: str = "product_of_coefficients"

    def to_json(self) -> Dict[str, Any]:
        return {
            "nde": self.nde,
            "nie": self.nie,
            "total_effect": self.total_effect,
            "prop_mediated": self.prop_mediated,
            "ci_nde": [self.ci_nde[0], self.ci_nde[1]],
            "ci_nie": [self.ci_nie[0], self.ci_nie[1]],
            "ci_total": [self.ci_total[0], self.ci_total[1]],
            "se_nde": self.se_nde,
            "se_nie": self.se_nie,
            "se_total": self.se_total,
            "coef_a": self.coef_a,
            "coef_b": self.coef_b,
            "coef_c_prime": self.coef_c_prime,
            "interaction_p_value": self.interaction_p_value,
            "outcome_type": self.outcome_type,
            "n": self.n,
            "n_covariates": self.n_covariates,
            "n_bootstrap": self.n_bootstrap,
            "n_successful": self.n_successful,
            "ci_level": self.ci_level,
            "random_state": self.random_state,
            "assume_no_interaction": True,
            "assume_sequential_ignorability": True,
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _design(exposure: np.ndarray, mediator: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cols: list[np.ndarray] = [np.ones(exposure.shape[0]), exposure, mediator]
    if cov.shape[1] > 0:
        cols.append(cov)
    return np.column_stack(cols)


def _ols_fit(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    # LAPACK can set benign FP flags (e.g. divide-by-zero inside gelsd) that
    # numpy would otherwise misattribute to a later matmul; suppress that
    # numerics noise here while the rank gate below stays fail-closed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coef, residuals, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("regression design is rank-deficient; cannot identify effects")
    return np.asarray(coef, dtype=float)


def _interaction_p_value(
    exposure: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Wald p-value for an exposure-mediator interaction term (OLS screen)."""
    interaction = exposure * mediator
    cols: list[np.ndarray] = [np.ones(exposure.shape[0]), exposure, mediator, interaction]
    if cov.shape[1] > 0:
        cols.append(cov)
    design = np.column_stack(cols)
    n, p = design.shape
    if n - p < 1:
        raise ValueError("insufficient data to screen the exposure-mediator interaction")
    coef = _ols_fit(design, outcome)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resid = outcome - design @ coef
        dof = n - p
        sigma2 = float(resid @ resid) / float(dof)
        try:
            cov_beta = sigma2 * np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError as exc:
            raise ValueError("interaction-screen design is singular") from exc
        se_inter = float(np.sqrt(max(cov_beta[3, 3], 0.0)))
    coef_inter = float(coef[3])
    if se_inter <= 0.0:
        # Perfect fit: any nonzero interaction coefficient is real signal.
        return 0.0 if abs(coef_inter) > 1e-8 else 1.0
    t_stat = coef_inter / se_inter
    return float(2.0 * stats.t.sf(abs(t_stat), dof))


def _fit_logistic(
    features: np.ndarray, target: np.ndarray, random_state: int
) -> Tuple[float, float]:
    """Return ``(c_prime, b)`` from an unpenalised sklearn logistic fit."""
    if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
        raise ValueError("logistic fit requires aligned 2D features and 1D target")
    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            random_state=random_state,
            max_iter=5000,
        )
        # Unpenalised fits on resampled data can overflow internally on the
        # way to (non-)convergence; that numerics noise is suppressed here
        # while the convergence and finiteness gates below stay fail-closed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(features, target)
    except Exception as exc:
        raise ValueError(f"logistic outcome model failed to fit: {exc}") from exc
    n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
    if n_iter >= 5000:
        raise ValueError("logistic outcome model failed to converge; refusing to report")
    coef = np.ravel(model.coef_)
    if coef.shape[0] < 2 or not np.isfinite(coef).all():
        raise ValueError("logistic outcome model produced non-finite coefficients")
    return float(coef[0]), float(coef[1])


def mediate(
    exposure: Sequence[float],
    mediator: Sequence[float],
    outcome: Sequence[float],
    covariates: Optional[Sequence[Sequence[float]]] = None,
    outcome_type: str = "linear",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 0,
    assume_no_interaction: bool = False,
    assume_sequential_ignorability: bool = False,
) -> MediationResult:
    """Estimate natural direct/indirect effects by the product method.

    Parameters
    ----------
    exposure, mediator, outcome:
        One-dimensional aligned samples of ``X``, ``M``, ``Y``.
    covariates:
        Optional baseline covariates as an ``(n, k)`` array (or a single
        one-dimensional covariate). Must be unaffected by exposure.
    outcome_type:
        ``"linear"`` for a continuous outcome (OLS) or ``"binary"`` for a
        0/1 outcome (unpenalised sklearn logistic on the log-odds scale).
    assume_no_interaction, assume_sequential_ignorability:
        Must both be ``True``; the kernel raises ``ValueError`` otherwise
        (fail closed) because the product decomposition is unidentified
        without these assumptions.
    """
    if outcome_type not in ("linear", "binary"):
        raise ValueError("outcome_type must be 'linear' or 'binary'")
    if assume_no_interaction is not True:
        raise ValueError(
            "refusing to estimate: the product method requires an explicit "
            "assume_no_interaction=True declaration (no exposure-mediator interaction)"
        )
    if assume_sequential_ignorability is not True:
        raise ValueError(
            "refusing to estimate: mediation requires an explicit "
            "assume_sequential_ignorability=True declaration (no unmeasured confounding)"
        )

    x = np.asarray(list(exposure), dtype=float)
    m = np.asarray(list(mediator), dtype=float)
    y = np.asarray(list(outcome), dtype=float)
    if x.ndim != 1 or m.ndim != 1 or y.ndim != 1:
        raise ValueError("exposure, mediator and outcome must be one-dimensional")
    if not (x.shape[0] == m.shape[0] == y.shape[0]):
        raise ValueError("exposure, mediator and outcome must have equal length")
    n = x.shape[0]
    if n < 3:
        raise ValueError("mediation needs at least three observations")
    if not (np.isfinite(x).all() and np.isfinite(m).all() and np.isfinite(y).all()):
        raise ValueError("exposure, mediator and outcome must all be finite (no NaN/inf)")
    if float(np.std(x)) == 0.0:
        raise ValueError("exposure has no variation; effects are unidentified")
    if float(np.std(m)) == 0.0:
        raise ValueError("mediator has no variation; indirect effect is unidentified")

    if covariates is None:
        cov = np.zeros((n, 0), dtype=float)
    else:
        cov = np.asarray(list(covariates), dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        if cov.ndim != 2 or cov.shape[0] != n:
            raise ValueError("covariates must align with n rows of exposure/mediator/outcome")
        if not np.isfinite(cov).all():
            raise ValueError("covariates must all be finite (no NaN/inf)")
    n_cov = cov.shape[1]

    if outcome_type == "binary":
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("binary outcome must take values in {0, 1}")
        if not (y == 1.0).any() or not (y == 0.0).any():
            raise ValueError("binary outcome needs both classes observed")

    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError("n_bootstrap must be an integer")
    n_boot = int(n_bootstrap)
    if n_boot < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(ci_level, bool) or not isinstance(ci_level, (int, float, np.floating)):
        raise ValueError("ci_level must be a number in (0, 1)")
    level = float(ci_level)
    if not (0.0 < level < 1.0):
        raise ValueError("ci_level must be strictly between 0 and 1")
    seed = _as_int(random_state, "random_state")

    n_params_outcome = 3 + n_cov
    if n - n_params_outcome < 1:
        raise ValueError("insufficient data for the outcome model")

    # Full-sample interaction gate (fail closed) before any estimate is built.
    interaction_p = _interaction_p_value(x, m, y, cov)
    if interaction_p < 0.05:
        raise ValueError(
            "exposure-mediator interaction detected "
            f"(screen p={interaction_p:.4g} < 0.05); the product method is invalid here"
        )

    # Mediator model (OLS, both outcome types).
    med_design = np.column_stack(
        [np.ones(n), x] + ([cov] if n_cov > 0 else [])
    )
    coef_med = _ols_fit(med_design, m)
    coef_a = float(coef_med[1])

    def _fit_outcome(
        xx: np.ndarray, mm: np.ndarray, yy: np.ndarray, cc: np.ndarray
    ) -> Tuple[float, float]:
        if outcome_type == "linear":
            coef = _ols_fit(_design(xx, mm, cc), yy)
            return float(coef[1]), float(coef[2])
        feats = np.column_stack([xx, mm] + ([cc] if cc.shape[1] > 0 else []))
        return _fit_logistic(feats, yy, seed)

    coef_c, coef_b = _fit_outcome(x, m, y, cov)
    nde = coef_c
    nie = coef_a * coef_b
    total = nde + nie
    prop = float(nie / total) if total != 0.0 else float("nan")

    # Percentile bootstrap (fixed seed). Failed replicates are skipped, not
    # imputed; the usable count is recorded on the result.
    rng = np.random.default_rng(seed)
    boot_nde: list[float] = []
    boot_nie: list[float] = []
    boot_total: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xx, mm, yy, cc = x[idx], m[idx], y[idx], cov[idx]
        if float(np.std(xx)) == 0.0 or float(np.std(mm)) == 0.0:
            continue
        if outcome_type == "binary" and (not (yy == 1.0).any() or not (yy == 0.0).any()):
            continue
        try:
            a_b = float(_ols_fit(
                np.column_stack([np.ones(n), xx] + ([cc] if n_cov > 0 else [])),
                mm,
            )[1])
            c_b, b_b = _fit_outcome(xx, mm, yy, cc)
        except ValueError:
            continue
        nde_b = c_b
        nie_b = a_b * b_b
        boot_nde.append(nde_b)
        boot_nie.append(nie_b)
        boot_total.append(nde_b + nie_b)
    n_ok = len(boot_nde)
    if n_ok == 0:
        raise ValueError("bootstrap produced no usable replicate")

    alpha = 1.0 - level

    def _ci(draws: list[float]) -> Tuple[Tuple[float, float], float]:
        arr = np.asarray(draws, dtype=float)
        low = float(np.quantile(arr, alpha / 2.0))
        high = float(np.quantile(arr, 1.0 - alpha / 2.0))
        se = float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0
        return (low, high), se

    ci_nde, se_nde = _ci(boot_nde)
    ci_nie, se_nie = _ci(boot_nie)
    ci_total, se_total = _ci(boot_total)

    outcome_literal: Literal["linear", "binary"] = (
        "linear" if outcome_type == "linear" else "binary"
    )
    return MediationResult(
        nde=nde,
        nie=nie,
        total_effect=total,
        prop_mediated=prop,
        ci_nde=ci_nde,
        ci_nie=ci_nie,
        ci_total=ci_total,
        se_nde=se_nde,
        se_nie=se_nie,
        se_total=se_total,
        coef_a=coef_a,
        coef_b=coef_b,
        coef_c_prime=coef_c,
        interaction_p_value=interaction_p,
        outcome_type=outcome_literal,
        n=n,
        n_covariates=n_cov,
        n_bootstrap=n_boot,
        n_successful=n_ok,
        ci_level=level,
        random_state=seed,
    )


__all__ = [
    "MediationResult",
    "mediate",
]
