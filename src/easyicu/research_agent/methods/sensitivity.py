"""Negative-control outcome and E-value helpers (O23).

Two small additions that strengthen the causal story without
importing DoWhy:

* :func:`compute_e_value` — VanderWeele & Ding (2017). Given a point
  estimate and its confidence interval, returns the E-value of the
  estimate and of the lower bound (or upper bound for protective
  effects). A larger E-value means a stronger effect is required of
  an unmeasured confounder to explain the association away.
* :func:`run_negative_control_check` — when the user supplies a
  biologically implausible outcome column (e.g. ``eye_color_change``
  for a SOFA-2 × ICU-mortality analysis), we refit the same primary
  model on that outcome and flag a *positive* association as a
  warning — it would indicate residual confounding.

Both functions are pure numpy; no SDK creep. Intended wiring: a
CausalSkill registers ``identification_strategy`` metadata with
``negative_control_outcome`` and ``e_value`` ids, the pipeline runs
this module, and :mod:`easyicu.research_agent.review.causal_audit` then upgrades the label from
``causal_overclaimed`` to ``causal_explicit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# E-value
# ---------------------------------------------------------------------------


def _rr_from_or(or_value: float, baseline_prevalence: float) -> float:
    """Convert an OR to an approximate RR using Zhang-Yu (1998).

    Needed because the E-value formula is defined for risk ratios. We
    clamp the baseline prevalence to (0, 1) open interval.
    """
    p = max(1e-6, min(1.0 - 1e-6, float(baseline_prevalence)))
    return or_value / (1 - p + p * or_value)


def _e_value_for_rr(rr: float) -> float:
    """Canonical VanderWeele–Ding E-value for a risk ratio."""
    r = float(rr)
    if r < 1.0:
        # Protective: invert and compute.
        r = 1.0 / r
    return r + math.sqrt(r * (r - 1.0))


class BaselinePrevalenceRequiredError(ValueError):
    """An odds ratio reached the E-value with no observed event rate.

    A subclass of ValueError so existing callers that guard the E-value with
    ``except ValueError`` keep working, while a caller that wants to report
    *this* cause specifically can catch the narrower type.
    """


@dataclass
class EValueResult:
    estimate: float
    estimate_type: str  # "or" | "rr" | "hr"
    e_value: float
    e_value_lower_bound: Optional[float] = None
    note: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "estimate_type": self.estimate_type,
            "e_value": self.e_value,
            "e_value_lower_bound": self.e_value_lower_bound,
            "note": self.note,
        }


def compute_e_value(
    *,
    estimate: float,
    ci: Optional[Tuple[float, float]] = None,
    estimate_type: str = "or",
    baseline_prevalence: Optional[float] = None,
) -> EValueResult:
    """Compute an E-value for a point estimate and optional CI.

    Args:
        estimate: OR / RR / HR point estimate.
        ci: two-sided 95% CI as ``(lower, upper)``. Optional.
        estimate_type: ``"or"``, ``"rr"``, or ``"hr"``.
        baseline_prevalence: the cohort's OBSERVED event rate, in (0, 1).
            REQUIRED when ``estimate_type == "or"`` for the Zhang-Yu
            conversion, and refused when absent or out of range.

    Returns an :class:`EValueResult`. For an OR / RR / HR above 1 we
    compute the E-value for the *lower* CI bound; for estimates below
    1 we use the *upper* bound (the "hardest to explain away" edge).

    Raises:
        BaselinePrevalenceRequiredError: for an OR with no usable observed
            event rate. This used to default to 0.1 with an explanatory note,
            which is a GUESSED scientific parameter that moves a reported
            number: for OR=2.0, an assumed 0.1 gives E=3.04 while a real
            rate of 0.214 gives E=2.68 -- overstating robustness. A note
            does not make an invented input
            defensible, and the caller cannot detect the substitution by
            looking at the value. Refusing here makes the guess impossible
            for every caller rather than for the one we happened to audit.
    """
    kind = estimate_type.lower()
    if kind not in {"or", "rr", "hr"}:
        raise ValueError(f"Unsupported estimate_type {estimate_type!r}")
    rr_point = float(estimate)
    note: Optional[str] = None
    if kind == "or":
        if baseline_prevalence is None:
            raise BaselinePrevalenceRequiredError(
                "An odds ratio cannot be converted to a risk ratio without the "
                "cohort's observed event rate. Supply baseline_prevalence from "
                "the run's own outcome-rate product, or do not report an "
                "E-value for this estimate."
            )
        if not 0.0 < float(baseline_prevalence) < 1.0:
            raise BaselinePrevalenceRequiredError(
                f"baseline_prevalence={baseline_prevalence!r} is not an observed "
                "event rate in (0, 1). A rate of 0 or 1 has no risk ratio, and a "
                "value outside the interval is not a rate at all."
            )
        note = (
            "OR converted to RR at the observed event rate "
            f"{float(baseline_prevalence):.4f}"
        )
        rr_point = _rr_from_or(estimate, baseline_prevalence)
    # HR ≈ RR for rare events; keep as-is.
    ev = _e_value_for_rr(rr_point)
    ev_bound: Optional[float] = None
    if ci is not None:
        lo, hi = float(ci[0]), float(ci[1])
        if rr_point >= 1:
            bound = lo
        else:
            bound = hi
        if kind == "or":
            bound_rr = _rr_from_or(bound, baseline_prevalence)
        else:
            bound_rr = bound
        # An interval that already contains the null has a CI E-value of
        # exactly 1: the data are compatible with no effect, so a confounder of
        # no strength is "needed" to explain the finding away (VanderWeele &
        # Ding 2017).
        #
        # This used to fall through to _e_value_for_rr, whose protective branch
        # inverts anything below 1. With a point estimate above the null and a
        # lower bound of 0.90 that reflected to 1/0.90 and reported 1.46 -- an
        # interval spanning the null presented as needing a confounder half
        # again as strong as the null to explain away. The comment here
        # asserted the fall-through was "~1", which holds only for a bound very
        # close to 1 and was never the general case.
        crosses_null = bound_rr <= 1.0 if rr_point >= 1 else bound_rr >= 1.0
        ev_bound = 1.0 if crosses_null else _e_value_for_rr(bound_rr)
    return EValueResult(
        estimate=estimate,
        estimate_type=kind,
        e_value=ev,
        e_value_lower_bound=ev_bound,
        note=note,
    )


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


@dataclass
class NegativeControlResult:
    outcome_column: str
    predictor_column: str
    n_used: int
    estimate: Optional[float]
    ci: Optional[Tuple[float, float]]
    p_value: Optional[float]
    significant_at_005: Optional[bool]
    note: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "outcome_column": self.outcome_column,
            "predictor_column": self.predictor_column,
            "n_used": self.n_used,
            "estimate": self.estimate,
            "ci": list(self.ci) if self.ci else None,
            "p_value": self.p_value,
            "significant_at_005": self.significant_at_005,
            "note": self.note,
        }


def _fit_simple_logistic(
    x: Sequence[float], y: Sequence[int]
) -> Optional[Tuple[float, float, float]]:
    """Minimal IRLS logistic for (intercept + one predictor).

    Returns ``(coef, std_err, p_value)`` or None if convergence fails.
    Kept deliberately tiny; the real logistic fit lives in the
    generated coder script. This version is only used as a negative-
    control probe, so speed and zero-dependency matter more than
    feature coverage.
    """
    try:
        import numpy as np
    except Exception:
        return None

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    n = len(x_arr)
    if n < 30 or len(np.unique(y_arr)) < 2:
        return None
    X = np.column_stack([np.ones(n), x_arr])
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
        W = p * (1 - p)
        # score and information
        z = eta + (y_arr - p) / np.maximum(W, 1e-9)
        XtWX = X.T @ (W[:, None] * X)
        try:
            inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            return None
        new_beta = inv @ (X.T @ (W * z))
        if np.max(np.abs(new_beta - beta)) < 1e-6:
            beta = new_beta
            break
        beta = new_beta
    coef = float(beta[1])
    try:
        inv = np.linalg.inv(X.T @ (W[:, None] * X))
    except Exception:
        return None
    se = float(math.sqrt(max(0.0, inv[1, 1])))
    if se == 0.0:
        return None
    z_stat = coef / se
    # Two-sided p-value via a tiny erf approximation.
    p_val = math.erfc(abs(z_stat) / math.sqrt(2.0))
    return coef, se, p_val


def run_negative_control_check(
    *,
    cohort_df: Any,
    predictor_column: str,
    negative_control_column: str,
) -> NegativeControlResult:
    """Fit a tiny logistic for ``negative_control_column ~ predictor``.

    A significant association here warns that residual confounding is
    present: an unrelated outcome should not track your predictor. A
    non-significant result is the happy path.
    """
    try:
        df = cohort_df.dropna(subset=[predictor_column, negative_control_column])
    except Exception as exc:
        return NegativeControlResult(
            outcome_column=negative_control_column,
            predictor_column=predictor_column,
            n_used=0,
            estimate=None,
            ci=None,
            p_value=None,
            significant_at_005=None,
            note=f"dataframe_error: {type(exc).__name__}: {exc}",
        )
    x = df[predictor_column].astype(float).tolist()
    y = df[negative_control_column].astype(int).tolist()
    fit = _fit_simple_logistic(x, y)
    if fit is None:
        return NegativeControlResult(
            outcome_column=negative_control_column,
            predictor_column=predictor_column,
            n_used=len(y),
            estimate=None,
            ci=None,
            p_value=None,
            significant_at_005=None,
            note="logistic fit failed or insufficient variance",
        )
    coef, se, p_val = fit
    lo = coef - 1.96 * se
    hi = coef + 1.96 * se
    return NegativeControlResult(
        outcome_column=negative_control_column,
        predictor_column=predictor_column,
        n_used=len(y),
        estimate=math.exp(coef),
        ci=(math.exp(lo), math.exp(hi)),
        p_value=p_val,
        significant_at_005=bool(p_val < 0.05),
    )


__all__ = [
    "BaselinePrevalenceRequiredError",
    "EValueResult",
    "NegativeControlResult",
    "compute_e_value",
    "run_negative_control_check",
]
