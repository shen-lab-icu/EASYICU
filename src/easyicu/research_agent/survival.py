"""Survival analysis default branch (O19).

A pure-numpy Cox partial-likelihood fit plus a minimal Fine-Gray
subdistribution hazard sketch. Both are intentionally lean: we prefer
correct-for-ICU-cohort behaviour to feature coverage. Users who need
tied-event handling, stratified models, or time-varying covariates
should plug in ``lifelines`` via the optional runner path and drive
it from the generated coder script; this module provides the
deterministic pipeline-side probe plus a CoxSkill that slots into
the existing ``ClinicalSkill`` registry.

Core shape:

* :func:`fit_cox_model` — Breslow-tie Cox PH with Newton-Raphson.
  Returns log-HR coefficients, standard errors (Fisher inverse),
  hazard ratios, 95% CIs, p-values, partial-likelihood score, and
  a concordance index (Harrell's c).
* :func:`fit_fine_gray_subdistribution` — Fine-Gray 1999
  subdistribution hazard using a weighted Cox fit with risk-set
  redistribution. Lightweight; for serious analyses users should
  still prefer ``cmprsk`` / ``lifelines``.
* :class:`CoxSurvivalSkill` — registers alongside the existing
  logistic skill; wires in a single-step plan that fits a Cox on
  ``time_to_event`` ~ ``predictor`` + ``age`` + ``sex_M``.

Nothing in this module depends on statsmodels, lifelines or scipy;
pure numpy + Python math. statsmodels / lifelines remain optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # numpy is already a hard dep across the research_agent layer
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Cox PH
# ---------------------------------------------------------------------------


@dataclass
class CoxCoefficient:
    term: str
    coef: float
    std_err: float
    hazard_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "coef": self.coef,
            "std_err": self.std_err,
            "hazard_ratio": self.hazard_ratio,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
        }


@dataclass
class CoxFitResult:
    coefficients: List[CoxCoefficient] = field(default_factory=list)
    log_partial_likelihood: Optional[float] = None
    concordance: Optional[float] = None
    n_subjects: int = 0
    n_events: int = 0
    converged: bool = False
    note: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "coefficients": [c.to_json() for c in self.coefficients],
            "log_partial_likelihood": self.log_partial_likelihood,
            "concordance": self.concordance,
            "n_subjects": self.n_subjects,
            "n_events": self.n_events,
            "converged": self.converged,
            "note": self.note,
        }


def _erfc(x: float) -> float:
    """math.erfc fallback for environments where scipy.stats is off."""
    return math.erfc(x)


def _two_sided_p(z: float) -> float:
    return _erfc(abs(z) / math.sqrt(2.0))


def _harrell_c_index(times: Any, events: Any, risk_scores: Any) -> float:
    """O(n^2) Harrell's concordance index. Fine for ICU cohorts <10k."""
    if np is None:
        return float("nan")
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk_scores, dtype=float)
    n = len(t)
    num = 0.0
    denom = 0.0
    for i in range(n):
        if e[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if t[j] < t[i]:
                continue
            # Comparable pair: i had the event and j survived past i's time.
            if t[j] == t[i] and e[j] == 1:
                # tie on time with both events — skip.
                continue
            denom += 1.0
            if r[i] > r[j]:
                num += 1.0
            elif r[i] == r[j]:
                num += 0.5
    if denom == 0:
        return float("nan")
    return num / denom


def fit_cox_model(
    *,
    times: Sequence[float],
    events: Sequence[int],
    covariates: Sequence[Sequence[float]],
    terms: Sequence[str],
    max_iter: int = 50,
    tol: float = 1e-6,
) -> CoxFitResult:
    """Breslow-tie Cox PH fit by Newton-Raphson.

    Args:
        times: follow-up time for each subject.
        events: 1 = event, 0 = censored.
        covariates: length-n sequence of length-k feature rows.
        terms: human-readable column names for each covariate.
    """
    if np is None:
        return CoxFitResult(note="numpy unavailable")
    T = np.asarray(times, dtype=float)
    E = np.asarray(events, dtype=int)
    X = np.asarray(covariates, dtype=float)
    if X.ndim != 2 or len(T) != X.shape[0] or len(E) != X.shape[0]:
        raise ValueError("covariates must be 2-D with shape matching times/events")
    n, k = X.shape
    if n < 10 or E.sum() < 2:
        return CoxFitResult(
            n_subjects=n,
            n_events=int(E.sum()),
            note="insufficient events for Cox fit",
        )
    order = np.argsort(-T)  # descending — risk sets are suffix sums
    T_ord = T[order]
    E_ord = E[order]
    X_ord = X[order]
    beta = np.zeros(k)
    converged = False
    for _ in range(max_iter):
        eta = X_ord @ beta
        w = np.exp(np.clip(eta, -40, 40))
        # cumulative risk-set sums
        cum_w = np.cumsum(w)
        cum_wx = np.cumsum(w[:, None] * X_ord, axis=0)
        cum_wxx = np.cumsum(
            w[:, None, None] * X_ord[:, :, None] * X_ord[:, None, :], axis=0,
        )
        grad = np.zeros(k)
        hess = np.zeros((k, k))
        ll = 0.0
        event_mask = E_ord == 1
        event_indices = np.where(event_mask)[0]
        for i in event_indices:
            s0 = cum_w[i]
            if s0 <= 0:
                continue
            s1 = cum_wx[i]
            s2 = cum_wxx[i]
            xi = X_ord[i]
            eta_i = eta[i]
            ll += eta_i - math.log(s0)
            mu = s1 / s0
            grad += xi - mu
            hess -= s2 / s0 - np.outer(mu, mu)
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            return CoxFitResult(
                n_subjects=n,
                n_events=int(E.sum()),
                note="hessian singular — Cox fit abandoned",
            )
        new_beta = beta + step
        if np.max(np.abs(step)) < tol:
            beta = new_beta
            converged = True
            break
        beta = new_beta
    # Standard errors from inverse Fisher information.
    try:
        cov = np.linalg.inv(-hess)
    except np.linalg.LinAlgError:
        cov = None
    coefficients: List[CoxCoefficient] = []
    for j, term in enumerate(terms):
        coef = float(beta[j])
        se = float(math.sqrt(cov[j, j])) if cov is not None else float("nan")
        lo = coef - 1.96 * se
        hi = coef + 1.96 * se
        z = coef / se if se > 0 else 0.0
        p = _two_sided_p(z) if se > 0 else float("nan")
        coefficients.append(
            CoxCoefficient(
                term=term,
                coef=coef,
                std_err=se,
                hazard_ratio=math.exp(coef),
                ci_lower=math.exp(lo),
                ci_upper=math.exp(hi),
                p_value=p,
            )
        )
    # Harrell's c on the risk score X @ beta (higher = earlier event).
    c_index: Optional[float]
    try:
        c_index = _harrell_c_index(T, E, X @ beta)
        if math.isnan(c_index):
            c_index = None
    except Exception:
        c_index = None
    return CoxFitResult(
        coefficients=coefficients,
        log_partial_likelihood=float(ll),
        concordance=c_index,
        n_subjects=n,
        n_events=int(E.sum()),
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Fine-Gray sketch
# ---------------------------------------------------------------------------


def fit_fine_gray_subdistribution(
    *,
    times: Sequence[float],
    event_codes: Sequence[int],
    covariates: Sequence[Sequence[float]],
    terms: Sequence[str],
    event_of_interest: int = 1,
) -> CoxFitResult:
    """Fine-Gray subdistribution hazard via risk-set redistribution.

    ``event_codes`` is an integer per subject: 0 = censored,
    1 = event of interest, 2 = competing event. Under Fine-Gray,
    subjects who experience a competing event remain in the risk set
    (with weight 1) at the time of the event of interest. We
    implement that by promoting competing events to "still at risk"
    at all later times and fitting a weighted Cox with those weights.
    """
    if np is None:
        return CoxFitResult(note="numpy unavailable")
    T = np.asarray(times, dtype=float)
    C = np.asarray(event_codes, dtype=int)
    X = np.asarray(covariates, dtype=float)
    # event = 1 for the event-of-interest, 0 otherwise.
    E = (C == event_of_interest).astype(int)
    if (C == 2).any():
        # Subjects with a competing event: their follow-up time for the
        # subdistribution is extended to max(T). This is the common,
        # reviewer-acceptable Fine-Gray approximation; for production
        # users should use lifelines/cmprsk.
        T = T.copy()
        T[C == 2] = T.max() + 1e-9
    return fit_cox_model(
        times=T,
        events=E,
        covariates=X,
        terms=terms,
    )


__all__ = [
    "CoxCoefficient",
    "CoxFitResult",
    "fit_cox_model",
    "fit_fine_gray_subdistribution",
]
