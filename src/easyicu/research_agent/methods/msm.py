"""Longitudinal MSM stabilised IPTW weights: sequential mechanical weighting.

For follow-up visits ``t = 0, ..., T-1`` with binary visit treatments ``A[t]``,
the stabilised inverse-probability weight for subject ``i`` is the cumulative
product of per-visit stabilised contributions::

    sw_i = prod_t  f(A[t]_i | num-history) / f(A[t]_i | den-history)

where ``f`` is the fitted visit propensity (Bernoulli). Concretely, with
``ps_den[t]_i = P(A[t]=1 | den_features[t])`` and
``ps_num[t]_i = P(A[t]=1 | num_features[t])`` from per-visit logistic fits::

    contrib[t]_i = ps_num/ps_den            if A[t]_i == 1
                   (1-ps_num)/(1-ps_den)    if A[t]_i == 0

This module owns the *mechanics only*: fitting, seed derivation, cumulative
multiplication, optional quantile truncation, per-visit diagnostics, and
fail-closed positivity. The *confounding semantics* — which covariates belong
in the denominator (treatment/confounder history) versus the numerator
(baseline stabiliser) — belong to the caller. The kernel never inspects column
meaning and never imputes a default history.

Mechanical choices in this skeleton
-----------------------------------
* Per-visit propensity: ``sklearn.linear_model.LogisticRegression`` (lbfgs,
  ``max_iter=5000``) with a fixed seed derived per visit as
  ``seed_used = random_state + visit_index`` (hardcoded rule, documented here
  and in :func:`stabilized_iptw`). Both the numerator and the denominator fit
  at visit ``t`` use that same derived seed, so reruns are bit-for-bit
  identical given the same inputs and ``random_state``.
* A ``None`` numerator entry means an intercept-only (marginal) numerator:
  ``ps_num[t]`` is the empirical ``mean(A[t])`` with no model fit.
* Optional quantile truncation (``trunc_quantiles=(q_low, q_high)``) clips the
  *final* cumulative weights at their empirical quantiles; bounds and the
  truncated flag are reported, never silent.
* Positivity is fail closed, mirroring
  :mod:`easyicu.research_agent.methods.doubly_robust`: denominator propensities
  at or beyond the hard bounds ``[1e-6, 1 - 1e-6]`` raise :class:`MSMError`,
  as does any denominator propensity outside the declared trim window
  (default ``[0.025, 0.975]``). The would-be trim proportion is reported on
  the error instead of trimming silently.
* Uncertainty is NOT estimated here: no bootstrap, no influence-function SE,
  no MSM outcome-model fit. The output is a weight vector plus diagnostics for
  a downstream (human-reviewed) MSM fit.

Identification assumptions (untestable from the data — stated, not verified)
----------------------------------------------------------------------------
1. **Sequential conditional exchangeability**: at each visit, treatment is
   independent of the potential outcomes given the caller's denominator
   history. The kernel cannot check this; a wrong denominator is a wrong
   answer with well-formed diagnostics.
2. **Consistency** and **positivity** (the latter is screened mechanically,
   see above; the screen can only see the fitted support, not the truth).
3. Correct specification of each per-visit propensity model.

What is NOT in this module (scope boundary)
--------------------------------------------
* The longitudinal g-formula (Monte-Carlo integration over time-varying
  confounders) — a different engine, not a flag on this one.
* Multi-stage dynamic-treatment-regime optimisation — no rule is learned here.
* Pooled fitting across visits: each visit is fit separately on its own
  caller-provided features; there is no pooled logistic model over
  person-visits and no handling of time-varying confounding beyond what the
  caller encoded in ``den_features[t]``.
* An MSM outcome fit or any causal conclusion.

Evidence ceiling: ``analysis_only``. Stabilised weights near 1 with a healthy
ESS mean the *mechanics* behaved; they cannot substitute for human
confirmation of the assumptions above. Nothing here is reportable as a causal
finding.

References
----------
Robins JM, Hernan MA, Brumback B. "Marginal structural models and causal
inference in epidemiology." *Epidemiology* 2000;11:550-560.
Hernan MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC, 2020,
Ch. 12, 21 (IP weighting, MSMs for time-varying treatments).
Cole SR, Hernan MA. "Constructing inverse probability weights for marginal
structural models." *Am J Epidemiol* 2008;168:656-664.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

#: Evidence ceiling for every product of this module.
EVIDENCE_CEILING = "analysis_only"

#: Hard denominator-propensity bounds — values at or beyond these make the
#: per-visit contributions ``ps_num/ps_den`` numerically degenerate.
#: Fail closed. Mirrors the doubly-robust kernel convention.
_PS_HARD_EPS = 1e-6

#: Fixed optimiser budget so fits are deterministic and complete quickly.
_MAX_ITER = 5000

#: Known limitations, repeated verbatim on every result so no caller can miss
#: the scope boundary.
LIMITATIONS: tuple[str, ...] = (
    "Caller-owned confounding semantics: the kernel never validates that "
    "den_features[t]/num_features[t] encode the correct histories.",
    "No longitudinal g-formula: Monte-Carlo integration over time-varying "
    "confounders is not implemented here.",
    "No multi-stage DTR optimisation: no treatment rule is learned or "
    "compared in this module.",
    "No pooled fitting: each visit is fit separately; there is no pooled "
    "person-visit model and no MSM outcome fit.",
    "No uncertainty quantification: weights and diagnostics only, no "
    "bootstrap or influence-function SE.",
)


class MSMError(ValueError):
    """The MSM weighting kernel refuses to proceed (fail closed)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "msm_invalid",
        trim_proportion: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.trim_proportion = float(trim_proportion)


@dataclass(frozen=True)
class MSMVisitDiagnostics:
    """Per-visit mechanical diagnostics (cumulative weights after visit ``t``)."""

    visit_index: int
    seed_used: int
    n: int
    ps_den_min: float
    ps_den_max: float
    ps_den_mean: float
    ps_num_min: float
    ps_num_max: float
    ps_num_mean: float
    incr_min: float
    incr_max: float
    cumul_mean: float
    cumul_min: float
    cumul_max: float
    cumul_ess: float

    def to_json(self) -> dict[str, Any]:
        return {
            "visit_index": self.visit_index,
            "seed_used": self.seed_used,
            "n": self.n,
            "ps_den_min": self.ps_den_min,
            "ps_den_max": self.ps_den_max,
            "ps_den_mean": self.ps_den_mean,
            "ps_num_min": self.ps_num_min,
            "ps_num_max": self.ps_num_max,
            "ps_num_mean": self.ps_num_mean,
            "incr_min": self.incr_min,
            "incr_max": self.incr_max,
            "cumul_mean": self.cumul_mean,
            "cumul_min": self.cumul_min,
            "cumul_max": self.cumul_max,
            "cumul_ess": self.cumul_ess,
        }


@dataclass(frozen=True)
class MSMWeightsResult:
    """Sequential stabilised IPTW weights with per-visit diagnostics."""

    weights: np.ndarray
    n: int
    n_visits: int
    random_state: int
    seeds_used: tuple[int, ...]
    ps_trim_low: float
    ps_trim_high: float
    trim_proportion: float
    per_visit: tuple[MSMVisitDiagnostics, ...]
    truncated: bool
    trunc_quantiles: tuple[float, float] | None
    trunc_low: float | None
    trunc_high: float | None
    weight_mean: float
    weight_min: float
    weight_max: float
    weight_ess: float
    limitations: tuple[str, ...] = LIMITATIONS
    evidence_ceiling: str = EVIDENCE_CEILING
    method: str = "sequential_stabilized_iptw"

    def to_json(self) -> dict[str, Any]:
        return {
            "weights": [float(v) for v in np.ravel(self.weights)],
            "n": self.n,
            "n_visits": self.n_visits,
            "random_state": self.random_state,
            "seeds_used": list(self.seeds_used),
            "ps_trim_low": self.ps_trim_low,
            "ps_trim_high": self.ps_trim_high,
            "trim_proportion": self.trim_proportion,
            "per_visit": [visit.to_json() for visit in self.per_visit],
            "truncated": self.truncated,
            "trunc_quantiles": (
                [self.trunc_quantiles[0], self.trunc_quantiles[1]]
                if self.trunc_quantiles is not None
                else None
            ),
            "trunc_low": self.trunc_low,
            "trunc_high": self.trunc_high,
            "weight_mean": self.weight_mean,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "weight_ess": self.weight_ess,
            "limitations": list(self.limitations),
            "evidence_ceiling": self.evidence_ceiling,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Input validation (fail closed, no silent coercion).
# ---------------------------------------------------------------------------


def _as_binary_vector(values: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise MSMError(
            f"{label} must be one-dimensional, got shape {arr.shape}",
            code="msm_shape_mismatch",
        )
    flat = arr.ravel()
    if flat.shape[0] == 0:
        raise MSMError(f"{label} is empty", code="msm_empty_input")
    try:
        numeric = flat.astype(float)
    except (TypeError, ValueError) as exc:
        raise MSMError(
            f"{label} must be binary 0/1", code="msm_not_binary"
        ) from exc
    if not bool(np.isfinite(numeric).all()):
        raise MSMError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="msm_not_finite",
        )
    unique = set(np.unique(numeric).tolist())
    if unique - {0.0, 1.0}:
        raise MSMError(
            f"{label} must be binary 0/1, got values {sorted(unique)[:5]}",
            code="msm_not_binary",
        )
    return numeric.astype(float)


def _as_feature_matrix(values: Any, *, label: str, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 1:
        raise MSMError(
            f"{label} must have shape ({n}, k>=1), got shape {arr.shape}",
            code="msm_shape_mismatch",
        )
    if not bool(np.isfinite(arr).all()):
        raise MSMError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="msm_covariates_not_finite",
        )
    return arr


def _check_trim_bounds(ps_trim: Any) -> tuple[float, float]:
    try:
        low, high = (float(ps_trim[0]), float(ps_trim[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise MSMError(
            "ps_trim must be a (low, high) pair",
            code="msm_trim_bounds_invalid",
        ) from exc
    if not (0.0 < low < high < 1.0):
        raise MSMError(
            f"ps_trim must satisfy 0 < low < high < 1, got ({low}, {high})",
            code="msm_trim_bounds_invalid",
        )
    return low, high


def _check_trunc_quantiles(trunc_quantiles: Any) -> tuple[float, float] | None:
    if trunc_quantiles is None:
        return None
    try:
        q_low, q_high = (float(trunc_quantiles[0]), float(trunc_quantiles[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise MSMError(
            "trunc_quantiles must be a (q_low, q_high) pair or None",
            code="msm_trunc_bounds_invalid",
        ) from exc
    if not (0.0 <= q_low < q_high <= 1.0):
        raise MSMError(
            f"trunc_quantiles must satisfy 0 <= q_low < q_high <= 1, "
            f"got ({q_low}, {q_high})",
            code="msm_trunc_bounds_invalid",
        )
    return q_low, q_high


def _ess(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    denom = float(np.sum(weights**2))
    if denom <= 0.0 or not np.isfinite(total) or not np.isfinite(denom):
        raise MSMError(
            "effective sample size is undefined (non-finite weights)",
            code="msm_weights_not_finite",
        )
    return (total**2) / denom


# ---------------------------------------------------------------------------
# Model fits (fixed seeds, deterministic).
# ---------------------------------------------------------------------------


def _fit_visit_propensity(
    features: np.ndarray, treated: np.ndarray, *, seed: int, label: str
) -> np.ndarray:
    try:
        model = LogisticRegression(max_iter=_MAX_ITER, random_state=int(seed))
        model.fit(features, treated)
    except Exception as exc:
        raise MSMError(
            f"{label}: logistic propensity fit failed: {exc}",
            code="msm_propensity_fit_failed",
        ) from exc
    n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
    if n_iter >= _MAX_ITER:
        raise MSMError(
            f"{label}: logistic propensity fit did not converge; refusing to weight",
            code="msm_propensity_not_converged",
        )
    scores = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if not bool(np.isfinite(scores).all()):
        raise MSMError(
            f"{label}: propensity fit produced non-finite scores",
            code="msm_propensity_not_finite",
        )
    return scores


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def stabilized_iptw(
    treatments: Sequence[Any],
    den_features: Sequence[Any],
    num_features: Sequence[Any | None],
    *,
    random_state: int = 0,
    ps_trim: tuple[float, float] = (0.025, 0.975),
    trunc_quantiles: tuple[float, float] | None = None,
    require_sequential_exchangeability: bool = False,
    require_correct_history_encoding: bool = False,
) -> MSMWeightsResult:
    """Sequential stabilised IPTW weights over follow-up visits (fail closed).

    Parameters
    ----------
    treatments:
        Length-``T`` sequence of binary (0/1) visit-treatment vectors, one
        ``(n,)`` vector per visit.
    den_features:
        Length-``T`` sequence of denominator feature matrices
        (``(n, k)`` each; 1-D input is treated as a single column). The
        caller encodes the treatment/confounder history here; the kernel
        only fits ``P(A[t]=1 | den_features[t])``.
    num_features:
        Length-``T`` sequence of numerator feature matrices (same shape
        contract as the denominator), or ``None`` at a visit for an
        intercept-only (marginal) numerator ``mean(A[t])``.
    random_state:
        Base seed. The per-visit seed is hardcoded as
        ``seed_used = random_state + visit_index``; both the numerator and
        the denominator logistic fit at visit ``t`` use that derived seed.
    ps_trim:
        Acceptable denominator-propensity window. Any fitted denominator
        score outside it raises :class:`MSMError` instead of being trimmed.
    trunc_quantiles:
        Optional ``(q_low, q_high)`` empirical quantiles at which the
        *final* cumulative weights are clipped. ``None`` (default) means no
        truncation.

    Returns :class:`MSMWeightsResult` with the cumulative weight vector, the
    per-visit diagnostics (cumulative mean/min/max/ESS after each visit), and
    the truncation receipt. ``trim_proportion`` is 0.0 on success (any
    positive value raises).

    Assumption gate (mirrors mediation/g-formula): sequential
    exchangeability given the encoded histories, and correct encoding of
    those histories, are reviewer-owned and cannot be verified from the
    arrays. Both flags must be explicitly ``True`` or the call is refused
    before any model is fitted — a well-formed but mis-encoded history
    would otherwise produce a well-formed wrong answer.
    """

    if require_sequential_exchangeability is not True:
        raise MSMError(
            "sequential exchangeability must be explicitly declared "
            "(require_sequential_exchangeability=True)",
            code="msm_assumption_undeclared",
        )
    if require_correct_history_encoding is not True:
        raise MSMError(
            "correct history encoding must be explicitly declared "
            "(require_correct_history_encoding=True)",
            code="msm_assumption_undeclared",
        )
    if not isinstance(random_state, (int, np.integer)) or isinstance(
        random_state, bool
    ):
        raise MSMError("random_state must be an integer", code="msm_seed_invalid")
    seed = int(random_state)
    low, high = _check_trim_bounds(ps_trim)
    trunc = _check_trunc_quantiles(trunc_quantiles)

    try:
        n_visits = len(treatments)  # type: ignore[arg-type]
    except TypeError as exc:
        raise MSMError(
            "treatments must be a sequence of per-visit vectors",
            code="msm_shape_mismatch",
        ) from exc
    if n_visits < 1:
        raise MSMError(
            "at least one visit is required", code="msm_empty_input"
        )
    if len(den_features) != n_visits or len(num_features) != n_visits:  # type: ignore[arg-type]
        raise MSMError(
            "treatments, den_features and num_features must list the same "
            f"number of visits (got {n_visits}, {len(den_features)}, "  # type: ignore[arg-type]
            f"{len(num_features)})",  # type: ignore[arg-type]
            code="msm_shape_mismatch",
        )

    treated_seq = [
        _as_binary_vector(values, label=f"treatments[{t}]")
        for t, values in enumerate(treatments)
    ]
    n = int(treated_seq[0].shape[0])
    for t, vec in enumerate(treated_seq):
        if int(vec.shape[0]) != n:
            raise MSMError(
                f"treatments[{t}] has {vec.shape[0]} rows, expected {n}",
                code="msm_shape_mismatch",
            )

    den_seq = [
        _as_feature_matrix(values, label=f"den_features[{t}]", n=n)
        for t, values in enumerate(den_features)
    ]
    num_seq: list[np.ndarray | None] = []
    for t, values in enumerate(num_features):
        if values is None:
            num_seq.append(None)
            continue
        num_seq.append(
            _as_feature_matrix(values, label=f"num_features[{t}]", n=n)
        )

    cumul = np.ones(n, dtype=float)
    per_visit: list[MSMVisitDiagnostics] = []
    seeds_used: list[int] = []
    for t in range(n_visits):
        visit_seed = seed + t  # hardcoded derivation rule; see docstring
        seeds_used.append(visit_seed)
        treated = treated_seq[t] == 1.0
        n_treated = int(treated.sum())
        if n_treated == 0 or n_treated == n:
            raise MSMError(
                f"treatments[{t}] must vary (one arm is empty)",
                code="msm_single_arm",
            )

        ps_den = _fit_visit_propensity(
            den_seq[t],
            treated_seq[t],
            seed=visit_seed,
            label=f"denominator visit {t}",
        )
        if not bool(np.isfinite(ps_den).all()):
            raise MSMError(
                f"denominator visit {t}: non-finite propensity scores",
                code="msm_propensity_not_finite",
                trim_proportion=1.0,
            )
        hard = (ps_den <= _PS_HARD_EPS) | (ps_den >= 1.0 - _PS_HARD_EPS)
        outside = (ps_den < low) | (ps_den > high)
        trim_proportion = float(outside.mean())
        if bool(hard.any()):
            raise MSMError(
                f"positivity violated at visit {t}: {int(hard.sum())}/{n} "
                f"denominator scores at or beyond [{_PS_HARD_EPS}, "
                f"{1.0 - _PS_HARD_EPS}] (trim-window proportion "
                f"{trim_proportion:.4f}); refusing to estimate",
                code="msm_propensity_out_of_bounds",
                trim_proportion=trim_proportion,
            )
        if bool(outside.any()):
            raise MSMError(
                f"positivity violated at visit {t}: {int(outside.sum())}/{n} "
                f"denominator scores outside trim window [{low}, {high}] "
                f"(trim proportion {trim_proportion:.4f}); refusing to trim "
                "silently",
                code="msm_positivity_trim_triggered",
                trim_proportion=trim_proportion,
            )

        if num_seq[t] is None:
            ps_num = np.full(n, float(treated_seq[t].mean()), dtype=float)
        else:
            assert num_seq[t] is not None
            ps_num = _fit_visit_propensity(
                num_seq[t],
                treated_seq[t],
                seed=visit_seed,
                label=f"numerator visit {t}",
            )
        if not bool(np.isfinite(ps_num).all()):
            raise MSMError(
                f"numerator visit {t}: non-finite propensity scores",
                code="msm_propensity_not_finite",
            )

        incr = np.where(treated, ps_num / ps_den, (1.0 - ps_num) / (1.0 - ps_den))
        if not bool(np.isfinite(incr).all()):
            raise MSMError(
                f"visit {t}: non-finite stabilised contributions",
                code="msm_weights_not_finite",
            )
        cumul = cumul * incr
        if not bool(np.isfinite(cumul).all()):
            raise MSMError(
                f"visit {t}: non-finite cumulative weights",
                code="msm_weights_not_finite",
            )
        per_visit.append(
            MSMVisitDiagnostics(
                visit_index=t,
                seed_used=visit_seed,
                n=n,
                ps_den_min=float(np.min(ps_den)),
                ps_den_max=float(np.max(ps_den)),
                ps_den_mean=float(np.mean(ps_den)),
                ps_num_min=float(np.min(ps_num)),
                ps_num_max=float(np.max(ps_num)),
                ps_num_mean=float(np.mean(ps_num)),
                incr_min=float(np.min(incr)),
                incr_max=float(np.max(incr)),
                cumul_mean=float(np.mean(cumul)),
                cumul_min=float(np.min(cumul)),
                cumul_max=float(np.max(cumul)),
                cumul_ess=_ess(cumul),
            )
        )

    truncated = False
    trunc_low: float | None = None
    trunc_high: float | None = None
    final_weights = cumul
    if trunc is not None:
        trunc_low = float(np.quantile(cumul, trunc[0]))
        trunc_high = float(np.quantile(cumul, trunc[1]))
        final_weights = np.clip(cumul, trunc_low, trunc_high)
        truncated = True

    return MSMWeightsResult(
        weights=np.asarray(final_weights, dtype=float),
        n=n,
        n_visits=n_visits,
        random_state=seed,
        seeds_used=tuple(seeds_used),
        ps_trim_low=low,
        ps_trim_high=high,
        trim_proportion=0.0,
        per_visit=tuple(per_visit),
        truncated=truncated,
        trunc_quantiles=trunc,
        trunc_low=trunc_low,
        trunc_high=trunc_high,
        weight_mean=float(np.mean(final_weights)),
        weight_min=float(np.min(final_weights)),
        weight_max=float(np.max(final_weights)),
        weight_ess=_ess(final_weights),
    )


__all__ = [
    "EVIDENCE_CEILING",
    "LIMITATIONS",
    "MSMError",
    "MSMVisitDiagnostics",
    "MSMWeightsResult",
    "stabilized_iptw",
]
