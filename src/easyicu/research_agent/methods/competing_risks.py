"""Cumulative incidence and baseline Fine-Gray regression under competing risks.

With competing events (e.g. death from another cause precludes the event of
interest), ``1 - Kaplan-Meier`` overestimates the absolute risk of the event
of interest because KM censors the competing events as if they merely left
follow-up. The **Aalen-Johansen estimator** generalises KM to this setting:
at each event time the overall survival just before that time is multiplied
by the cause-specific hazard of the event of interest, and those increments
are cumulated::

    CIF_k(t) = sum_{t_i <= t} S(t_i-) * d_{ki} / n_i

where ``S`` is the all-event survival, ``d_{ki}`` the number of type-``k``
events at ``t_i`` and ``n_i`` the risk set. When the data contain no
competing event, this reduces exactly to ``1 - KM`` (covered by a test).

Estimation is delegated to ``lifelines`` 0.30 ``AalenJohansenFitter`` — no
risk-set arithmetic is hand-rolled here. Ties in event times are jittered
inside lifelines; the fitter is constructed with an explicit fixed ``seed``
so results are bit-for-bit reproducible. Group comparison is a descriptive
CIF difference at a fixed time with a percentile-bootstrap CI. This module
deliberately provides **no Gray test**: lifelines 0.30 ships none, and this
kernel will not impersonate one. Any hypothesis test over CIF curves needs a
separately validated implementation.

``fit_fine_gray`` uses local R ``survival::finegray`` and weighted ``coxph``
with subject-clustered sandwich SEs. It consumes a closed 0/1/2 endpoint and
baseline covariates; the caller owns the cohort and administrative horizon.

Fail-closed validation: length mismatch, non-finite or negative durations,
non-integral or negative event codes, an unobserved ``event_of_interest``,
or anything other than exactly two groups for the comparison all raise
``ValueError``.

Claim ceiling: ``analysis_only``. Descriptive CIF summaries, not reportable
evidence of a group effect.

Reference
---------
Aalen OO, Johansen S. "An empirical transition matrix for non-homogeneous
Markov chains based on censored observations." *Scand J Stat* 1978;5:141-150.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple
import warnings
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter

from ..r_runtime import run_rscript


@dataclass(frozen=True)
class CIFResult:
    """Cumulative incidence curve for one event of interest."""

    times: Tuple[float, ...]
    cif: Tuple[float, ...]
    event_of_interest: int
    n: int
    n_interest: int
    n_competing: int
    n_censored: int
    random_state: int
    claim_ceiling: str = "analysis_only"
    method: str = "aalen_johansen_lifelines"

    def to_json(self) -> Dict[str, Any]:
        return {
            "times": list(self.times),
            "cif": list(self.cif),
            "event_of_interest": self.event_of_interest,
            "n": self.n,
            "n_interest": self.n_interest,
            "n_competing": self.n_competing,
            "n_censored": self.n_censored,
            "random_state": self.random_state,
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


@dataclass(frozen=True)
class CIFDifferenceResult:
    """Descriptive between-group CIF difference at a fixed time.

    No hypothesis test is performed: ``gray_test`` is ``False`` and
    ``hypothesis_test`` is ``None`` by construction.
    """

    time: float
    cif_a: float
    cif_b: float
    diff: float
    ci_low: float
    ci_high: float
    se: float
    group_a: Any
    group_b: Any
    n_a: int
    n_b: int
    event_of_interest: int
    n_bootstrap: int
    n_successful: int
    ci_level: float
    random_state: int
    gray_test: bool = False
    hypothesis_test: None = None
    claim_ceiling: str = "analysis_only"
    method: str = "aalen_johansen_descriptive_difference_bootstrap"

    def to_json(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "cif_a": self.cif_a,
            "cif_b": self.cif_b,
            "diff": self.diff,
            "ci": [self.ci_low, self.ci_high],
            "se": self.se,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "event_of_interest": self.event_of_interest,
            "n_bootstrap": self.n_bootstrap,
            "n_successful": self.n_successful,
            "ci_level": self.ci_level,
            "random_state": self.random_state,
            "gray_test": self.gray_test,
            "hypothesis_test": self.hypothesis_test,
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_inputs(
    durations: Sequence[float],
    events: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(list(durations), dtype=float)
    e_raw = np.asarray(list(events), dtype=float)
    if d.ndim != 1 or e_raw.ndim != 1:
        raise ValueError("durations and events must be one-dimensional")
    if d.shape[0] != e_raw.shape[0]:
        raise ValueError("durations and events must have equal length")
    if d.shape[0] < 1:
        raise ValueError("competing-risks estimation needs at least one subject")
    if not np.isfinite(d).all() or not np.isfinite(e_raw).all():
        raise ValueError("durations and events must all be finite (no NaN/inf)")
    if (d < 0.0).any():
        raise ValueError("durations must be non-negative")
    if ((e_raw < 0.0) | (e_raw != np.floor(e_raw))).any():
        raise ValueError("events must be non-negative integers (0 = censored)")
    return d, e_raw.astype(int)


def _validate_event_of_interest(events: np.ndarray, event_of_interest: object) -> int:
    code = _as_int(event_of_interest, "event_of_interest")
    if code <= 0:
        raise ValueError("event_of_interest must be a positive integer event code")
    if not (events == code).any():
        raise ValueError(f"event_of_interest={code} was never observed; refusing a zero curve")
    return code


def _fit_cif(
    durations: np.ndarray,
    events: np.ndarray,
    code: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit lifelines Aalen-Johansen on its native grid; return ``(times, cif)``.

    No ``timeline`` is passed through: lifelines 0.30 misaligns the
    Aalen-Johansen table when a custom timeline is supplied (NaN curve), so
    arbitrary-point reads use the exact step evaluation in
    :func:`_cif_at_time` instead.
    """
    fitter = AalenJohansenFitter(seed=seed)
    # lifelines' internal variance/CI arithmetic can overflow in exp on
    # degenerate resamples; only the point curve is consumed below, and its
    # finiteness is gated explicitly, so that numerics noise is suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitter.fit(
            pd.Series(np.asarray(durations, dtype=float)),
            pd.Series(np.asarray(events, dtype=int)),
            event_of_interest=code,
        )
    frame = fitter.cumulative_density_
    times = np.asarray(frame.index.to_numpy(dtype=float), dtype=float)
    values = np.asarray(frame.iloc[:, 0].to_numpy(dtype=float), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Aalen-Johansen fit produced non-finite CIF values")
    return times, values


def _cif_at_time(times: np.ndarray, values: np.ndarray, time: float) -> float:
    """Right-continuous step evaluation: value at the last knot ``<= time``."""
    pos = int(np.searchsorted(times, time, side="right")) - 1
    if pos < 0:
        return 0.0
    return float(values[pos])


def estimate_cif(
    durations: Sequence[float],
    events: Sequence[int],
    event_of_interest: int,
    timeline: Sequence[float] | None = None,
    random_state: int = 0,
) -> CIFResult:
    """Cumulative incidence function for ``event_of_interest``.

    ``events`` uses 0 for right-censoring and positive integers for distinct
    event types; every other positive code is treated as competing. ``seed``
    fixes lifelines' tie-jitter so the curve is reproducible. An explicit
    ``timeline`` is read off the fitted step curve (exact for a
    right-continuous step); it is never passed into lifelines.
    """
    d, e = _validate_inputs(durations, events)
    code = _validate_event_of_interest(e, event_of_interest)
    seed = _as_int(random_state, "random_state")
    times, values = _fit_cif(d, e, code, seed)
    if timeline is not None:
        tl = np.asarray(list(timeline), dtype=float)
        if tl.ndim != 1 or tl.shape[0] < 1:
            raise ValueError("timeline must be a non-empty one-dimensional sequence")
        if not np.isfinite(tl).all() or (tl < 0.0).any():
            raise ValueError("timeline must be finite and non-negative")
        if (np.diff(tl) < 0.0).any():
            raise ValueError("timeline must be non-decreasing")
        knots, curve = times, values
        times = np.asarray(tl, dtype=float)
        values = np.asarray([_cif_at_time(knots, curve, float(tt)) for tt in tl])
    return CIFResult(
        times=tuple(float(t) for t in times.tolist()),
        cif=tuple(float(v) for v in values.tolist()),
        event_of_interest=code,
        n=int(d.shape[0]),
        n_interest=int(np.sum(e == code)),
        n_competing=int(np.sum((e != 0) & (e != code))),
        n_censored=int(np.sum(e == 0)),
        random_state=seed,
    )


def cif_difference(
    durations: Sequence[float],
    events: Sequence[int],
    groups: Sequence[object],
    event_of_interest: int,
    time: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 0,
) -> CIFDifferenceResult:
    """Descriptive CIF difference ``CIF_a(time) - CIF_b(time)`` with bootstrap CI.

    ``groups`` must take exactly two distinct values. No Gray test, no
    p-value: the only uncertainty reported is the percentile-bootstrap CI of
    the descriptive difference at the fixed ``time``.
    """
    d, e = _validate_inputs(durations, events)
    code = _validate_event_of_interest(e, event_of_interest)
    g = np.asarray(list(groups), dtype=object)
    if g.ndim != 1 or g.shape[0] != d.shape[0]:
        raise ValueError("groups must be one-dimensional with the same length as durations")
    if isinstance(time, bool) or not isinstance(time, (int, float, np.floating)):
        raise ValueError("time must be a number")
    t = float(time)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("time must be finite and non-negative")

    labels = sorted({x for x in g.tolist()}, key=repr)
    if len(labels) != 2:
        raise ValueError(f"cif_difference needs exactly two groups, got {len(labels)}")
    label_a, label_b = labels[0], labels[1]

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

    mask_a = g == label_a
    mask_b = g == label_b

    def _group_cif_at(mask: np.ndarray, dd: np.ndarray, ee: np.ndarray) -> float:
        if int(np.sum(mask)) == 0:
            return float("nan")
        if not (ee[mask] == code).any():
            # Replicate (or group) with no event of interest observed: the
            # Aalen-Johansen curve is identically zero, which is the honest
            # conditional estimate. (A *study* group with no such event is
            # still refused fail-closed at the point-estimate gate below.)
            return 0.0
        times, values = _fit_cif(dd[mask], ee[mask], code, seed)
        return _cif_at_time(times, values, t)

    cif_a = _group_cif_at(mask_a, d, e)
    cif_b = _group_cif_at(mask_b, d, e)
    if not (e[mask_a] == code).any() or not (e[mask_b] == code).any():
        raise ValueError("both groups must observe the event of interest")
    if not np.isfinite(cif_a) or not np.isfinite(cif_b):  # pragma: no cover
        raise ValueError("group CIF evaluation failed")
    diff = cif_a - cif_b

    rng = np.random.default_rng(seed)
    n = d.shape[0]
    draws: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        da = _group_cif_at(mask_a[idx], d[idx], e[idx])
        db = _group_cif_at(mask_b[idx], d[idx], e[idx])
        delta = da - db
        if np.isfinite(delta):
            draws.append(float(delta))
    if not draws:
        raise ValueError("bootstrap produced no usable replicate")
    arr = np.asarray(draws, dtype=float)
    alpha = 1.0 - level
    ci_low = float(np.quantile(arr, alpha / 2.0))
    ci_high = float(np.quantile(arr, 1.0 - alpha / 2.0))
    se = float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0

    return CIFDifferenceResult(
        time=t,
        cif_a=cif_a,
        cif_b=cif_b,
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        group_a=label_a,
        group_b=label_b,
        n_a=int(np.sum(mask_a)),
        n_b=int(np.sum(mask_b)),
        event_of_interest=code,
        n_bootstrap=n_boot,
        n_successful=len(draws),
        ci_level=level,
        random_state=seed,
    )


__all__ = [
    "fit_fine_gray",
    "FineGrayFit",
    "FineGrayError",
    "CIFDifferenceResult",
    "CIFResult",
    "cif_difference",
    "estimate_cif",
]


_FINE_GRAY_RSCRIPT = r'''
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], check.names=FALSE)
terms <- grep("^easyicu_cov_", names(d), value=TRUE)
d$easyicu_status <- factor(d$easyicu_event, levels=0:2,
                          labels=c("censor", "interest", "competing"))
fit <- withCallingHandlers({
  expanded <- finegray(as.formula(paste(
    "Surv(easyicu_time, easyicu_status) ~",
    paste(c(terms, "easyicu_id"), collapse=" + ")
  )), data=d, etype="interest")
  coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~",
                         paste(terms, collapse=" + "))),
        data=expanded, weights=fgwt, cluster=easyicu_id, robust=TRUE,
        ties="efron", singular.ok=FALSE)
}, warning=function(w) {
  stop(paste("EASYICU_FINE_GRAY_WARNING:", conditionMessage(w)), call.=FALSE)
})
s <- summary(fit)$coefficients
if (!"robust se" %in% colnames(s)) stop("robust covariance unavailable")
write.csv(data.frame(term=rownames(s), coefficient=s[,"coef"],
                    standard_error=s[,"robust se"], p_value=s[,"Pr(>|z|)"],
                    r_version=as.character(getRversion()),
                    survival_version=as.character(packageVersion("survival"))),
          args[2], row.names=FALSE)
'''


class FineGrayError(ValueError):
    """The declared subdistribution model cannot be fitted reportably."""

    def __init__(self, message: str, *, code: str = "fine_gray_input_invalid"):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class FineGrayFit:
    estimates: pd.DataFrame
    receipt: dict[str, Any]


def fit_fine_gray(
    frame: pd.DataFrame, *, duration_col: str, event_col: str, covariates: Sequence[str],
) -> FineGrayFit:
    """Fit cause 1 vs cause 2 on complete numeric, baseline covariates.

    Each original row is one subject. Sandwich SEs cluster the expanded rows
    by that subject; censoring weights are pooled across subjects. The caller
    owns event recoding, horizon, and the complete-case population.
    """
    columns = list(covariates)
    required = [duration_col, event_col, *columns]
    if not columns or len(set(required)) != len(required) or not set(required) <= set(frame):
        raise FineGrayError("Fine-Gray requires distinct declared time, event and model columns")
    values = frame[required].apply(pd.to_numeric, errors="raise")
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise FineGrayError("Fine-Gray requires complete finite input")
    if (values[duration_col] <= 0).any() or not values[event_col].isin([0, 1, 2]).all():
        raise FineGrayError("Fine-Gray requires positive time and 0/1/2 event codes")
    if not {1, 2} <= set(values[event_col]):
        raise FineGrayError("Fine-Gray requires observed interest and competing events")
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise FineGrayError("local Rscript is unavailable", code="fine_gray_runtime_unavailable")
    mapping = {f"easyicu_cov_{i}": name for i, name in enumerate(columns)}
    local = pd.DataFrame({
        "easyicu_id": np.arange(len(values)),
        "easyicu_time": values[duration_col].to_numpy(),
        "easyicu_event": values[event_col].to_numpy(),
        **{alias: values[name].to_numpy() for alias, name in mapping.items()},
    })
    with tempfile.TemporaryDirectory(prefix="easyicu-fine-gray-") as directory:
        root = Path(directory)
        local.to_csv(root / "input.csv", index=False)
        completed = run_rscript(
            rscript=rscript, script=_FINE_GRAY_RSCRIPT,
            args=[str(root / "input.csv"), str(root / "estimates.csv")], timeout_s=60,
        )
        if completed.timed_out:
            raise FineGrayError("Fine-Gray fit timed out", code="fine_gray_fit_timeout")
        if "EASYICU_FINE_GRAY_WARNING:" in completed.stderr:
            raise FineGrayError("Fine-Gray fit emitted a warning", code="fine_gray_fit_warning")
        if completed.returncode != 0 or not (root / "estimates.csv").is_file():
            raise FineGrayError("Fine-Gray fit failed", code="fine_gray_fit_failed")
        result = pd.read_csv(root / "estimates.csv")
    required_result = {"term", "coefficient", "standard_error", "p_value", "r_version", "survival_version"}
    if not required_result <= set(result) or result.term.tolist() != list(mapping):
        raise FineGrayError("Fine-Gray coefficient roster mismatch", code="fine_gray_result_invalid")
    result["hazard_ratio"] = np.exp(result.coefficient)
    result["ci_low"] = np.exp(result.coefficient - 1.959963984540054 * result.standard_error)
    result["ci_high"] = np.exp(result.coefficient + 1.959963984540054 * result.standard_error)
    numeric = result[["coefficient", "standard_error", "hazard_ratio", "ci_low", "ci_high", "p_value"]]
    if (not np.isfinite(numeric.to_numpy()).all() or (result.standard_error <= 0).any()
            or (result.ci_low <= 0).any() or not result.p_value.between(0, 1).all()):
        raise FineGrayError("Fine-Gray returned invalid inference", code="fine_gray_result_invalid")
    receipt = {
        "method": "survival::finegray+coxph", "ties": "efron",
        "variance_estimator": "subject_cluster_robust", "censoring_weights": "pooled",
        "n": len(values), "n_events": int(values[event_col].eq(1).sum()),
        "r_version": str(result.r_version.iloc[0]),
        "survival_version": str(result.survival_version.iloc[0]),
        "proportional_subdistribution_hazards_tested": False,
    }
    result["term"] = result.term.map(mapping)
    return FineGrayFit(result.drop(columns=["r_version", "survival_version"]), receipt)
