"""Gray's K-sample test for cumulative incidence functions (analysis only).

Hand-rolled implementation (wrap-vs-rewrite rule, reason 1: ``lifelines``
0.30 ships no Gray test, so there is no package quantity to wrap; reason 3:
every risk-set step is auditable here). The statistic follows Gray (1988):
a weighted log-rank-type score over the cause of interest, computed on
*subdistribution* risk sets in which subjects who already failed from the
competing cause are retained with inverse-probability-of-censoring weights
(Fine & Gray 1999 weighting), with Gray's rho-family time weight
``W(t) = S(t-)^rho`` (default ``rho=0``, i.e. Gray's test proper, ``W=1``).

Score construction (``K`` groups, cause of interest ``c``, competing cause
``c'``). At each distinct time ``t_j`` with a cause-``c`` event:

* modified risk set per group
  ``Y*_ij = #{T >= t_j, group i} + sum G(t_j)/G(T_l)`` over group-``i``
  subjects with ``T_l < t_j`` and a competing event, where ``G`` is the
  pooled Kaplan-Meier estimate of the censoring distribution
  (right-continuous step; identically 1 when nothing is censored);
* observed ``d_ij`` and expected ``e_ij = d_j Y*_ij / Y*_j`` cause-``c``
  events, accumulated as ``Z_i += W_j (d_ij - e_ij)``;
* covariance accumulated in log-rank form,
  ``V[i,l] += f_j W_j^2 (delta_il Y*_ij - Y*_ij Y*_lj / Y*_j)`` with
  ``f_j = ((Y*_j - d_j)/(Y*_j - 1)) d_j / Y*_j^2`` (the ``Y*_j = 1`` case
  takes the same ``1`` that the ``inf -> 1`` replacement gives there).

The test statistic is ``Z' pinv(V) Z`` on the first ``K-1`` groups with
``K-1`` degrees of freedom — the same algebraic shape as the lifelines
multivariate log-rank, which is exactly why property (a) below holds.

Because no R/cmp rsk reference is available in this environment, validity is
self-evidenced by two hard properties (both covered by tests):

(a) *Degeneracy.* With no competing event the carried set is empty, so the
    modified risk sets equal the standard risk sets and ``W = 1``; the
    algebra above is then term-for-term the lifelines multivariate log-rank
    on the cause-``c`` indicator, and the p-values must agree (tolerance
    1e-6 in the test).
(b) *Hand-worked competing-risk example.* Four subjects, ``1`` = interest,
    ``2`` = competing, groups ``A``/``B``::

        A1: T=1 cause 2 | B1: T=2 cause 1 | A2: T=3 cause 1 | B2: T=4 cause 1

    Nothing is censored, so every IPCW weight is exactly ``G/G = 1`` while
    the carry-over is still exercised (at ``t=2`` group A has one subject
    under follow-up but a subdistribution risk set of two — a plain
    log-rank censoring the competing event would use one). Step by step:

    * ``t=2``: ``Y*_A=2, Y*_B=2, Y*=4``, ``d=1`` (B1).
      ``E_A=E_B=0.5``; ``Z_A += -0.5``, ``Z_B += +0.5``.
      ``f = (3/3)(1/16) = 1/16``; ``V_AA += (1/16)(2)(2) = 1/4``.
    * ``t=3``: ``Y*_A=2`` (A2 plus carried A1), ``Y*_B=1``, ``Y*=3``,
      ``d=1`` (A2). ``E_A=2/3``; ``Z_A += +1/3``.
      ``f = (2/2)(1/9) = 1/9``; ``V_AA += (1/9)(2)(1) = 2/9``.
    * ``t=4``: ``Y*_A=1`` (carried A1), ``Y*_B=1``, ``Y*=2``, ``d=1``
      (B2). ``E_A=0.5``; ``Z_A += -0.5``.
      ``f = (1/1)(1/4) = 1/4``; ``V_AA += (1/4)(1)(1) = 1/4``.

    Totals: ``Z_A = -1/2 + 1/3 - 1/2 = -2/3``,
    ``V_AA = 1/4 + 2/9 + 1/4 = 13/18``, so
    ``chi2 = (4/9)/(13/18) = 8/13`` on 1 df
    (``p = chi2.sf(8/13, 1) ~= 0.4328``). The test pins ``8/13`` to 1e-12.

Determinism contract: pure NumPy arithmetic on sorted unique times with
deterministic label ordering; no RNG is involved, so rerunning with
identical inputs yields byte-identical :meth:`GrayTestResult.to_json`.

Fail-closed inputs (all raise :class:`GrayTestError`, a ``ValueError``):

* event codes outside ``{0, 1, 2}`` (0 = censored, 1 = event of interest,
  2 = competing) or a non-integral code;
* no observed event of interest (the test would have an empty risk set);
* a single distinct group;
* an empty input, negative or non-finite durations, length mismatches;
* a censoring-KM value of zero at a needed time (IPCW weights undefined);
* a non-positive modified risk set at a cause-``c`` event time.

Known limitation (documented, not silently worked around): the censoring
distribution ``G`` is pooled across groups, as in Gray's original proposal.
If censoring depends strongly on group, the IPCW weights are misspecified
and the test can mis-lead; that setting needs a group-stratified extension
this kernel does not claim.

Claim ceiling: ``analysis_only``. A group comparison of cumulative
incidence is not reportable evidence of a group effect without a
preregistered protocol and independent review.

References
----------
Gray RJ. "A class of K-sample tests for comparing the cumulative incidence
of a competing risk." *Ann Stat* 1988;16(3):1141-1154.
Fine JP, Gray RJ. "A proportional hazards model for the subdistribution of
a competing risk." *J Am Stat Assoc* 1999;94(446):496-509.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy import stats

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"

_ALLOWED_CODES = (0, 1, 2)


class GrayTestError(ValueError):
    """A Gray-test input the kernel refuses to run on."""


def _coerce_durations(durations: object, n_expected: int | None = None) -> np.ndarray:
    try:
        values = np.asarray(list(durations), dtype=float).ravel()
    except (TypeError, ValueError):
        raise GrayTestError("durations must be a numeric one-dimensional vector") from None
    if n_expected is not None and values.shape[0] != n_expected:
        raise GrayTestError("durations, events and groups must have equal length")
    if values.shape[0] == 0:
        raise GrayTestError("Gray's test needs at least one subject (empty risk set)")
    if not np.all(np.isfinite(values)):
        raise GrayTestError("durations must all be finite (no NaN/inf)")
    if np.any(values < 0.0):
        raise GrayTestError("durations must be non-negative")
    return values


def _coerce_events(events: object, n: int) -> np.ndarray:
    try:
        raw = np.asarray(list(events), dtype=float).ravel()
    except (TypeError, ValueError):
        raise GrayTestError("events must be a numeric one-dimensional vector") from None
    if raw.shape[0] != n:
        raise GrayTestError("durations, events and groups must have equal length")
    if raw.shape[0] == 0:
        raise GrayTestError("Gray's test needs at least one subject (empty risk set)")
    if not np.all(np.isfinite(raw)):
        raise GrayTestError("events must all be finite (no NaN/inf)")
    if np.any(raw != np.floor(raw)):
        raise GrayTestError(
            "event codes must be integers in {0=censored, 1=interest, 2=competing}"
        )
    codes = raw.astype(int)
    illegal = sorted(set(codes.tolist()) - set(_ALLOWED_CODES))
    if illegal:
        raise GrayTestError(
            "event codes must be 0=censored, 1=event of interest, 2=competing; "
            f"illegal codes observed: {illegal}"
        )
    return codes


def _censoring_km_right_continuous(
    times: np.ndarray, is_censored: np.ndarray
) -> Dict[float, float]:
    """Pooled KM of the censoring distribution as a right-continuous step."""

    curve: Dict[float, float] = {}
    survival = 1.0
    for moment in sorted(set(times.tolist())):
        at_risk = int(np.sum(times >= moment))
        censored_here = int(np.sum((times == moment) & is_censored))
        if at_risk <= 0:  # pragma: no cover - defensive; a subject sits at its own time
            raise GrayTestError("empty risk set while estimating the censoring curve")
        survival *= (at_risk - censored_here) / at_risk
        curve[float(moment)] = float(survival)
    return curve


def _eventfree_km_left_continuous(
    times: np.ndarray, any_event: np.ndarray
) -> Dict[float, float]:
    """Pooled event-free survival just *before* each time (for rho weights)."""

    left: Dict[float, float] = {}
    survival = 1.0
    for moment in sorted(set(times.tolist())):
        left[float(moment)] = float(survival)
        at_risk = int(np.sum(times >= moment))
        failed_here = int(np.sum((times == moment) & any_event))
        survival *= (at_risk - failed_here) / at_risk
    return left


@dataclass(frozen=True)
class GrayTestResult:
    """Typed Gray K-sample output; JSON form is the digest input."""

    statistic: float
    df: int
    p_value: float
    group_labels: tuple[Any, ...]
    group_scores: tuple[float, ...]
    group_n: tuple[int, ...]
    n: int
    n_interest: int
    n_competing: int
    n_censored: int
    rho: float
    event_of_interest: int
    competing_event: int
    claim_ceiling: str = "analysis_only"
    method: str = "gray_k_sample_score_ipcw"

    def to_json(self) -> Dict[str, Any]:
        return {
            "statistic": float(self.statistic),
            "df": int(self.df),
            "p_value": float(self.p_value),
            "group_labels": list(self.group_labels),
            "group_scores": [float(value) for value in self.group_scores],
            "group_n": [int(value) for value in self.group_n],
            "n": int(self.n),
            "n_interest": int(self.n_interest),
            "n_competing": int(self.n_competing),
            "n_censored": int(self.n_censored),
            "rho": float(self.rho),
            "event_of_interest": int(self.event_of_interest),
            "competing_event": int(self.competing_event),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def result_sha256(result: GrayTestResult) -> str:
    """Return the canonical digest of one Gray-test result."""

    if not isinstance(result, GrayTestResult):
        raise TypeError("result_sha256 requires a GrayTestResult")
    return canonical_sha256(result.to_json())


def gray_test(
    durations: object,
    events: object,
    groups: object,
    *,
    event_of_interest: int = 1,
    rho: float = 0.0,
) -> GrayTestResult:
    """Run Gray's K-sample test for the cause-``event_of_interest`` CIFs.

    ``events`` uses 0 for right-censoring, 1 for the event of interest and
    2 for the competing event. ``groups`` holds one label per subject.
    ``rho >= 0`` selects Gray's rho-family time weight
    ``W(t) = S(t-)^rho``; ``rho=0`` is Gray's test proper.
    """

    if isinstance(event_of_interest, bool) or event_of_interest not in (1, 2):
        raise GrayTestError("event_of_interest must be 1 or 2")
    code = int(event_of_interest)
    competing = 2 if code == 1 else 1
    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.floating)):
        raise GrayTestError("rho must be a non-negative number")
    rho_value = float(rho)
    if not np.isfinite(rho_value) or rho_value < 0.0:
        raise GrayTestError("rho must be a finite non-negative number")

    times = _coerce_durations(durations)
    n = int(times.shape[0])
    codes = _coerce_events(events, n)
    group_array = np.asarray(list(groups), dtype=object)
    if group_array.ndim != 1 or group_array.shape[0] != n:
        raise GrayTestError("durations, events and groups must have equal length")
    labels = sorted(set(group_array.tolist()), key=repr)
    if len(labels) < 2:
        raise GrayTestError(
            f"Gray's test needs at least two groups, got {len(labels)}"
        )
    n_groups = len(labels)
    group_index = {label: pos for pos, label in enumerate(labels)}
    group_pos = np.asarray([group_index[label] for label in group_array.tolist()])

    n_interest = int(np.sum(codes == code))
    if n_interest == 0:
        raise GrayTestError(
            f"event of interest {code} was never observed; "
            "the test has an empty risk set"
        )
    n_competing = int(np.sum(codes == competing))
    n_censored = int(np.sum(codes == 0))

    censor_curve = _censoring_km_right_continuous(times, codes == 0)
    eventfree_left = _eventfree_km_left_continuous(times, codes != 0)

    score = np.zeros(n_groups, dtype=float)
    observed_total = np.zeros(n_groups, dtype=float)
    cov = np.zeros((n_groups, n_groups), dtype=float)

    cause_times = sorted(set(times[codes == code].tolist()))
    for moment in cause_times:
        under_followup = (times >= moment).astype(float)
        weight = 1.0 if rho_value == 0.0 else float(eventfree_left[float(moment)]) ** rho_value
        if not np.isfinite(weight):
            raise GrayTestError("Gray rho weight is non-finite")
        risk = np.zeros(n_groups, dtype=float)
        for pos in range(n_groups):
            in_group = group_pos == pos
            risk[pos] = float(np.sum(under_followup[in_group] > 0.0))
            past_competing = (times[in_group] < moment) & (
                codes[in_group] == competing
            )
            for past in times[in_group][past_competing].tolist():
                denom = censor_curve[float(past)]
                if denom <= 0.0:
                    raise GrayTestError(
                        "censoring KM is zero at a competing-event time; "
                        "IPCW weights are undefined here"
                    )
                risk[pos] += censor_curve[float(moment)] / denom
        risk_total = float(np.sum(risk))
        if not np.isfinite(risk_total) or risk_total <= 0.0:
            raise GrayTestError("empty subdistribution risk set at a cause event time")
        at_moment = times == moment
        observed = np.zeros(n_groups, dtype=float)
        for pos in range(n_groups):
            observed[pos] = float(
                np.sum(at_moment & (codes == code) & (group_pos == pos))
            )
        events_now = float(np.sum(observed))
        score += weight * (observed - events_now * risk / risk_total)
        observed_total += observed
        if events_now > 0.0:
            if risk_total <= 1.0:
                factor = events_now / risk_total**2
            else:
                factor = (
                    (risk_total - events_now)
                    / (risk_total - 1.0)
                    * events_now
                    / risk_total**2
                )
            scaled = factor * weight * weight
            # Log-rank covariance on the (modified) risk sets: off-diagonal
            # -f W^2 Y_i Y_l, diagonal f W^2 Y_i (Y* - Y_i). This mirrors the
            # lifelines multivariate-log-rank matrix algebra term for term,
            # which is what makes the no-competition degeneracy exact.
            for i in range(n_groups):
                for ell in range(n_groups):
                    cov[i, ell] += scaled * (
                        (risk_total * risk[i] if i == ell else 0.0)
                        - risk[i] * risk[ell]
                    )

    reduced_score = score[:-1]
    reduced_cov = cov[:-1, :-1]
    if not (np.all(np.isfinite(reduced_score)) and np.all(np.isfinite(reduced_cov))):
        raise GrayTestError("Gray score computation produced non-finite values")
    try:
        inverse = np.linalg.pinv(reduced_cov)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - pinv rarely raises
        raise GrayTestError(f"Gray covariance inversion failed: {exc}") from exc
    statistic = float(reduced_score @ inverse @ reduced_score)
    if not np.isfinite(statistic) or statistic < 0.0:
        # Tiny negative values are floating-point noise around a true zero
        # (identical CIFs); anything clearly negative is a bug, fail closed.
        if statistic > -1e-12:
            statistic = 0.0
        else:
            raise GrayTestError("Gray statistic is inadmissible (negative)")
    degrees = n_groups - 1
    p_value = float(stats.chi2.sf(statistic, degrees))
    if not np.isfinite(p_value):
        raise GrayTestError("Gray p-value is non-finite")

    counts = tuple(int(np.sum(group_pos == pos)) for pos in range(n_groups))
    return GrayTestResult(
        statistic=statistic,
        df=degrees,
        p_value=p_value,
        group_labels=tuple(labels),
        group_scores=tuple(float(value) for value in score.tolist()),
        group_n=counts,
        n=n,
        n_interest=n_interest,
        n_competing=n_competing,
        n_censored=n_censored,
        rho=rho_value,
        event_of_interest=code,
        competing_event=competing,
    )


__all__ = [
    "TOOL_VERSION",
    "GrayTestError",
    "GrayTestResult",
    "gray_test",
    "result_sha256",
]
