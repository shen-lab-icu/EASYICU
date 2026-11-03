"""Gray's K-sample test for cumulative incidence functions (analysis only).

This kernel is a NumPy port of the score and variance routine that CRAN
``cmprsk`` uses for its ``cuminc()`` K-sample test: subroutines ``crstm`` and
``crst`` in ``src/crstm.f`` (Robert Gray, GPL; port reference: cmprsk
2.2-12, ``crstm.f`` sha256
``a359f29936573aebe84a56abdf91edf26cf7e46464f473c1b8b85ff1b650b829``).
``lifelines`` ships no Gray test, so there is no Python package quantity to
wrap; the oracle values in
``tests/research_agent/core/test_gray_test_matches_the_cmprsk_kernel.py``
were produced by compiling that Fortran file with gfortran and calling it on
each fixture, so the port is checked against the original kernel rather
than against its own arithmetic.

Statistic (Gray 1988). Subjects are sorted by time and processed one
distinct time at a time. For each group ``k`` the kernel keeps the
left-continuous all-cause Kaplan-Meier ``S_k(t-)`` and the left-continuous
cause-of-interest cumulative incidence ``F_k(t-)``. With ``Y_k`` the
ordinary risk set (``T >= t``), ``R_k = Y_k (1 - F_k(t-)) / S_k(t-)`` and
``d_k`` the cause-of-interest events at ``t``, the score for group ``k``
accumulates ``W(t) (d_k - d R_k / sum_r R_r)``. Groups whose ordinary risk
set is empty contribute nothing at that time. The time weight is
``W(t) = (1 - F(t-))^rho`` where ``F`` is the pooled cumulative incidence
of the event of interest (``rho=0`` gives ``W=1``). The covariance is
Gray's asymptotic estimator, including the correction terms for censoring
and for competing events (``crst`` arrays ``a``, ``c``, ``v2`` and
``v3``); it is *not* a log-rank hypergeometric variance, so even without
competing events the result is close to, but not identical with, a
log-rank test.

The test statistic is ``s' V^{-1} s`` over the first ``K-1`` groups on
``K-1`` degrees of freedom, as ``cuminc()`` computes it. ``cuminc()``
reports a statistic of ``-1`` when ``V`` is singular; this kernel refuses
instead (:class:`GrayTestError`). A single stratum is supported (the
public signature has no strata argument).

Determinism contract: pure NumPy arithmetic on stably sorted times with
deterministic label ordering; no RNG is involved, so rerunning with
identical inputs yields byte-identical :meth:`GrayTestResult.to_json`.

Fail-closed inputs (all raise :class:`GrayTestError`, a ``ValueError``):

* event codes outside ``{0, 1, 2}`` (0 = censored, 1 and 2 = the two
  causes) or a non-integral code;
* no observed event of interest;
* a single distinct group;
* an empty input, negative or non-finite durations, length mismatches;
* a singular score covariance (``cuminc()`` would print ``-1``);
* any non-finite intermediate value.

Claim ceiling: ``analysis_only``. A group comparison of cumulative
incidence is not reportable evidence of a group effect without a
preregistered protocol and independent review.

References
----------
Gray RJ. "A class of K-sample tests for comparing the cumulative incidence
of a competing risk." *Ann Stat* 1988;16(3):1141-1154.
Gray RJ. cmprsk: Subdistribution Analysis of Competing Risks. R package
version 2.2-12, ``src/crstm.f``.
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


def _crst_single_stratum(
    times: np.ndarray,
    cause: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of cmprsk ``crst`` for one stratum.

    ``times`` must be sorted ascending; ``cause`` is 0 (censored), 1 (event
    of interest) or 2 (any other cause); ``group`` holds 0-based group
    indices. Returns the scores of all ``n_groups`` groups (``crst`` itself
    returns the first ``n_groups - 1``; the last is computed by the same
    update) and the ``(n_groups-1)``-square covariance of the first
    ``n_groups - 1`` scores. Comments name the Fortran arrays and labels.
    """

    ng = n_groups
    ng1 = ng - 1
    n = int(times.shape[0])
    rs = np.bincount(group, minlength=ng).astype(float)  # ordinary risk sets
    score = np.zeros(ng, dtype=float)
    v = np.zeros((ng1, ng1), dtype=float)  # packed lower triangle in Fortran
    f1m = np.zeros(ng, dtype=float)  # left-continuous group CIF
    f1 = np.zeros(ng, dtype=float)  # right-continuous group CIF
    skmm = np.ones(ng, dtype=float)  # left-continuous group all-cause KM
    skm = np.ones(ng, dtype=float)  # right-continuous group all-cause KM
    v3 = np.zeros(ng, dtype=float)
    v2 = np.zeros((ng1, ng), dtype=float)
    c = np.zeros((ng, ng), dtype=float)
    fm = 0.0  # left-continuous pooled CIF
    f = 0.0  # right-continuous pooled CIF

    ll = 0
    while ll < n:
        lu = ll
        while lu + 1 < n and not times[lu + 1] > times[ll]:
            lu += 1
        d = np.zeros((3, ng), dtype=float)  # d(0:2, ng)
        np.add.at(d, (cause[ll : lu + 1], group[ll : lu + 1]), 1.0)
        nd1 = float(np.sum(d[1]))
        nd2 = float(np.sum(d[2]))
        if nd1 > 0.0 or nd2 > 0.0:
            live = rs > 0.0
            tr = 0.0
            tq = 0.0
            for i in range(ng):  # label 60
                if not live[i]:
                    continue
                td = d[1, i] + d[2, i]
                skm[i] = skmm[i] * (rs[i] - td) / rs[i]
                f1[i] = f1m[i] + (skmm[i] * d[1, i]) / rs[i]
                tr += rs[i] / skmm[i]
                tq += rs[i] * (1.0 - f1m[i]) / skmm[i]
            f = fm + nd1 / tr
            fb = (1.0 - fm) ** rho
            a = np.zeros((ng, ng), dtype=float)
            for i in range(ng):  # labels 66/67
                if not live[i]:
                    continue
                t1 = rs[i] / skmm[i]
                a[i, i] = fb * t1 * (1.0 - t1 / tr)
                if a[i, i] != 0.0:
                    c[i, i] += a[i, i] * nd1 / (tr * (1.0 - fm))
                for j in range(i + 1, ng):
                    if not live[j]:
                        continue
                    a[i, j] = -fb * t1 * rs[j] / (skmm[j] * tr)
                    if a[i, j] != 0.0:
                        c[i, j] += a[i, j] * nd1 / (tr * (1.0 - fm))
            for i in range(1, ng):  # labels 68/69: mirror the upper triangle
                for j in range(i):
                    a[i, j] = a[j, i]
                    c[i, j] = c[j, i]
            for i in range(ng):  # label 74 (crst stops at ng-1)
                if not live[i]:
                    continue
                score[i] += fb * (
                    d[1, i] - nd1 * rs[i] * (1.0 - f1m[i]) / (skmm[i] * tq)
                )
            if nd1 > 0.0:  # labels 72/70/71
                for k in range(ng):
                    if not live[k]:
                        continue
                    t4 = 1.0
                    if skm[k] > 0.0:
                        t4 = 1.0 - (1.0 - f) / skm[k]
                    t5 = 1.0
                    if nd1 > 1.0:
                        t5 = 1.0 - (nd1 - 1.0) / (tr * skmm[k] - 1.0)
                    t3 = t5 * skmm[k] * nd1 / (tr * rs[k])
                    v3[k] += t4 * t4 * t3
                    t1 = a[:ng1, k] - t4 * c[:ng1, k]
                    v2[:, k] += t1 * t4 * t3
                    v += np.outer(t1, t1) * t3
            if nd2 > 0.0:  # labels 82/80/81
                for k in range(ng):
                    if skm[k] <= 0.0 or d[2, k] <= 0.0:
                        continue
                    t4 = (1.0 - f) / skm[k]
                    t5 = 1.0
                    if d[2, k] > 1.0:
                        t5 = 1.0 - (d[2, k] - 1.0) / (rs[k] - 1.0)
                    t3 = t5 * ((skmm[k] ** 2) * d[2, k]) / (rs[k] ** 2)
                    v3[k] += t4 * t4 * t3
                    t1 = t4 * c[:ng1, k]
                    v2[:, k] -= t1 * t4 * t3
                    v += np.outer(t1, t1) * t3
        # label 90: leave this time
        np.subtract.at(rs, group[ll : lu + 1], 1.0)
        fm = f
        f1m = f1.copy()
        skmm = skm.copy()
        ll = lu + 1

    # label 30: add the censoring/competing correction terms
    cu = c[:ng1, :]
    v += (cu * v3) @ cu.T + cu @ v2.T + v2 @ cu.T
    return score, v


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
    method: str = "gray_1988_k_sample_cmprsk_crstm"

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

    ``events`` uses 0 for right-censoring and 1/2 for the two causes;
    ``event_of_interest`` picks which cause is compared and the other is the
    competing event. ``groups`` holds one label per subject. ``rho >= 0``
    selects Gray's time weight ``W(t) = (1 - F(t-))^rho`` on the pooled
    cumulative incidence of the event of interest, as in ``cuminc()``.
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

    order = np.argsort(times, kind="stable")
    cause = np.where(codes == code, 1, np.where(codes == competing, 2, 0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score, cov = _crst_single_stratum(
            times[order], cause[order], group_pos[order], n_groups, rho_value
        )
    reduced_score = score[:-1]
    if not (np.all(np.isfinite(score)) and np.all(np.isfinite(cov))):
        raise GrayTestError("Gray score computation produced non-finite values")
    # cuminc() takes a QR of the covariance and reports -1 unless it has full
    # rank; refuse instead of inventing a statistic.
    if np.linalg.matrix_rank(cov) < n_groups - 1:
        raise GrayTestError(
            "Gray score covariance is singular; the K-sample statistic is undefined"
        )
    try:
        statistic = float(reduced_score @ np.linalg.solve(cov, reduced_score))
    except np.linalg.LinAlgError as exc:  # pragma: no cover - rank checked above
        raise GrayTestError(f"Gray covariance inversion failed: {exc}") from exc
    if not np.isfinite(statistic) or statistic < 0.0:
        # Tiny negative values are floating-point noise around a true zero
        # (identical CIFs); anything clearly negative is a bug, fail closed.
        if np.isfinite(statistic) and statistic > -1e-12:
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
