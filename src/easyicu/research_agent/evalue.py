"""E-value: sensitivity of an observed association to unmeasured confounding.

An adjusted effect estimate answers "how large is the association after the
covariates we measured?"; it says nothing about the covariates we *did not*
measure. The E-value (VanderWeele & Ding, *Ann Intern Med* 2017) turns that
worry into a single number: the minimum strength of association -- on the
risk-ratio scale -- that an unmeasured confounder would need with **both** the
exposure and the outcome, above and beyond the measured covariates, to fully
explain away the observed effect. A larger E-value means the finding is more
robust; a value near 1 means a weak confounder could nullify it.

Formula
-------
Let ``RR`` be the observed risk ratio (with ``RR`` reflected to the >= 1 side,
since the E-value is symmetric about the null ``RR = 1``: an ``RR`` of ``r`` and
its inverse ``1 / r`` carry the same confounding burden). Then

    E-value = RR + sqrt(RR * (RR - 1)).

For estimates on other scales we first approximate a risk ratio:

* hazard ratio (``kind="hr"``) -- treated directly as ``RR`` (the standard
  approximation; exact for rare outcomes, conservative otherwise);
* odds ratio with a rare outcome (``kind="or_rare"``) -- ``OR ~= RR`` directly;
* odds ratio with a common outcome (``kind="or_common"``) -- ``RR ~= sqrt(OR)``,
  VanderWeele & Ding's square-root approximation.

The E-value for a confidence interval uses the limit **closest to the null**:
if the interval excludes 1 that limit is the lower (for RR > 1) or upper (for
RR < 1) bound; if the interval already contains 1 the E-value is 1 (a confounder
of no strength is "needed" because the data are already compatible with the
null).

Pure ``math`` -- no numpy, scipy, or SDK dependency; importable anywhere.

Reference
---------
VanderWeele TJ, Ding P. "Sensitivity Analysis in Observational Research:
Introducing the E-Value." *Annals of Internal Medicine* 2017;167(4):268-274.
doi:10.7326/M16-2607.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_KINDS = ("rr", "hr", "or_rare", "or_common")


@dataclass(frozen=True)
class EValueResult:
    """E-values for a point estimate and its confidence interval."""

    point_evalue: float
    ci_evalue: float


def _as_risk_ratio(estimate: float, kind: str) -> float:
    """Approximate a risk ratio from ``estimate`` on the ``kind`` scale."""

    kind = str(kind).lower()
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {_VALID_KINDS!r}, got {kind!r}"
        )
    value = float(estimate)
    if not math.isfinite(value):
        raise ValueError("estimate must be a finite number")
    if value <= 0.0:
        raise ValueError("estimate (a ratio) must be strictly positive")
    if kind == "or_common":
        # Square-root approximation converts an odds ratio for a common
        # outcome onto the risk-ratio scale.
        return math.sqrt(value)
    # rr / hr / or_rare are all read directly as risk ratios.
    return value


def _evalue_from_rr(rr: float) -> float:
    """Core E-value formula on an already-risk-ratio-scaled estimate.

    Reflects ``rr`` to the >= 1 side (the E-value is symmetric about the
    null) and returns ``rr' + sqrt(rr' * (rr' - 1))``.
    """

    rr = float(rr)
    if rr <= 0.0:
        raise ValueError("risk ratio must be strictly positive")
    rr_prime = rr if rr >= 1.0 else 1.0 / rr
    return rr_prime + math.sqrt(rr_prime * (rr_prime - 1.0))


def evalue_point(estimate: float, kind: str = "rr") -> float:
    """E-value for a point estimate on the ``kind`` scale.

    ``kind`` in ``{"rr", "hr", "or_rare", "or_common"}``. The estimate is
    approximated as a risk ratio (identity for ``rr``/``hr``/``or_rare``,
    ``sqrt(OR)`` for ``or_common``), reflected to the >= 1 side, and passed
    through ``RR + sqrt(RR * (RR - 1))``. Returns ``1.0`` at the null.
    """

    rr = _as_risk_ratio(estimate, kind)
    return _evalue_from_rr(rr)


def evalue_ci(ci_low: float, ci_high: float, kind: str = "rr") -> float:
    """E-value for the confidence limit closest to the null (RR = 1).

    * CI entirely above 1 (``ci_low > 1``): apply the point formula to
      ``ci_low``.
    * CI entirely below 1 (``ci_high < 1``): apply the point formula to
      ``ci_high`` (reflected via the 1/x symmetry inside the core formula).
    * CI crossing 1 (``ci_low <= 1 <= ci_high``): return ``1.0`` -- the
      interval already includes the null, so no confounding is required.

    Both limits are first mapped onto the risk-ratio scale via ``kind``.
    """

    low = _as_risk_ratio(ci_low, kind)
    high = _as_risk_ratio(ci_high, kind)
    if low > high:
        raise ValueError("ci_low must not exceed ci_high (on the RR scale)")
    if low > 1.0:
        # Entirely above the null: nearest limit is the lower bound.
        return _evalue_from_rr(low)
    if high < 1.0:
        # Entirely below the null: nearest limit is the upper bound.
        return _evalue_from_rr(high)
    # Interval straddles RR = 1.
    return 1.0


def evalue(
    estimate: float,
    ci_low: float,
    ci_high: float,
    kind: str = "rr",
) -> EValueResult:
    """Bundle the point and CI E-values into an :class:`EValueResult`."""

    return EValueResult(
        point_evalue=evalue_point(estimate, kind),
        ci_evalue=evalue_ci(ci_low, ci_high, kind),
    )


__all__ = [
    "EValueResult",
    "evalue",
    "evalue_ci",
    "evalue_point",
]
