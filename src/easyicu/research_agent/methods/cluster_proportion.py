"""Fixed-weight patient-cluster proportion bounds (Hoeffding 1963, Thm. 2).

For independent patient clusters, let Z_i be the within-patient event fraction
and w_i=m_i/N the fixed fraction of denominator stays contributed by patient i.
Z_i lies in [0,1], with arbitrary dependence among that patient's stays.
Applying the theorem to w_i Z_i and both tails gives
P(|sum(w_i Z_i)-sum(w_i E[Z_i])| >= r) <= 2 exp(-2 r^2/sum(w_i^2)).

The target is conditional on the fixed denominator weights, not an equally
weighted patient risk or the ratio from an unrestricted random-size cluster
population. The plan must declare this target and conditional independence.
DOI: 10.1080/01621459.1963.10500830, equation (2.6).
"""

from __future__ import annotations

import math
from numbers import Integral, Real


def patient_weighted_hoeffding_interval(
    events: int,
    denominator: int,
    *,
    cluster_count: int,
    cluster_size_square_sum: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Return outward-rounded percentage bounds from actual patient weights.

    The sum of squared cluster sizes is a sufficient arithmetic receipt; it
    is computed from resolved patient identities on this exact denominator.
    It is not the distinct-patient count or a binomial trial count. No standard
    error or covariance estimate is constructed by this concentration bound.
    """

    for name, value in (
        ("events", events), ("denominator", denominator),
        ("cluster_count", cluster_count),
        ("cluster_size_square_sum", cluster_size_square_sum),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer count")
    if denominator <= 0 or not 0 <= events <= denominator:
        raise ValueError("events and denominator do not define a proportion")
    if not 2 <= cluster_count <= denominator:
        raise ValueError("the denominator requires at least two patient clusters")
    quotient, remainder = divmod(denominator, cluster_count)
    minimum = (cluster_count - remainder) * quotient**2 + remainder * (quotient + 1)**2
    maximum = (denominator - cluster_count + 1)**2 + cluster_count - 1
    if (
        not minimum <= cluster_size_square_sum <= maximum
        or (cluster_size_square_sum - denominator) % 2
    ):
        raise ValueError("cluster_size_square_sum contradicts the denominator and patients")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not math.isfinite(confidence_level)
        or not 0.5 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite and strictly between 0.5 and 1")
    weight_square_sum = cluster_size_square_sum / denominator**2
    radius = math.sqrt(
        0.5 * weight_square_sum * (math.log(2.0) - math.log1p(-confidence_level))
    )
    estimate = events / denominator
    # Nearest rounding could shrink the covered set. Round each endpoint out.
    low = math.floor(100_000_000 * max(0.0, estimate - radius)) / 1_000_000
    high = math.ceil(100_000_000 * min(1.0, estimate + radius)) / 1_000_000
    return low, high
