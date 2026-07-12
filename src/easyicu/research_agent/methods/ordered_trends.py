"""Validated primitives for trends across ordered independent groups.

This module deliberately stops below the research-workflow layer.  It does not
choose variables, define a cohort, label clinical categories, write a result
table, or decide what belongs in a manuscript.  Those remain agent decisions.
The functions here only provide reusable, deterministic calculations that an
agent-authored analysis can call and that a validator can independently replay.

Two tests are exposed:

``cochran_armitage_trend``
    A score test for a binary outcome across ordered groups.  Its scores are an
    explicit modelling choice: consecutive ranks are the default, but callers
    may pass clinically justified scores.  Consequently this test must not be
    described as using order alone.

``jonckheere_terpstra_trend``
    The Jonckheere--Terpstra test for an ordered shift in an ordinal or
    continuous outcome.  It uses individual observations, counts cross-group
    concordant/discordant pairs, gives response ties half credit in ``J``, and
    uses the tie-corrected asymptotic null variance.  Spearman correlation is
    not a substitute for this calculation.

Very small normal-tail probabilities are retained on the log scale.  The
machine-readable ``p_value`` is bounded below by ``1e-300`` so CSV/JSON output
never states ``p=0``; ``log_p_value`` and ``p_value_reporting`` preserve the
actual tail information.

References
----------
Jonckheere AR. A distribution-free k-sample test against ordered alternatives.
Biometrika. 1954;41:133-145.

Terpstra TJ. The asymptotic normality and consistency of Kendall's test against
trend, when ties are present in one ranking. Indagationes Mathematicae.
1952;14:327-333.

SAS/STAT FREQ documentation, Jonckheere--Terpstra test details:
https://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/
procstat_freq_details75.htm
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

Alternative = Literal["two-sided", "greater", "less"]
_MIN_REPORTED_P = 1e-300
_LOG_MIN_REPORTED_P = math.log(_MIN_REPORTED_P)


@dataclass(frozen=True)
class WilsonInterval:
    """Wilson score interval for a binomial proportion."""

    event_n: int
    n: int
    estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    method: str = "Wilson score interval"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly record."""

        return asdict(self)


@dataclass(frozen=True)
class OrderedTrendResult:
    """Common machine-readable result returned by both ordered trend tests."""

    test_name: str
    input_n: int
    n: int
    excluded_n: int
    group_order: tuple[Any, ...]
    group_sizes: tuple[int, ...]
    alternative: Alternative
    statistic: float
    statistic_type: str
    z_statistic: float
    p_value: float
    p_value_reporting: str
    log_p_value: float
    negative_log10_p: float
    p_value_bounded: bool
    effect_size: float
    effect_size_name: str
    implementation: str
    score_scheme: str | None = None
    scores: tuple[float, ...] | None = None
    chi_square: float | None = None
    expected_statistic: float | None = None
    variance: float | None = None
    pair_count: int | None = None
    concordance_minus_discordance: float | None = None
    n_unique_outcomes: int | None = None
    outcome_tie_group_count: int | None = None
    tie_correction: bool = False
    continuity_correction: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly record."""

        payload = asdict(self)
        payload["group_order"] = list(self.group_order)
        payload["group_sizes"] = list(self.group_sizes)
        if self.scores is not None:
            payload["scores"] = list(self.scores)
        return payload


def _validate_alternative(alternative: str) -> Alternative:
    normalised = str(alternative).strip().lower()
    if normalised not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return normalised  # type: ignore[return-value]


def _normal_tail(z_statistic: float, alternative: Alternative) -> dict[str, Any]:
    """Stable normal-tail probability plus a nonzero serialization value."""

    z = float(z_statistic)
    if not math.isfinite(z):
        raise ValueError("z_statistic must be finite")
    if alternative == "two-sided":
        log_p = min(0.0, math.log(2.0) + float(log_ndtr(-abs(z))))
    elif alternative == "greater":
        log_p = float(log_ndtr(-z))
    else:
        log_p = float(log_ndtr(z))

    bounded = log_p < _LOG_MIN_REPORTED_P
    p_value = _MIN_REPORTED_P if bounded else float(math.exp(log_p))
    if bounded:
        reporting = f"<{_MIN_REPORTED_P:.0e}"
    elif p_value < 0.001:
        reporting = f"{p_value:.3e}"
    else:
        reporting = f"{p_value:.6g}"
    return {
        "p_value": p_value,
        "p_value_reporting": reporting,
        "log_p_value": log_p,
        "negative_log10_p": -log_p / math.log(10.0),
        "p_value_bounded": bounded,
    }


def wilson_interval(
    event_n: int,
    n: int,
    *,
    alpha: float = 0.05,
) -> WilsonInterval:
    """Return a two-sided Wilson score interval for ``event_n / n``.

    Unlike a Wald interval, the Wilson interval stays inside ``[0, 1]`` and is
    well behaved at zero events or at all events.  An empty denominator is not
    estimable and therefore raises rather than fabricating a zero-width result.
    """

    if isinstance(event_n, bool) or isinstance(n, bool):
        raise TypeError("event_n and n must be integer counts")
    if int(event_n) != event_n or int(n) != n:
        raise ValueError("event_n and n must be integer counts")
    event_n = int(event_n)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if event_n < 0 or event_n > n:
        raise ValueError("event_n must be between 0 and n")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    z = float(norm.ppf(1.0 - float(alpha) / 2.0))
    estimate = event_n / n
    z2_over_n = z * z / n
    denominator = 1.0 + z2_over_n
    centre = (estimate + z2_over_n / 2.0) / denominator
    half_width = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return WilsonInterval(
        event_n=event_n,
        n=n,
        estimate=float(estimate),
        ci_low=float(max(0.0, centre - half_width)),
        ci_high=float(min(1.0, centre + half_width)),
        alpha=float(alpha),
    )


def cochran_armitage_trend(
    event_counts: Sequence[int],
    totals: Sequence[int],
    *,
    scores: Sequence[float] | None = None,
    group_order: Sequence[Any] | None = None,
    alternative: Alternative = "two-sided",
) -> OrderedTrendResult:
    """Cochran--Armitage score test for a binary ordered-group trend.

    ``event_counts`` and ``totals`` contain one entry per ordered group.  If
    ``scores`` is omitted, consecutive scores ``0, 1, ..., k-1`` are used.  The
    caller must report that spacing assumption; it is not an order-only test.
    """

    alternative = _validate_alternative(alternative)
    events = np.asarray(list(event_counts), dtype=float)
    ns = np.asarray(list(totals), dtype=float)
    if events.ndim != 1 or ns.ndim != 1 or events.size != ns.size:
        raise ValueError("event_counts and totals must be one-dimensional and aligned")
    if events.size < 2:
        raise ValueError("at least two ordered groups are required")
    if not np.all(np.isfinite(events)) or not np.all(np.isfinite(ns)):
        raise ValueError("event_counts and totals must be finite")
    if not np.all(events == np.floor(events)) or not np.all(ns == np.floor(ns)):
        raise ValueError("event_counts and totals must be integer counts")
    if np.any(ns <= 0) or np.any(events < 0) or np.any(events > ns):
        raise ValueError("each total must be positive and contain its event count")

    if scores is None:
        score_values = np.arange(events.size, dtype=float)
        score_scheme = "consecutive_ordinal_ranks"
    else:
        score_values = np.asarray(list(scores), dtype=float)
        score_scheme = (
            "consecutive_ordinal_ranks"
            if np.array_equal(score_values, np.arange(events.size, dtype=float))
            else "prespecified_numeric_scores"
        )
    if score_values.shape != events.shape or not np.all(np.isfinite(score_values)):
        raise ValueError("scores must contain one finite value per ordered group")
    if np.any(np.diff(score_values) <= 0):
        raise ValueError(
            "scores must be strictly increasing in the declared group order"
        )

    if group_order is None:
        order = tuple(range(int(events.size)))
    else:
        order = tuple(group_order)
        if len(order) != events.size or len(set(order)) != len(order):
            raise ValueError("group_order must contain one unique label per group")

    n_total = float(ns.sum())
    event_total = float(events.sum())
    if event_total <= 0.0 or event_total >= n_total:
        raise ValueError("the pooled binary outcome must contain events and non-events")
    pooled_risk = event_total / n_total
    weighted_score_sum = float(np.dot(ns, score_values))
    observed_score_sum = float(np.dot(events, score_values))
    expected_score_sum = pooled_risk * weighted_score_sum
    # The usual Cochran-Armitage null conditions on both table margins.  The
    # N/(N-1) finite-population factor is therefore required; omitting it gives
    # the related unconditional binomial score variance, not the standard CA
    # test implemented by statsmodels/SAS.
    score_variance = (
        pooled_risk
        * (1.0 - pooled_risk)
        * (
            float(np.dot(ns, score_values * score_values))
            - weighted_score_sum * weighted_score_sum / n_total
        )
        * n_total
        / (n_total - 1.0)
    )
    if not math.isfinite(score_variance) or score_variance <= 0.0:
        raise ValueError("Cochran-Armitage variance is zero or non-finite")

    z_statistic = (observed_score_sum - expected_score_sum) / math.sqrt(score_variance)
    tail = _normal_tail(z_statistic, alternative)
    n_int = int(n_total)
    return OrderedTrendResult(
        test_name="Cochran-Armitage trend test",
        input_n=n_int,
        n=n_int,
        excluded_n=0,
        group_order=order,
        group_sizes=tuple(int(value) for value in ns.tolist()),
        alternative=alternative,
        statistic=float(z_statistic),
        statistic_type="z",
        z_statistic=float(z_statistic),
        chi_square=float(z_statistic * z_statistic),
        p_value=float(tail["p_value"]),
        p_value_reporting=str(tail["p_value_reporting"]),
        log_p_value=float(tail["log_p_value"]),
        negative_log10_p=float(tail["negative_log10_p"]),
        p_value_bounded=bool(tail["p_value_bounded"]),
        effect_size=float(z_statistic / math.sqrt(n_int)),
        effect_size_name="signed_z_over_sqrt_n",
        expected_statistic=float(expected_score_sum),
        variance=float(score_variance),
        implementation="easyicu.methods.ordered_trends.cochran_armitage_trend.v1",
        score_scheme=score_scheme,
        scores=tuple(float(value) for value in score_values.tolist()),
    )


def _is_numeric_label(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _resolve_group_order(
    observed_groups: list[Any], group_order: Sequence[Any] | None
) -> tuple[Any, ...]:
    unique_observed = list(dict.fromkeys(observed_groups))
    if group_order is None:
        if not all(_is_numeric_label(value) for value in unique_observed):
            raise ValueError(
                "group_order is required for non-numeric group labels; "
                "alphabetical order is not a scientific ordering"
            )
        return tuple(sorted(unique_observed, key=float))

    order = tuple(group_order)
    if len(order) < 2 or len(set(order)) != len(order):
        raise ValueError("group_order must contain at least two unique labels")
    missing = [value for value in unique_observed if value not in order]
    empty = [value for value in order if value not in unique_observed]
    if missing or empty:
        raise ValueError(
            "group_order must match the observed groups exactly; "
            f"missing={missing!r}, empty={empty!r}"
        )
    return order


def _falling_factorial_2(counts: np.ndarray) -> float:
    return float(np.sum(counts * (counts - 1.0)))


def _falling_factorial_3(counts: np.ndarray) -> float:
    return float(np.sum(counts * (counts - 1.0) * (counts - 2.0)))


def jonckheere_terpstra_trend(
    values: Sequence[float],
    groups: Sequence[Any],
    *,
    group_order: Sequence[Any] | None = None,
    alternative: Alternative = "two-sided",
) -> OrderedTrendResult:
    """Tie-corrected asymptotic Jonckheere--Terpstra ordered trend test.

    Missing/non-finite outcomes and missing group labels are removed pairwise.
    String labels require an explicit ``group_order``.  Numeric labels may be
    sorted numerically when no order is supplied, but callers are encouraged to
    pass the design order explicitly.  The returned ``J`` statistic counts an
    earlier-group value below a later-group value as one and a cross-group tie
    as one half.
    """

    alternative = _validate_alternative(alternative)
    raw_values = list(values)
    raw_groups = list(groups)
    if len(raw_values) != len(raw_groups):
        raise ValueError("values and groups must share length")
    if not raw_values:
        raise ValueError("values and groups are empty")

    clean_values: list[float] = []
    clean_groups: list[Any] = []
    for raw_value, raw_group in zip(raw_values, raw_groups):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        group_missing = raw_group is None
        if not group_missing:
            try:
                group_missing = bool(np.asarray(raw_group != raw_group).item())
            except (TypeError, ValueError):
                group_missing = False
        if math.isfinite(value) and not group_missing:
            clean_values.append(value)
            clean_groups.append(raw_group)

    if len(clean_values) < 2:
        raise ValueError("fewer than two finite observations remain")
    order = _resolve_group_order(clean_groups, group_order)
    grouped = [
        np.sort(
            np.asarray(
                [
                    value
                    for value, group in zip(clean_values, clean_groups)
                    if group == label
                ],
                dtype=float,
            )
        )
        for label in order
    ]
    group_sizes = np.asarray([len(group) for group in grouped], dtype=float)
    if len(grouped) < 2 or np.any(group_sizes <= 0):
        raise ValueError("at least two non-empty ordered groups are required")

    # J = sum over earlier/later groups of wins + half cross-group ties.
    # Search in each sorted earlier group for every later observation, avoiding
    # an O(n^2) materialisation of all pairs.
    j_statistic = 0.0
    for earlier_index, earlier_values in enumerate(grouped[:-1]):
        for later_values in grouped[earlier_index + 1 :]:
            below = np.searchsorted(earlier_values, later_values, side="left")
            at_or_below = np.searchsorted(earlier_values, later_values, side="right")
            j_statistic += float(np.sum(below + 0.5 * (at_or_below - below)))

    n = len(clean_values)
    pair_count_float = (float(n * n) - float(np.dot(group_sizes, group_sizes))) / 2.0
    pair_count = int(round(pair_count_float))
    expected_j = pair_count / 2.0
    t_statistic = 2.0 * j_statistic - pair_count

    _, tie_counts_int = np.unique(
        np.asarray(clean_values, dtype=float), return_counts=True
    )
    tie_counts = tie_counts_int.astype(float)
    n2 = float(n * (n - 1))
    group2 = _falling_factorial_2(group_sizes)
    tie2 = _falling_factorial_2(tie_counts)
    variance_j = (n2 - group2) * (n2 - tie2) / (8.0 * n2)
    if n >= 3:
        n3 = float(n * (n - 1) * (n - 2))
        group3 = _falling_factorial_3(group_sizes)
        tie3 = _falling_factorial_3(tie_counts)
        variance_j += (n3 - group3) * (n3 - tie3) / (36.0 * n3)
    if not math.isfinite(variance_j) or variance_j <= 0.0:
        raise ValueError(
            "Jonckheere-Terpstra variance is zero or non-finite; "
            "the ordered trend is not estimable"
        )

    z_statistic = t_statistic / math.sqrt(4.0 * variance_j)
    tail = _normal_tail(z_statistic, alternative)
    return OrderedTrendResult(
        test_name="Jonckheere-Terpstra trend test",
        input_n=len(raw_values),
        n=n,
        excluded_n=len(raw_values) - n,
        group_order=order,
        group_sizes=tuple(int(value) for value in group_sizes.tolist()),
        alternative=alternative,
        statistic=float(j_statistic),
        statistic_type="J",
        z_statistic=float(z_statistic),
        p_value=float(tail["p_value"]),
        p_value_reporting=str(tail["p_value_reporting"]),
        log_p_value=float(tail["log_p_value"]),
        negative_log10_p=float(tail["negative_log10_p"]),
        p_value_bounded=bool(tail["p_value_bounded"]),
        effect_size=float(t_statistic / pair_count),
        effect_size_name="cross_group_concordance_minus_discordance",
        expected_statistic=float(expected_j),
        variance=float(variance_j),
        pair_count=pair_count,
        concordance_minus_discordance=float(t_statistic),
        n_unique_outcomes=int(len(tie_counts)),
        outcome_tie_group_count=int(np.sum(tie_counts > 1)),
        tie_correction=True,
        continuity_correction=False,
        implementation="easyicu.methods.ordered_trends.jonckheere_terpstra_trend.v1",
    )


__all__ = [
    "OrderedTrendResult",
    "WilsonInterval",
    "cochran_armitage_trend",
    "jonckheere_terpstra_trend",
    "wilson_interval",
]
